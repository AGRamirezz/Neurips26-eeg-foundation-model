"""Deterministic, split-aware subsetting for pipeline-proof runs.

Two independent levers, because they solve different problems:

1. Record selection (before download). Picking exact recordings is the only way to
   touch a 220 GB corpus from a Colab disk. EEGDash queries records directly, so a
   single recording is a few hundred MB rather than the whole release.

2. Group subsetting (after download). A deterministic fraction of whatever is on
   disk, sampled over *groups* rather than windows so that split semantics survive.

The grouping key is what makes this safe. Each track is scored on a specific shift,
and the unit of that shift is the thing a subset must not break:

    track 1  image identity   train and test images must stay disjoint
    track 2  subject/session  sessions are the held-out axis
    track 3  subject          sleepers are the held-out axis
    track 4  user/stage       both are held-out axes

Sampling windows at random would silently leak across every one of those.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

import numpy as np

# Grouping key per track: the unit a subset is allowed to drop whole.
GROUP_KEY = {
    "track1": "subject",      # see warning below - NOT image_id
    "track2": "session",
    "track3": "subject",
    "track4": "user_stage",
}

# Track 1 is a special case and getting it wrong invalidates the score.
#
# The metric is retrieval against the held-out gallery, so chance = k / gallery_size.
# Subsetting by image identity shrinks the gallery and inflates the score for reasons
# that have nothing to do with the model: a 2% subset of 500 images leaves a gallery
# of 10, where chance top-5 is 50%.
#
# So on track 1, subset the TRAINING split only - by subject, or by trial repetitions
# per image - and leave the test gallery intact. Reduce how much EEG the model learns
# from, never how many candidates it must rank against.
TRACK1_RULE = "subset train only; keep the test gallery whole"


@dataclass
class SubsetSpec:
    """How much data to use, and how to slice it."""

    fraction: float = 0.02
    seed: int = 33
    group_key: str = "image_id"
    min_groups: int = 2
    enabled: bool = True
    notes: str = field(default="")

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

    @property
    def is_full(self) -> bool:
        return not self.enabled or self.fraction == 1.0


def subset_groups(
    group_ids: Sequence[Hashable],
    spec: SubsetSpec,
) -> np.ndarray:
    """Return a boolean mask keeping a deterministic fraction of whole groups.

    Selection is by stable hash of the group id, not by shuffle, so the same group
    is kept or dropped regardless of ordering, dataset version, or how many other
    groups exist. That means a 2% run and a later 10% run are nested rather than
    disjoint, and a cache built for one is reusable by the other.
    """
    ids = np.asarray(group_ids)
    if spec.is_full:
        return np.ones(len(ids), dtype=bool)

    unique = np.unique(ids)
    n_keep = max(spec.min_groups, int(round(len(unique) * spec.fraction)))
    n_keep = min(n_keep, len(unique))

    scored = sorted(unique, key=lambda g: _stable_hash(g, spec.seed))
    keep = set(scored[:n_keep])
    return np.fromiter((i in keep for i in ids), dtype=bool, count=len(ids))


def _stable_hash(value: Hashable, seed: int) -> int:
    payload = f"{seed}:{value}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def chance_top_k(gallery_size: int, k: int = 5) -> float:
    """Chance accuracy (%) for top-k retrieval against a gallery of this size.

    Track 1's score is meaningless without the gallery size beside it. Top-5 against
    200 candidates is 2.5% at chance; against 16,740 it is 0.03%. Subsetting shrinks
    the gallery, which inflates the score for reasons that have nothing to do with
    the model, so every reported number needs its gallery size attached.
    """
    if gallery_size <= 0:
        raise ValueError("gallery_size must be positive")
    return 100.0 * min(k, gallery_size) / gallery_size


def describe(
    group_ids: Sequence[Hashable],
    mask: np.ndarray,
    spec: SubsetSpec,
    gallery_size: int | None = None,
) -> str:
    """One-line provenance for a run record. Never report a score without this."""
    ids = np.asarray(group_ids)
    kept_groups = len(np.unique(ids[mask]))
    total_groups = len(np.unique(ids))
    lines = [
        f"subset: {mask.sum():,}/{len(mask):,} samples "
        f"({kept_groups}/{total_groups} {spec.group_key} groups, "
        f"fraction={spec.fraction:g}, seed={spec.seed})",
    ]
    if gallery_size is not None:
        lines.append(
            f"gallery: {gallery_size:,} candidates "
            f"(chance top-5 = {chance_top_k(gallery_size):.2f}%)"
        )
    if spec.is_full:
        lines.append("FULL RUN - no subsetting applied")
    else:
        lines.append("PIPELINE PROOF - scores are not comparable to full-data runs")
    return "\n".join(lines)


def assert_disjoint(train_groups: Iterable[Hashable], test_groups: Iterable[Hashable]) -> None:
    """Fail loudly if a subset broke the split the track is scored on."""
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        sample = list(overlap)[:5]
        raise AssertionError(
            f"{len(overlap)} groups appear in both train and test (e.g. {sample}). "
            "The subset broke the split."
        )


def assert_gallery_intact(gallery_size: int, expected: int) -> None:
    """Track 1 guard: the test gallery must survive subsetting untouched.

    Shrinking the gallery raises the score without improving the model, so a run
    whose gallery differs from the full-data gallery cannot be compared to anything.
    """
    if gallery_size != expected:
        raise AssertionError(
            f"gallery is {gallery_size:,}, expected {expected:,}. Subsetting reached the "
            f"test split: chance top-5 moved from {chance_top_k(expected):.3f}% to "
            f"{chance_top_k(gallery_size):.3f}%. {TRACK1_RULE}."
        )
