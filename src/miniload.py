"""Load a few recordings at a time, under an explicit byte budget.

NeuralBench downloads whole corpora. EEGDash queries individual records, and the
record metadata carries `ntimes` and `nchans`, so the size of a selection is known
before anything transfers. That is what makes an end-to-end run possible on tens of
megabytes instead of tens of gigabytes.

Budget first, download second.
"""

from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

BYTES_PER_SAMPLE = 4  # float32


def estimate_mb(record: dict[str, Any]) -> float:
    """Uncompressed in-memory size of one recording, from metadata alone."""
    n = record.get("ntimes") or 0
    c = record.get("nchans") or 0
    return n * c * BYTES_PER_SAMPLE / 1e6


@dataclass
class Selection:
    records: list[dict[str, Any]]
    est_mb: float
    skipped_over_budget: int

    def __len__(self) -> int:
        return len(self.records)


def select_under_budget(
    records: Sequence[dict[str, Any]],
    budget_mb: float = 100.0,
    max_records: int | None = None,
    exclude_sessions: Iterable[str] = ("02old",),
    require_complete: bool = True,
    spread_by: str | None = "subject",
) -> Selection:
    """Take records until the budget is reached, spread across `spread_by` first.

    Filters first (missing files, superseded sessions), then accumulates. A record
    larger than the remaining budget is skipped rather than ending the selection, so
    one outlier does not truncate an otherwise fine batch.

    `spread_by` matters more than it looks. Record lists arrive grouped by subject, so
    taking the first N in order yields N recordings from one person: enough to prove
    the code runs, useless for seeing whether it handles variation. Interleaving costs
    the same bytes.
    """
    exclude = set(exclude_sessions)
    pool = [
        r for r in records
        if not (require_complete and r.get("_has_missing_files"))
        and r.get("session") not in exclude
    ]
    if spread_by is not None:
        pool = _interleave(pool, spread_by)

    picked: list[dict[str, Any]] = []
    total = 0.0
    skipped = 0
    for r in pool:
        size = estimate_mb(r)
        if total + size > budget_mb:
            skipped += 1
            continue
        picked.append(r)
        total += size
        if max_records is not None and len(picked) >= max_records:
            break

    return Selection(records=picked, est_mb=total, skipped_over_budget=skipped)


def _interleave(records: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Round-robin the records across distinct values of `field`."""
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for r in records:
        buckets.setdefault(r.get(field), []).append(r)

    out: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda k: (k is None, k))
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop(0))
    return out


def rss_mb() -> float:
    """Resident set size of this process, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def host_limits() -> dict[str, float]:
    """RAM, free disk, and GPU memory, so a budget can be set against real numbers."""
    import shutil

    out: dict[str, float] = {}
    try:
        out["ram_total_gb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    du = shutil.disk_usage("/")
    out["disk_free_gb"] = du.free / 1e9
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            out["gpu_name"] = torch.cuda.get_device_name(0)  # type: ignore[assignment]
    except ImportError:
        pass
    return out


def epochs_mb(n_epochs: int, n_chans: int, n_times: int) -> float:
    """Size of an epoched array. The thing that actually grows, unlike the raw files."""
    return n_epochs * n_chans * n_times * BYTES_PER_SAMPLE / 1e6


def targets_mb(n_stimuli: int, dim: int = 1536) -> float:
    """Size of the frozen image embeddings. One per unique stimulus, not per trial."""
    return n_stimuli * dim * BYTES_PER_SAMPLE / 1e6
