"""Verify downloaded data is what the track expects, and record what was taken.

Two jobs, both cheap and both easy to skip until they cost you a day.

Verification answers "is this the data we think it is" with per-track assertions
rather than a vibe: channel count, sampling rate, whether targets are present at
all. Expectations we have not confirmed are recorded as `unknown` instead of being
asserted, because a check that passes by accident is worse than no check.

The manifest is a small JSON beside the data saying what was fetched, when, and
whether it verified. Later scripts read it instead of re-deriving the state of the
disk, and Rule 4 wants the provenance anyway.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

MANIFEST_VERSION = 1


@dataclass
class Check:
    name: str
    status: str          # "ok" | "fail" | "unknown"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def __str__(self) -> str:
        mark = {"ok": "ok  ", "fail": "FAIL", "unknown": "?   "}[self.status]
        return f"  [{mark}] {self.name}{': ' + self.detail if self.detail else ''}"


def _cmp(name: str, got, expected, unit: str = "") -> Check:
    if expected is None:
        return Check(name, "unknown", f"got {got}{unit}, no verified expectation")
    if got == expected:
        return Check(name, "ok", f"{got}{unit}")
    return Check(name, "fail", f"got {got}{unit}, expected {expected}{unit}")


def verify_records(records: Sequence[dict[str, Any]], track) -> list[Check]:
    """Checks that run on record metadata, before anything downloads."""
    checks = [Check("records found", "ok" if records else "fail", f"{len(records)} records")]
    if not records:
        return checks

    sfreqs = {r.get("sampling_frequency") for r in records if r.get("sampling_frequency")}
    nchans = {r.get("nchans") for r in records if r.get("nchans")}
    checks.append(_cmp("sampling rate", sfreqs.pop() if len(sfreqs) == 1 else sorted(sfreqs),
                       track.expect_sfreq, " Hz"))
    checks.append(_cmp("channel count", nchans.pop() if len(nchans) == 1 else sorted(nchans),
                       track.expect_n_chans))

    incomplete = sum(1 for r in records if r.get("_has_missing_files"))
    checks.append(Check("complete files", "ok" if incomplete == 0 else "fail",
                        f"{incomplete} records flagged with missing files"))

    if track.exclude_sessions:
        present = {r.get("session") for r in records} & set(track.exclude_sessions)
        checks.append(Check("excluded sessions absent", "ok" if not present else "fail",
                            f"found {sorted(present)}" if present else
                            f"none of {list(track.exclude_sessions)}"))

    subjects = {r.get("subject") for r in records}
    checks.append(Check("subjects", "ok", f"{len(subjects)} distinct"))
    return checks


def verify_recording(raw, track) -> list[Check]:
    """Checks that need the signal itself. `raw` is an MNE Raw."""
    checks = [
        _cmp("sampling rate", float(raw.info["sfreq"]), track.expect_sfreq, " Hz"),
        _cmp("channel count", len(raw.ch_names), track.expect_n_chans),
    ]

    montage = raw.get_montage()
    checks.append(Check(
        "montage present", "ok" if montage is not None else "fail",
        "set a standard montage before using a position-aware encoder"
        if montage is None else "",
    ))

    n_ann = len(raw.annotations)
    checks.append(Check("annotations", "ok" if n_ann else "fail", f"{n_ann} events"))

    misc = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "misc"]
    if track.key == "4":
        checks.append(_cmp("joint-angle channels", len(misc), 20))
    elif misc:
        checks.append(Check("misc channels", "ok", f"{len(misc)} present"))

    return checks


def format_checks(checks: Sequence[Check], title: str = "") -> str:
    lines = [title] if title else []
    lines += [str(c) for c in checks]
    n_fail = sum(1 for c in checks if c.status == "fail")
    n_unknown = sum(1 for c in checks if c.status == "unknown")
    lines.append(f"  -> {len(checks) - n_fail - n_unknown} ok, {n_fail} failed, "
                 f"{n_unknown} unverifiable")
    return "\n".join(lines)


@dataclass
class Manifest:
    track: str
    slug: str
    dataset: str
    fetched_at: str
    n_records: int
    n_subjects: int
    mb: float
    checks: list[dict] = field(default_factory=list)
    records: list[str] = field(default_factory=list)
    version: int = MANIFEST_VERSION
    notes: str = ""

    @property
    def verified(self) -> bool:
        return all(c["status"] != "fail" for c in self.checks)


def build_manifest(track, records: Sequence[dict], checks: Sequence[Check],
                   mb: float, notes: str = "") -> Manifest:
    return Manifest(
        track=track.key,
        slug=track.slug,
        dataset=track.eegdash_id or track.smallest_dataset,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_records=len(records),
        n_subjects=len({r.get("subject") for r in records}),
        mb=round(mb, 1),
        checks=[asdict(c) for c in checks],
        records=[r.get("bids_relpath", "") for r in records],
        notes=notes,
    )


def write_manifest(path: Path, manifest: Manifest) -> None:
    Path(path).write_text(json.dumps(asdict(manifest), indent=2))


def read_manifest(path: Path) -> Manifest | None:
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data.pop("version", None)
    return Manifest(**data, version=MANIFEST_VERSION)


def overview(root: Path, all_tracks: dict) -> str:
    """One line per track: what is on disk and whether it verified."""
    lines = [f"{'track':<14} {'records':>7} {'subj':>5} {'MB':>8}  status"]
    for track in all_tracks.values():
        m = read_manifest(Path(root) / track.slug / "manifest.json")
        if m is None:
            lines.append(f"{track.slug:<14} {'-':>7} {'-':>5} {'-':>8}  empty")
        else:
            status = "verified" if m.verified else "CHECKS FAILED"
            lines.append(f"{track.slug:<14} {m.n_records:>7} {m.n_subjects:>5} "
                         f"{m.mb:>8.1f}  {status} ({m.fetched_at[:10]})")
    return "\n".join(lines)
