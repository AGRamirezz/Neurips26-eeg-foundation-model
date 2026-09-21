"""A persistent data store on Drive, plus an inventory of what is already in it.

Colab wipes `/content` between sessions, so anything cached there is re-fetched
every run. That is the difference between a smoke test that starts in seconds and
one that spends ten minutes re-downloading data it already had yesterday.

The rule this module encodes: **anything expensive to obtain lives on Drive**;
only genuinely scratch work stays local. A prepare cache looks like a good
candidate for local disk right up until you remember that rebuilding Track 1's
means running DINOv2 over every stimulus again.

Nothing here downloads. It reports what exists and what a request would add, so
the decision to fetch is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# NeMAR accession -> human name, for readable inventories.
KNOWN = {
    "nm000133": "Alljoined-1 (track 1)",
    "nm000134": "Alljoined-1.6M (track 1)",
    "nm000135": "BNCI2014-004 (track 2 warm-up)",
    "nm000185": "Sleep-EDF (track 3)",
    "nm000232": "THINGS-EEG2 (track 1)",
    "nm000281": "emg2pose (track 4)",
}

SIGNAL_SUFFIXES = {".edf", ".bdf", ".fif", ".set", ".vhdr", ".eeg", ".cnt", ".nwb"}


@dataclass
class Entry:
    dataset: str
    name: str
    n_signal_files: int
    n_files: int
    bytes: int

    @property
    def mb(self) -> float:
        return self.bytes / 1e6

    @property
    def subjects(self) -> int:
        return self._subjects

    def __post_init__(self) -> None:
        self._subjects = 0


def inventory(root: Path) -> list[Entry]:
    """What is already on disk, one row per dataset directory."""
    root = Path(root)
    if not root.exists():
        return []

    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f for f in d.rglob("*") if f.is_file()]
        if not files:
            continue
        entry = Entry(
            dataset=d.name,
            name=KNOWN.get(d.name, "unrecognised"),
            n_signal_files=sum(1 for f in files if f.suffix.lower() in SIGNAL_SUFFIXES),
            n_files=len(files),
            bytes=sum(f.stat().st_size for f in files),
        )
        entry._subjects = len({p.name for p in d.rglob("sub-*") if p.is_dir()})
        out.append(entry)
    return out


def format_report(entries: list[Entry], root: Path) -> str:
    if not entries:
        return f"data store at {root}\n  (empty)"
    width = max(len(e.dataset) for e in entries)
    lines = [f"data store at {root}", ""]
    lines.append(f"  {'dataset'.ljust(width)}  {'subjects':>8} {'recordings':>10} {'MB':>9}  name")
    for e in entries:
        lines.append(
            f"  {e.dataset.ljust(width)}  {e.subjects:>8} {e.n_signal_files:>10} "
            f"{e.mb:>9.1f}  {e.name}"
        )
    lines.append("")
    lines.append(f"  total {sum(e.mb for e in entries):.1f} MB across {len(entries)} datasets")
    return "\n".join(lines)


def held(entries: list[Entry], dataset: str) -> Entry | None:
    return next((e for e in entries if e.dataset == dataset), None)


def plan(entries: list[Entry], dataset: str, want_mb: float) -> tuple[str, float]:
    """Decide what a request for `want_mb` of `dataset` actually implies.

    Returns an action and the megabytes it would add, so a caller can print the
    consequence before committing to it.
    """
    have = held(entries, dataset)
    if have is None:
        return "fetch", want_mb
    if have.mb >= want_mb:
        return "reuse", 0.0
    return "extend", want_mb - have.mb


def choose(entries: list[Entry], dataset: str, want_mb: float, ask=input) -> tuple[str, float]:
    """Print the situation and let the caller decide. Falls back to reuse/fetch.

    Deliberately not automatic: re-downloading is the slow path, so it should be a
    choice rather than a side effect of running a cell.
    """
    action, delta = plan(entries, dataset, want_mb)
    have = held(entries, dataset)

    if action == "reuse":
        print(f"{dataset}: {have.mb:.1f} MB already on disk, enough for {want_mb:.0f} MB request")
        print("  [r] reuse as-is (default)   [a] add more   [f] refetch from scratch")
        try:
            answer = (ask("  choice [r]: ").strip().lower() or "r")
        except (EOFError, OSError):
            answer = "r"
        if answer.startswith("a"):
            return "extend", want_mb
        if answer.startswith("f"):
            return "fetch", want_mb
        return "reuse", 0.0

    if action == "extend":
        print(f"{dataset}: {have.mb:.1f} MB on disk, request wants {want_mb:.0f} MB")
        print(f"  continuing would add ~{delta:.0f} MB")
    else:
        print(f"{dataset}: nothing on disk, would fetch ~{delta:.0f} MB")
    return action, delta
