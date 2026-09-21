"""Per-track configuration, so one notebook can run any of the four.

Every value here is taken from the challenge site, the NeuralBench docs, or the
`neuralbench --help` output captured on 2026-09-18. Where a number is a published
reference it says which model and dataset it was measured on, because a reference
measured on a corpus you are not training on is not a target.

`eegdash_id` is the NeMAR accession for cheap record-level selection. NeuralBench
downloads whole corpora; EEGDash does not, which is the only way to touch the large
tracks from a Colab disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Track:
    key: str
    name: str
    modality: str                 # neuralbench positional 1
    task: str                     # neuralbench positional 2
    shift: str
    metric: str
    metric_key: str
    higher_is_better: bool
    output_shape: str             # what predict() must return
    smallest_dataset: str         # --dataset value, lowercase
    smallest_gb: float
    eegdash_id: str | None
    references: dict[str, float]  # published, on the dataset named below
    reference_dataset: str
    prize: str
    slug: str                     # folder name under the Drive root
    expect_sfreq: float | None    # None where we have not verified it
    expect_n_chans: int | None
    exclude_sessions: tuple[str, ...] = ()
    target_kind: str = ""         # what the label looks like, for inspection
    notes: str = ""
    caveats: list[str] = field(default_factory=list)


TRACKS: dict[str, Track] = {
    "1": Track(
        key="1",
        slug="track1_image",
        expect_sfreq=256.0,
        expect_n_chans=32,
        exclude_sessions=("02old",),
        target_kind="image identity -> frozen DINOv2 embedding",
        name="EEG-to-Image",
        modality="eeg",
        task="image",
        shift="cross-stimulus",
        metric="Top-5 retrieval accuracy (%)",
        metric_key="test/full_retrieval/top5_acc_subject-agg",
        higher_is_better=True,
        output_shape="(B, 1536)  DINOv2-giant embedding",
        smallest_dataset="xu2024alljoined",
        smallest_gb=4.4,
        eegdash_id="nm000134",           # Alljoined-1.6M; Alljoined-1 is nm000133
        references={"chance": 2.22, "eegnet": 28.13, "reve": 84.75},
        reference_dataset="Gifford2022Large (THINGS-EEG2)",
        prize="no cash; SF internship + cross-track award",
        notes="Only track whose --prepare needs a GPU: it runs DINOv2 over every stimulus.",
        caveats=[
            "REVE's reference is leakage-contaminated: it was pretrained on THINGS-EEG2.",
            "Target space is fixed to dinov2-giant depth 0.6667, mean-pooled. Do not swap for CLIP.",
            "val/batch_top5_acc ranks within a batch and is not comparable to the score.",
        ],
    ),
    "2": Track(
        key="2",
        slug="track2_bci",
        expect_sfreq=None,
        expect_n_chans=None,
        exclude_sessions=(),
        target_kind="one of three cued mental commands",
        name="BCI decoding",
        modality="eeg",
        task="motor_imagery",
        shift="cross-session",
        metric="Balanced accuracy (%)",
        metric_key="test/balanced_accuracy",
        higher_is_better=True,
        output_shape="(B,)  class labels",
        smallest_dataset="tangermann2012",
        smallest_gb=1.0,
        eegdash_id="nm000135",           # BNCI2014-004, the EEGDash warm-up pair
        references={"chance": 24.81, "eegnet": 58.58, "reve": 68.04},
        reference_dataset="Stieger2021Continuous",
        prize="$2,000",
        notes="Eval subjects give labelled calibration sessions 1-3, so per-session "
              "Riemannian recentring is a cheap, strong floor.",
        caveats=[
            "The 2026 corpus is Graz/BrainHero, released Sep 21; tangermann2012 is a stand-in.",
            "Stieger is 940 GB. Never omit --dataset.",
        ],
    ),
    "3": Track(
        key="3",
        slug="track3_sleep",
        expect_sfreq=100.0,
        expect_n_chans=None,
        exclude_sessions=(),
        target_kind="seconds to first stable N2",
        name="Sleep onset",
        modality="eeg",
        task="sleep_onset",
        shift="cross-user",
        metric="Binned MAE (seconds, lower better)",
        metric_key="test/bmae",
        higher_is_better=False,
        output_shape="(B,)  seconds to first stable N2",
        smallest_dataset="kemp2000analysis",
        smallest_gb=8.0,
        eegdash_id="nm000185",           # Sleep-EDF
        references={"chance": 205.42, "eegnet": 143.30, "reve": 134.89},
        reference_dataset="Kemp2000Analysis (Sleep-EDF)",
        prize="$2,000 or remote Muse internship",
        notes="Cheapest track end to end (~6-8 min/seed) and the only one whose published "
              "reference is measured on the dataset we can actually train on. Strongest G1.",
        caveats=[
            "NeuralBench takes the earliest annotated N2 with no persistence rule; the "
            "competition says *stable* N2. Target definition, and it could move bMAE.",
            "bMAE weights four time-to-onset bins equally, so long horizons count as much "
            "as short ones. Sampler and loss matter more than architecture here.",
        ],
    ),
    "4": Track(
        key="4",
        slug="track4_pose",
        expect_sfreq=2000.0,
        expect_n_chans=16,
        exclude_sessions=(),
        target_kind="20 joint angles, MISC channels in the same file",
        name="EMG-to-Pose",
        modality="emg",
        task="pose",
        shift="cross-user",
        metric="Mean angular error (degrees, lower better)",
        metric_key="test/mae",
        higher_is_better=False,
        output_shape="(B, 20, T)  joint-angle trajectories",
        smallest_dataset="salter2024emg2pose",
        smallest_gb=330.0,
        eegdash_id="nm000281",           # emg2pose
        references={"neuropose": 17.5},
        reference_dataset="Salter2024Emg2pose",
        prize="$2,000",
        notes="770 GB for the full corpus at 2.08 GB per recording-hour. Subset by trimming "
              "hours per subject, not subject count: the shift is cross-user, so subject "
              "diversity is the axis under test.",
        caveats=[
            "Pass -m vemg2pose to every command including --download and --prepare; the "
            "model config widens data.duration to 5.895 s for TDS left context.",
            "test/mae is radians. Multiply by 57.29578 for degrees.",
            "Do not downsample targets to shrink the cache: scoring expects 10,000 frames.",
            "emg2pose is CC-BY-NC-SA-4.0, UmeTrack CC-BY-NC-4.0. Non-commercial.",
        ],
    ),
}


def get(key: str) -> Track:
    key = str(key).strip()
    if key not in TRACKS:
        raise KeyError(f"no track {key!r}; choose from {sorted(TRACKS)}")
    return TRACKS[key]


def cli(track: Track, *args: str) -> str:
    """The neuralbench invocation for this track, with --dataset always set.

    Track 4 needs `-m vemg2pose` on every subcommand, but only when the caller has
    not already chosen a model: two -m flags is a different command, not a stronger
    one.
    """
    parts = ["neuralbench", track.modality, track.task, "--dataset", track.smallest_dataset]
    args = list(args)
    if track.key == "4" and not any(a in ("-m", "--model") for a in args):
        parts += ["-m", "vemg2pose"]
    return " ".join(parts + args)


def summary(track: Track) -> str:
    refs = "  ".join(f"{k} {v}" for k, v in track.references.items())
    lines = [
        f"Track {track.key} - {track.name}   ({track.shift})",
        f"  metric    {track.metric}   [{track.metric_key}]",
        f"  predict   {track.output_shape}",
        f"  dataset   {track.smallest_dataset}  (~{track.smallest_gb:g} GB)",
        f"  published {refs}   on {track.reference_dataset}",
        f"  prize     {track.prize}",
    ]
    if track.notes:
        lines.append(f"  note      {track.notes}")
    for c in track.caveats:
        lines.append(f"  caveat    {c}")
    return "\n".join(lines)


def paths(root, track: Track) -> dict:
    """Per-track folders under the Drive root, created on demand.

    `raw` holds what EEGDash fetched, `prepared` the windowed arrays, and weights
    and submissions stay apart so a bad training run cannot overwrite something
    already sent.
    """
    from pathlib import Path

    base = Path(root) / track.slug
    out = {
        "base": base,
        "raw": base / "raw",
        "prepared": base / "prepared",
        "weights": base / "weights",
        "submissions": base / "submissions",
        "manifest": base / "manifest.json",
    }
    for key, value in out.items():
        if key != "manifest":
            value.mkdir(parents=True, exist_ok=True)
    return out


def shared_paths(root) -> dict:
    """Checkpoints and logs shared across tracks.

    DINOv2 is 4.5 GB and REVE is used by three tracks, so per-track copies would
    waste more Drive than the datasets do.
    """
    from pathlib import Path

    base = Path(root) / "shared"
    out = {"base": base, "hf": base / "hf", "logs": base / "logs"}
    for value in out.values():
        value.mkdir(parents=True, exist_ok=True)
    return out
