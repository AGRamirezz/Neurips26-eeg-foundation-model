# NeurIPS 2026 — EEG Foundation Model Challenge

Work toward the [EEG/EMG Foundation Challenge 2026](https://neural-interfaces26.github.io/),
run at the NeurIPS Brain & Body Workshop. The challenge asks for biosignal decoding
models that generalize beyond the conditions they were trained on, across four tracks.

## The tracks

| # | Track | Generalization shift | Input | Metric |
|---|-------|----------------------|-------|--------|
| 01 | EEG-to-Image | Cross-stimulus | 32-ch EEG epoch | Top-5 retrieval accuracy |
| 02 | BCI decoding | Cross-session | 64-ch EEG, 3 cued commands | Balanced accuracy |
| 03 | Sleep onset | Cross-user | 4-ch wearable EEG at home | Binned MAE (seconds) |
| 04 | EMG-to-Pose | Cross-user | 16-ch wrist sEMG | Mean angular error (degrees) |

Warm-up phase runs Sep 21 – Oct 25 2026; the sealed final phase runs Oct 25 – Nov 21 2026.

## Approach

Start from publicly pretrained EEG encoders rather than pretraining from scratch, adapt
them with parameter-efficient fine-tuning, and validate locally against the competition's
own metrics before submitting. The first milestone is a verified pipeline per track —
data in, model trained, evaluation out, submission packaged — with local scores that track
the official ones closely enough to avoid surprises on the leaderboard.

## Repository layout

```
notebooks/                  track<N>_<step>_<name>, or common_ where it serves every track
  common_acquire.ipynb      small fetch per track into Drive, verified, with a manifest
  track1_01_prepare.ipynb   pull the EEG and build the frozen image-embedding target table
  track1_02_features.ipynb  epoch, run the frozen REVE encoder once, cache features
src/
  tracks.py     per-track config: task, metric, expected channels and rate, references
  acquire.py    verification checks and the on-disk manifest
  datastore.py  inventory of what is already downloaded, and top-up planning
  miniload.py   byte-budget record selection, and the run-based split track 1 needs
  subset.py     split-aware subsetting that preserves each track's held-out axis
  harness.py    local stand-in for the competition harness, plus the four track metrics
```

Work runs as two independent streams: **A** builds the data path, **B** verifies the submission
path. B does not depend on A. Stream B is done: all four tracks have a scored submission on the
leaderboard, each sitting at its floor by design, which proves the submission contract end to end.

Data lives outside the repo, in a Google Drive folder with one directory per track plus a shared
folder for model checkpoints.

## Data and licensing

No datasets or model weights are redistributed here. Each corpus keeps its original
license and is obtained from its own source. Some are access-gated or non-commercial —
check the terms of any dataset or checkpoint before using it.

## Status

Setting up. Track pipelines in progress.
