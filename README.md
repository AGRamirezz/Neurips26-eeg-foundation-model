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
notebooks/
  A1_acquire.ipynb    fetch a small amount of real data per track, verify it, write a manifest
  A2_inspect.ipynb    confirm metadata and labels are usable; plot signal against target
  A3_train.ipynb      train per track, save weights
  A4_evaluate.ipynb   local evaluation and submission packaging
  B1_baseline_submit.ipynb   untrained and pretrained baselines, to verify the submission path
src/
  tracks.py     per-track config: task, metric, expected channels and rate, references
  acquire.py    verification checks and the on-disk manifest
  datastore.py  inventory of what is already downloaded, and top-up planning
  miniload.py   byte-budget record selection, spread across subjects
  subset.py     split-aware subsetting that preserves each track's held-out axis
  harness.py    local stand-in for the competition harness, plus the four track metrics
```

Work runs as two independent streams: **A** builds the data path, **B** verifies the submission
path. B does not depend on A, which matters because A is currently blocked on upstream bugs.

Data lives outside the repo, in a Google Drive folder with one directory per track plus a shared
folder for model checkpoints.

## Data and licensing

No datasets or model weights are redistributed here. Each corpus keeps its original
license and is obtained from its own source. Some are access-gated or non-commercial —
check the terms of any dataset or checkpoint before using it.

## Status

Setting up. Track pipelines in progress.
