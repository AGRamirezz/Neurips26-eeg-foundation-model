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
  track1_03_head.ipynb      fit the 512 -> 1536 head by ridge, score it, build the zip
  track1_04_diagnose.ipynb  why a head is at chance: pipeline, pooling, or representation
  track1_05_train.ipynb     ClipLoss probe on frozen features, then LoRA on QKVO
  track1_06_signal.ipynb    is the image signal in the raw EEG, and at what latency
  track1_07_eegnet.ipynb    the organisers' EEGNet-CLIP recipe on the harness's own preprocessing
  track1_08_things_prepare.ipynb  warm-up corpus (THINGS-EEG2): one session per person, small on Drive
  track1_09_reve_lora.ipynb       REVE + LoRA with a per-electrode head, against EEGNet on the same data
  track1_10_things_eeg1_prepare.ipynb  THINGS-EEG1: 50 more people in the THINGS-EEG2 window format, test concepts excluded
  track2_01_prepare.ipynb   Dreyer 2023 warm-up windows and labels via MOABB, made as the harness makes them
  track2_02_eegnet.ipynb    the organisers' EEGNet motor-imagery classifier, scored by balanced accuracy, packaged
  track3_01_prepare.ipynb   Sleep-EDF warm-up windows, targets and split, made as the harness makes them
  track3_02_eegnet.ipynb    the organisers' EEGNet sleep-onset regressor, scored by bMAE, packaged
  track3_03_muse_prepare.ipynb  sealed-phase Muse data (nm000287): windows, seen and unseen splits
  track4_01_prepare.ipynb   emg2pose warm-up: a balanced 2.5 GB sample across people, movements and hands
  track4_02_eegnet.ipynb    the organisers' EEGNet pose regressor with L1 loss and band rotation, packaged
  track4_03_vemg2pose.ipynb Meta's pretrained VEMG2Pose: zero-shot score, fine-tune, package the better
src/
  tracks.py     per-track config: task, metric, expected channels and rate, references
  acquire.py    verification checks and the on-disk manifest
  datastore.py  inventory of what is already downloaded, and top-up planning
  miniload.py   byte-budget record selection, and the run-based split track 1 needs
  subset.py     split-aware subsetting that preserves each track's held-out axis
  harness.py    local stand-in for the competition harness, plus the four track metrics
solvers/        one submission.py per track, flat-zipped with its weights
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
