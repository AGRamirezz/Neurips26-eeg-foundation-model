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
notebooks/     One notebook per track: data, training, evaluation, packaging
src/           Shared code — metrics, adapters, submission scaffolding
configs/       Training and evaluation configs
submissions/   Packaged submissions (contents not tracked)
data/          Local datasets and caches (not tracked)
outputs/       Run artifacts, scores, logs (not tracked)
docs/          Notes and writeups
```

## Data and licensing

No datasets or model weights are redistributed here. Each corpus keeps its original
license and is obtained from its own source. Some are access-gated or non-commercial —
check the terms of any dataset or checkpoint before using it.

## Status

Setting up. Track pipelines in progress.
