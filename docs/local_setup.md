# Local setup: build and test a submission without Colab

Packaging and contract-testing a submission need no GPU and no competition data.
Only training does. This is the laptop half.

## One-time setup

The competition repo and the virtualenv live *outside* this repo, as siblings:

```
<parent>/
  Neurips26-eeg-foundation-model/   this repo
  2026-competition/                 git clone of the organizers' repo
  .venv-compet/                     python 3.13 venv
  submissions/                      build output, not tracked anywhere
```

```bash
cd <parent>
git clone --depth 1 https://github.com/neural-interfaces26/2026-competition.git
python3.13 -m venv .venv-compet
./.venv-compet/bin/pip install "benchopt>=1.10" torch numpy scikit-learn pandas \
                               braindecode huggingface_hub safetensors
```

The worker image runs **Python 3.12**. 3.13 works locally; keep the gap in mind if
anything version-sensitive appears.

`benchopt install tracks/<track>` pulls the full data stack (neuralbench, moabb,
mne, awscli). The list above is enough for `-d Simulated` and is several GB smaller.

## Test a submission

`Simulated` needs no download. It is not only a format check: each of its 20
synthetic images gets a signal template plus a random embedding, so retrieval is
genuinely learnable and a working model should beat chance on it.

```bash
cd <parent>/2026-competition
cp <submission_dir>/submission.py tracks/<track>/solvers/my_submission.py
COMPET_SUBMISSION_DIR="<submission_dir>" \
  ../.venv-compet/bin/benchopt run tracks/<track> -d Simulated -s <Solver.name> --no-plot
```

Results land in `tracks/<track>/outputs/*.parquet`. Read
`objective_*` columns; always read `objective_n_candidates` beside a retrieval
score, since chance is `k / n_candidates`.

## Assemble the ZIP

Everything at the root. A nested directory is the most common ingestion failure.

```bash
zip -j my_submission.zip submission.py weights.pt <any other artefacts>
python -c "import zipfile,sys; n=zipfile.ZipFile(sys.argv[1]).namelist(); \
           assert not [x for x in n if '/' in x], n; print('flat:', n)" my_submission.zip
```

## Track 1 notes discovered by doing this

**REVE base is 69.3M parameters**, not the 14M quoted in the NeuralBench docs. The
encoder-only checkpoint is ~277 MB.

**Drop the head from the checkpoint.** REVE's `final_layer` is sized from
`n_times`, so a checkpoint containing it only loads at the exact window length it
was saved at. Excluding those four tensors lets one file load at any
channel/window/output combination, with the head rebuilt from `meta`.

**Bundle the position bank.** REVE downloads `positions.json` from HuggingFace when
it constructs. Ship the file and set `REVE_POSITIONS_PATH` to `meta["submission_dir"]`
in `load_model`, or that download runs on every submission and counts against the
one-hour limit. Verify by hiding `~/mne_data/reve_positions.json` and re-running.

**Pad short windows.** REVE's `patch_size` is 200; `Simulated` supplies 120 samples
and errors without padding.

## Per-track findings from building all four baselines

Each checkpoint forces something different. Discovered by trying to load them, not
from documentation.

**Track 1 · REVE** (`brain-bzh/reve-base`). 69.3M parameters, not the 14M the
NeuralBench docs state; 277 MB encoder-only. `final_layer` is sized from
`n_times`, so it must be dropped for the checkpoint to load at any window length.
`patch_size` is 200, so shorter windows need padding. It fetches `positions.json`
from HuggingFace at construction — bundle it and set `REVE_POSITIONS_PATH` to
`meta["submission_dir"]`, or that download repeats on every submission.

**Track 2 · EEGPT** (`braindecode/eegpt-pretrained`). The most portable of the
four: no `n_times` dependency (the patch embedding is a conv over time only) and
no `n_chans` dependency in the weights. Identical load behaviour at 8/200,
62/1024 and 64/2000. Drop the `chans_id` buffer, which the model rebuilds.
Untrained after loading: the head *and* `chan_proj`, which sits before the
encoder, so the pretrained weights see randomly-mixed channels.

**Track 3 · BIOT** (`braindecode/biot-pretrained-shhs-prest-18chs`). Trained at
**200 Hz**: the patch embedding expects 101 frequency bins and 100 Hz data yields
51, so the weights refuse to load. Resample in `predict`. `channel_tokens` is a
learned [18, 256] table, so BIOT does *not* project arbitrary channel counts —
that wrapper is NeuralBench's, not the model's. Slice the table to `n_chans`.
Drop `encoder.index`, a non-persistent `arange(n_chans)` buffer.

**Track 4 · NeuroPose**. No pretrained EMG-to-pose weights exist anywhere.
NeuroPose emits `(B, T, n_joints)`; the contract wants `(B, n_joints, T)`.
VEMG2Pose is the stronger architecture but consumes 1790 samples of left context,
so it cannot be contract-tested against `Simulated` at 400 samples.

## Warm-up corpora differ from the sealed cohorts

Read off the ingestion logs. This matters because it means warm-up rank is a poor
guide on several tracks.

| track | warm-up study | sealed cohort |
|---|---|---|
| 1 | `gifford2022large` (THINGS-EEG2) | Alljoined, 32-ch consumer |
| 2 | `dreyer2023`, **2 classes** | Graz/BrainHero, **3 commands** |
| 3 | `Sleep-EDF` | Muse, 4-ch wearable |
| 4 | `Salter2024Emg2pose` | held-out users/stages |

Track 1's warm-up corpus is one REVE was pretrained on. Track 2's warm-up is
binary left/right motor imagery while the sealed task is three commands, two of
which are non-motor.
