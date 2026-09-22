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
