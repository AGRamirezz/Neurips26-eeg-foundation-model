"""Track 03 submission: BIOT pretrained on SHHS + PREST, head rebuilt from meta.

Checkpoint is `braindecode/biot-pretrained-shhs-prest-18chs`. Chosen for domain:
the Sleep Heart Health Study contributes ~5M samples of overnight polysomnography,
so sleep dominates its pretraining. The alternatives are pretrained on clinical
routine EEG (TUEG) or on broad evoked-paradigm mixtures, neither of which is the
slow low-arousal regime this track is scored on.

Three things the checkpoint forces, all discovered by trying to load it:

1. It was trained at **200 Hz**. The patch embedding expects 101 frequency bins;
   feeding 100 Hz data produces 51 and the weights will not load. Inputs are
   resampled to 200 Hz in `predict` rather than rebuilding the projection.
2. `channel_tokens` is a learned **[18, 256]** table, one row per channel. BIOT
   does not project arbitrary channel counts: that wrapper belongs to NeuralBench,
   not the model. We slice the table to the channels we actually have.
3. `final_layer` is sized by `n_outputs`, so it is not shipped and is rebuilt from
   `meta`, leaving it randomly initialised.

Consequence: this scores no better than chance until the head is trained. Track 3
is a regression, so an untrained head emits arbitrary second values and bMAE may
land far worse than the provided median floor. That is expected, not a fault.
"""

import torch
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import BIOT

CKPT_SFREQ = 200.0     # the rate the checkpoint was trained at
MAX_CHANS = 18         # rows in the pretrained channel-token table


class BiotRegressor(torch.nn.Module):
    """BIOT plus the resampling its checkpoint requires, emitting (B,) seconds."""

    def __init__(self, n_chans, n_times_resampled, n_outputs):
        super().__init__()
        self.net = BIOT(
            n_chans=n_chans,
            n_times=n_times_resampled,
            n_outputs=n_outputs,
            sfreq=CKPT_SFREQ,
        )
        self.n_times_resampled = n_times_resampled

    def forward(self, X):
        if X.shape[-1] != self.n_times_resampled:
            X = F.interpolate(
                X, size=self.n_times_resampled, mode="linear", align_corners=False
            )
        return self.net(X)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        out = self(X)
        return out.reshape(out.shape[0])      # (B, 1) -> (B,), seconds


class Solver(CompetSolver):

    name = "BIOT-shhs-frozen"

    def load_model(self, meta):
        n_chans = int(meta["n_chans"])
        if n_chans > MAX_CHANS:
            raise ValueError(
                f"{n_chans} channels exceeds the pretrained token table ({MAX_CHANS})"
            )
        # Resample length so the model sees CKPT_SFREQ regardless of the source rate.
        n_times = max(1, round(int(meta["n_times"]) * CKPT_SFREQ / float(meta["sfreq"])))

        model = BiotRegressor(n_chans, n_times, int(meta["n_outputs"]))

        state = torch.load(
            meta["submission_dir"] / "weights.pt",
            map_location=meta["device"],
            weights_only=True,
        )
        tokens = "encoder.channel_tokens.weight"
        if tokens in state:
            state[tokens] = state[tokens][:n_chans]
        state = {f"net.{k}": v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in weights.pt: {unexpected[:5]}")
        if any("final_layer" not in k for k in missing):
            raise RuntimeError(f"encoder weights missing: {missing[:5]}")
        return model.to(meta["device"]).eval()
