"""Track 02 submission: pretrained EEGPT encoder, head rebuilt from meta.

Checkpoint is `braindecode/eegpt-pretrained`. Chosen for domain: EEGPT's
pretraining corpus is PhysioNetMI (109 subjects of motor imagery) plus the
Tsinghua SSVEP benchmark and M3CV. Of the available encoders it is the only one
whose pretraining is BCI paradigms rather than clinical recordings or a broad
evoked mixture, which is what this track decodes.

Unlike REVE and BIOT, the checkpoint imposes no dimensional constraint: the
patch embedding is a convolution over time only, and channel handling lives in a
projection layer rather than the pretrained weights. The same file loads at 8,
62 or 64 channels and at any window length.

Two things are still deliberate:

1. `chans_id` is a derived buffer the model rebuilds at the right size, so it is
   not shipped. Loading the checkpoint's 62-wide version into a narrower model
   would be wrong.
2. Inputs are resampled to 256 Hz. Nothing in the weights depends on sampling
   rate, but the patch convolution has a fixed 64-sample kernel, so its receptive
   field in *seconds* moves with the rate. 256 Hz is what the checkpoint was
   trained at.

Untrained after loading: the head and the channel projection, ~33k of 25.3M
parameters. The projection sits *before* the encoder, so this scores at chance
until both are fitted.
"""

import torch
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import EEGPT

CKPT_SFREQ = 256.0     # rate the checkpoint was trained at
MIN_TIMES = 64         # patch kernel


class EegptClassifier(torch.nn.Module):
    """EEGPT with checkpoint-rate resampling, emitting (B,) class indices."""

    def __init__(self, n_chans, n_times_resampled, n_classes):
        super().__init__()
        self.net = EEGPT(
            n_chans=n_chans,
            n_times=n_times_resampled,
            n_outputs=n_classes,
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
        # The track wants one class index per window, not logits.
        return self(X).argmax(dim=1)


class Solver(CompetSolver):

    name = "EEGPT-bci-frozen"

    def load_model(self, meta):
        n_classes = int(meta.get("n_classes") or meta["n_outputs"])
        n_times = max(
            MIN_TIMES,
            round(int(meta["n_times"]) * CKPT_SFREQ / float(meta["sfreq"])),
        )
        model = EegptClassifier(int(meta["n_chans"]), n_times, n_classes)

        state = torch.load(
            meta["submission_dir"] / "weights.pt",
            map_location=meta["device"],
            weights_only=True,
        )
        state = {f"net.{k}": v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in weights.pt: {unexpected[:5]}")
        allowed = ("net.final_layer", "net.chan_proj", "net.chans_id")
        stray = [k for k in missing if not k.startswith(allowed)]
        if stray:
            raise RuntimeError(f"encoder weights missing: {stray[:5]}")
        return model.to(meta["device"]).eval()
