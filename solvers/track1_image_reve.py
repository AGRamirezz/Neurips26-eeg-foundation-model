"""Track 01 submission: pretrained REVE encoder, freshly-built projection head.

The encoder is `brain-bzh/reve-base`, pretrained by masked autoencoding on
60,000+ hours of EEG across 92 datasets. Its four `final_layer` tensors are
deliberately NOT shipped: that head is sized from `n_times`, so a checkpoint
containing it would only load against the exact window length it was saved at.
Dropping it lets one 277 MB file serve any dimensions the server hands us, at
the cost of the head starting random.

Consequence, stated plainly: this scores at chance. A random linear map into
the target embedding space carries no information about which image was seen.
What it measures is inference cost for a 69M-parameter transformer on the real
evaluation set, and that the weights load inside the worker.
"""

import os

import torch
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import REVE

MIN_TIMES = 200  # REVE's patch_size; shorter inputs cannot be patched


class ReveEmbedder(torch.nn.Module):
    """REVE plus the padding it needs on short windows."""

    def __init__(self, **kwargs):
        super().__init__()
        self.net = REVE(**kwargs)

    def forward(self, X):
        if X.shape[-1] < MIN_TIMES:
            X = F.pad(X, (0, MIN_TIMES - X.shape[-1]))
        return self.net(X)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X)


class Solver(CompetSolver):

    name = "REVE-base-frozen"

    def load_model(self, meta):
        # REVE fetches its position bank from HuggingFace at construction. Point it
        # at the copy shipped in this ZIP instead: the download would otherwise run
        # on every submission and count against the one-hour evaluation limit.
        os.environ["REVE_POSITIONS_PATH"] = str(meta["submission_dir"])

        model = ReveEmbedder(
            n_chans=meta["n_chans"],
            n_times=max(int(meta["n_times"]), MIN_TIMES),
            n_outputs=meta["n_outputs"],
            chs_info=meta["chs_info"],
            sfreq=meta["sfreq"],
        )
        state = torch.load(
            meta["submission_dir"] / "weights.pt",
            map_location=meta["device"],
            weights_only=True,
        )
        # Checkpoint holds bare REVE keys; our wrapper nests them under `net`.
        state = {f"net.{k}": v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in weights.pt: {unexpected[:5]}")
        if any(not k.startswith("net.final_layer") for k in missing):
            raise RuntimeError(f"encoder weights missing: {missing[:5]}")
        return model.to(meta["device"]).eval()
