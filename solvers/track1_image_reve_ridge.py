"""Track 01 submission: frozen REVE encoder, ridge-fitted pooled head.

The encoder is `brain-bzh/reve-base`, unchanged: all 139 tensors are bit-identical
to the published checkpoint and no gradient ever reached them. What is new is the
head.

REVE's own `final_layer` flattens `n_chans x n_patches x embed_dim`, so its width is
a property of the corpus it was fitted on and it cannot transfer between montages.
This ships a pooled head instead: tokens averaged over channels and patches, then a
single `Linear(512, n_outputs)` fitted in closed form by ridge regression on cached
features. That makes the head independent of channel count and window length, which
matters because the corpora in play disagree on both.

Input is resampled to 200 Hz to match REVE's pretraining, then z-scored per channel
and clipped at 15 SD, which is what the training features saw. The feature
standardisation used during fitting is already folded into the linear weights.
"""

import os

import torch

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import REVE

MIN_TIMES = 200        # REVE's patch_size; shorter inputs cannot be patched
TRAIN_SFREQ = 200.0    # REVE is pretrained at 200 Hz
CLIP_SD = 15.0


def resample_fft(x, n_out):
    """Resample along time by truncating or zero-padding the rfft spectrum.

    Pure torch on purpose: nothing is installed at evaluation, and linear
    interpolation aliases when downsampling.
    """
    n_in = x.shape[-1]
    if n_in == n_out:
        return x
    spec = torch.fft.rfft(x.float(), dim=-1)
    out = torch.zeros(
        *spec.shape[:-1], n_out // 2 + 1, dtype=spec.dtype, device=x.device
    )
    keep = min(spec.shape[-1], out.shape[-1])
    out[..., :keep] = spec[..., :keep]
    return torch.fft.irfft(out, n=n_out, dim=-1) * (n_out / n_in)


class RevePooled(torch.nn.Module):
    """REVE with its flattening head replaced by a pooled, portable one."""

    def __init__(self, sfreq, n_outputs, chs_info, **kwargs):
        super().__init__()
        self.net = REVE(
            n_outputs=n_outputs, sfreq=TRAIN_SFREQ, chs_info=chs_info, **kwargs
        )
        # Unused once we read tokens directly, and 25M random parameters otherwise.
        self.net.final_layer = torch.nn.Identity()
        self.head = torch.nn.Linear(self.net.embed_dim, n_outputs)
        self.in_sfreq = float(sfreq)

        # The position bank silently returns only the channels it recognises, which
        # then mismatches the input width ("size of tensor a (63) must match ... (57)").
        # Resolve the names ourselves so the two always agree, tolerating case
        # differences in montage naming (FP1 vs Fp1).
        bank = self.net._position_bank.mapping
        folded = {k.casefold(): k for k in bank}
        keep, resolved = [], []
        for i, ch in enumerate(c["ch_name"] for c in chs_info):
            name = ch if ch in bank else folded.get(ch.casefold())
            if name is not None:
                keep.append(i)
                resolved.append(name)
        if not keep:
            raise RuntimeError(
                f"no channel name resolved against REVE's position bank: "
                f"{[c['ch_name'] for c in chs_info][:8]}"
            )
        self.register_buffer("keep_idx", torch.tensor(keep), persistent=False)
        self.register_buffer(
            "pos", self.net.get_positions(resolved), persistent=False
        )

    def forward(self, X):
        X = X.index_select(1, self.keep_idx)
        n_out = max(int(round(X.shape[-1] * TRAIN_SFREQ / self.in_sfreq)), MIN_TIMES)
        X = resample_fft(X, n_out)
        mu = X.mean(dim=-1, keepdim=True)
        sd = X.std(dim=-1, keepdim=True)
        X = torch.clamp((X - mu) / (sd + 1e-8), -CLIP_SD, CLIP_SD)
        pos = self.pos.unsqueeze(0).expand(X.shape[0], -1, -1)
        feats = self.net(X, pos=pos, return_features=True)["features"]
        return self.head(feats.mean(dim=(1, 2)))

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X)


class Solver(CompetSolver):

    name = "REVE-base-frozen-ridge-head"

    def load_model(self, meta):
        # REVE fetches its position bank from HuggingFace at construction. Point it
        # at the copy in this ZIP: the download would otherwise run on every
        # submission and count against the one-hour evaluation budget.
        os.environ["REVE_POSITIONS_PATH"] = str(meta["submission_dir"])

        model = RevePooled(
            sfreq=meta["sfreq"],
            n_outputs=meta["n_outputs"],
            n_chans=meta["n_chans"],
            n_times=MIN_TIMES,   # forward is dynamic; this only sized the dropped head
            chs_info=meta["chs_info"],
        )
        state = torch.load(
            meta["submission_dir"] / "weights.pt",
            map_location=meta["device"],
            weights_only=True,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in weights.pt: {unexpected[:5]}")
        if missing:
            raise RuntimeError(f"weights missing: {missing[:5]}")
        return model.to(meta["device"]).eval()
