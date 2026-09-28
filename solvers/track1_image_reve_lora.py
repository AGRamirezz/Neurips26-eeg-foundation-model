"""Track 01 submission: REVE with LoRA on the attention projections, pooled head.

Every pretrained tensor is unchanged. LoRA freezes each attention projection `W` and
learns a rank-r correction `B·A` beside it, so the checkpoint's own weights are never
overwritten; `B` trains from zero, meaning the model starts at exactly the pretrained
function.

Adapted modules are the 44 `transformer.layers.*.0.{to_qkv,to_out}` projections named
in `neuralbench/models/reve.yaml`. The head pools tokens over channels and patches and
maps `embed_dim -> n_outputs`, which keeps it independent of channel count and window
length; REVE's own `final_layer` flattens and so is corpus-specific.

Input is resampled to 200 Hz to match pretraining, z-scored per channel and clipped at
15 SD, matching how the training windows were built.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import REVE

MIN_TIMES = 200        # REVE's patch_size
TRAIN_SFREQ = 200.0
CLIP_SD = 15.0
LORA_RANK = 16         # must match training
LORA_ALPHA = 32
LORA_TARGETS = ("to_qkv", "to_out")


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum. Pure torch: nothing is installed
    at evaluation, and linear interpolation aliases when downsampling."""
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


class LoRALinear(nn.Module):
    """Frozen base plus a rank-r update. Structure must match training exactly, or
    the state dict will not load."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


class RevePooledLoRA(nn.Module):

    def __init__(self, sfreq, n_outputs, chs_info, **kwargs):
        super().__init__()
        self.net = REVE(
            n_outputs=n_outputs, sfreq=TRAIN_SFREQ, chs_info=chs_info, **kwargs
        )
        self.net.final_layer = nn.Identity()

        names = [
            n for n, m in self.net.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in LORA_TARGETS
        ]
        for name in names:
            parent = self.net.get_submodule(name.rsplit(".", 1)[0])
            setattr(
                parent,
                name.split(".")[-1],
                LoRALinear(self.net.get_submodule(name), LORA_RANK, LORA_ALPHA),
            )
        self.n_lora = len(names)
        self.head = nn.Linear(self.net.embed_dim, n_outputs)
        self.in_sfreq = float(sfreq)

        # The position bank returns only the names it recognises and merely logs the
        # shortfall, which later surfaces as a tensor-size mismatch in the forward.
        # Resolve names here so the input width and the positions always agree.
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
                "no channel name resolved against REVE's position bank: "
                f"{[c['ch_name'] for c in chs_info][:8]}"
            )
        self.register_buffer("keep_idx", torch.tensor(keep), persistent=False)
        self.register_buffer("pos", self.net.get_positions(resolved), persistent=False)

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

    name = "REVE-base-lora-qkvo-clip"

    def load_model(self, meta):
        os.environ["REVE_POSITIONS_PATH"] = str(meta["submission_dir"])

        model = RevePooledLoRA(
            sfreq=meta["sfreq"],
            n_outputs=meta["n_outputs"],
            chs_info=meta["chs_info"],
            n_chans=meta["n_chans"],
            n_times=MIN_TIMES,   # forward is dynamic; this only sized the dropped head
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
