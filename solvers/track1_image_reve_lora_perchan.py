"""Track 01 submission: REVE with LoRA adapters and a per-electrode head.

The encoder is `brain-bzh/reve-base`. Its pretrained weights are frozen; LoRA learns a
small rank-r correction beside each of the 44 attention projections
(`transformer.layers.*.0.{to_qkv,to_out}`), so no pretrained weight is overwritten.

The head gives every electrode its own weights. Averaging electrodes together, or
applying one shared map to each, both lost the signal on this task, and each electrode
needing its own mapping was the consistent finding. Each electrode's 512-d token is
shrunk to a few numbers by one shared layer, then a final layer with separate weights
per electrode maps them to the target embedding. The shrink keeps that layer small.

That makes the head specific to the electrodes it was trained on. Inputs are matched to
them by name: extra electrodes are ignored and missing ones are filled with zeros, which
after the harness's scaling means "no information". Electrodes REVE's position bank does
not know are left out, at training and at inference alike.

Input is the harness's own preprocessing (120 Hz, RobustScaler, baseline, clamp),
resampled to REVE's 200 Hz and standardised per window inside the model, so the
training windows and the scoring windows go through identical code.

This file is also the model definition the training notebook uses, so the two cannot
drift apart.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import REVE

REVE_SFREQ = 200.0     # REVE's pretraining rate
PATCH = 200            # REVE's patch_size in samples
CLIP_SD = 15.0
LORA_TARGETS = ("to_qkv", "to_out")


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum. Pure torch, since nothing is
    installed at evaluation and linear interpolation aliases."""
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
    """Frozen base plus a rank-r update. B starts at zero, so training begins at
    exactly the pretrained function."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


class PerElectrodeHead(nn.Module):
    """Shared shrink per electrode, then separate weights per electrode."""

    def __init__(self, n_chans, embed_dim, shrink, n_outputs, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.shrink = nn.Linear(embed_dim, shrink)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_chans * shrink, n_outputs)

    def forward(self, tokens):                    # (B, chan, embed)
        z = self.drop(F.gelu(self.shrink(self.norm(tokens))))
        return self.out(z.flatten(1))


class RevePerChannel(nn.Module):

    def __init__(self, channels, n_outputs, in_sfreq, rank=8, alpha=16, shrink=32,
                 dropout=0.3, encoder_state=None):
        super().__init__()
        self.in_sfreq = float(in_sfreq)
        self.net = REVE(
            n_outputs=n_outputs, n_chans=len(channels), n_times=PATCH,
            sfreq=REVE_SFREQ, chs_info=[{"ch_name": c} for c in channels],
        )
        self.net.final_layer = nn.Identity()
        if encoder_state is not None:             # training: pretrained weights first
            self.net.load_state_dict(encoder_state, strict=False)
        for p in self.net.parameters():
            p.requires_grad_(False)
        names = [
            n for n, m in self.net.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in LORA_TARGETS
        ]
        for name in names:
            parent = self.net.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.split(".")[-1],
                    LoRALinear(self.net.get_submodule(name), rank, alpha))
        self.n_lora = len(names)

        # The position bank returns only names it recognises and merely logs the
        # shortfall, which later breaks the forward. Resolve here, identically for
        # training and inference, and leave unknown electrodes out.
        bank = self.net._position_bank.mapping
        folded = {k.casefold(): k for k in bank}
        keep, resolved = [], []
        for i, ch in enumerate(channels):
            name = ch if ch in bank else folded.get(ch.casefold())
            if name is not None:
                keep.append(i)
                resolved.append(name)
        if not keep:
            raise RuntimeError(f"no electrode resolved in REVE's bank: {channels[:8]}")
        self.used = resolved
        self.register_buffer("keep_idx", torch.tensor(keep), persistent=False)
        self.register_buffer("pos", self.net.get_positions(resolved), persistent=False)
        self.head = PerElectrodeHead(len(keep), self.net.embed_dim, shrink,
                                     n_outputs, dropout)

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if n.endswith((".A", ".B"))]

    def forward(self, X):                          # (B, len(channels), T) at in_sfreq
        X = X.float().index_select(1, self.keep_idx)
        n_out = max(int(round(X.shape[-1] * REVE_SFREQ / self.in_sfreq)), PATCH)
        X = resample_fft(X, n_out)
        X = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-8)
        X = torch.clamp(X, -CLIP_SD, CLIP_SD)
        pos = self.pos.unsqueeze(0).expand(X.shape[0], -1, -1)
        feats = self.net(X, pos=pos, return_features=True)["features"]
        return self.head(feats.mean(dim=2))       # average the time patches


class NameMatched:
    """Reorders incoming electrodes into the trained order, zero-filling gaps."""

    def __init__(self, model, channels, in_names, device):
        self.model, self.device = model, device
        lookup = {n.casefold(): i for i, n in enumerate(in_names)}
        self.src = [lookup.get(c.casefold(), -1) for c in channels]
        missing = [c for c, i in zip(channels, self.src) if i < 0]
        print(
            f"[REVE-LoRA] matched {len(channels) - len(missing)}/{len(channels)} "
            f"training channels by name among {len(in_names)} inputs; "
            f"zero-filled: {missing if missing else 'none'}; REVE uses "
            f"{len(model.used)} with known positions"
        )

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        out = torch.zeros(X.shape[0], len(self.src), X.shape[-1], device=self.device)
        for dst, src in enumerate(self.src):
            if src >= 0:
                out[:, dst] = X[:, src]
        return self.model(out)


class Solver(CompetSolver):

    name = "REVE-LoRA-per-electrode"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        os.environ["REVE_POSITIONS_PATH"] = str(sub)   # bundled; no download at eval
        cfg = json.loads((sub / "reve_config.json").read_text())
        model = RevePerChannel(
            cfg["channels"], cfg["n_outputs"], meta["sfreq"], rank=cfg["rank"],
            alpha=cfg["alpha"], shrink=cfg["shrink"], dropout=cfg["dropout"],
        )
        state = torch.load(sub / "weights.pt", map_location=meta["device"],
                           weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or missing:
            raise RuntimeError(f"weights mismatch: missing {missing[:4]}, "
                               f"unexpected {unexpected[:4]}")
        model = model.to(meta["device"]).eval()
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        return NameMatched(model, cfg["channels"], in_names, meta["device"])
