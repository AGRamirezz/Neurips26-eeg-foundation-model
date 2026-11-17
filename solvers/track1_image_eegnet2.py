"""Track 01 submission: a wider EEGNet-style network with a funnel head, plain torch.

The window (electrodes x time, 120 Hz) is standardised per window, then:
temporal convolution (one 1x64 filter bank per electrode), a depthwise spatial convolution over
all electrodes (the step that mixes electrodes), pooling, a separable convolution, pooling, and a
funnel head (Linear to `hidden`, GELU, dropout, Linear to the 1536-d target). With the default
sizes (F1 16, D 2, F2 64, pools 4 and 2) the feature map entering the head has 64 x 15 = 960 values,
instead of the 48 of the stock braindecode EEGNet, so the 1536-d output is not confined to a
48-dimensional subspace.

Written in plain torch (no braindecode), so nothing is installed at evaluation. Inputs are matched to
the trained electrodes by name (case-insensitive); extra electrodes are ignored and missing ones are
zero-filled. The window is trimmed or zero-padded to the trained length.

This file is also the model definition the training notebook uses, so the two cannot drift apart.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver

TARGET_SFREQ = 120.0
CLIP_SD = 15.0


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum (pure torch; nothing is installed at evaluation)."""
    n_in = x.shape[-1]
    if n_in == n_out:
        return x
    spec = torch.fft.rfft(x.float(), dim=-1)
    out = torch.zeros(*spec.shape[:-1], n_out // 2 + 1, dtype=spec.dtype, device=x.device)
    keep = min(spec.shape[-1], out.shape[-1])
    out[..., :keep] = spec[..., :keep]
    return torch.fft.irfft(out, n=n_out, dim=-1) * (n_out / n_in)


class EEGNetCLIP(nn.Module):
    def __init__(self, channels, n_outputs, in_sfreq, F1=16, D=2, F2=64, kernel=64, depth_kernel=16,
                 pool1=4, pool2=2, hidden=512, dropout=0.25, n_times=120):
        super().__init__()
        self.in_sfreq, self.n_times = float(in_sfreq), int(n_times)
        C = len(channels)
        self.pad_t = nn.ZeroPad2d((kernel // 2 - 1, kernel // 2, 0, 0))     # 'same' length for an even kernel
        self.temporal = nn.Conv2d(1, F1, (1, kernel), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)  # mixes all electrodes
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)
        self.pad_s = nn.ZeroPad2d((depth_kernel // 2 - 1, depth_kernel // 2, 0, 0))
        self.depth = nn.Conv2d(F1 * D, F1 * D, (1, depth_kernel), groups=F1 * D, bias=False)
        self.point = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)
        self.feat_dim = F2 * ((self.n_times // pool1) // pool2)
        self.head = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_outputs))

    def forward(self, X):                          # (B, n_channels in trained order, T)
        X = X.float()
        n_out = max(int(round(X.shape[-1] * TARGET_SFREQ / self.in_sfreq)), 8)
        X = resample_fft(X, n_out)
        if X.shape[-1] > self.n_times:
            X = X[..., : self.n_times]
        elif X.shape[-1] < self.n_times:
            X = F.pad(X, (0, self.n_times - X.shape[-1]))
        X = (X - X.mean(dim=(1, 2), keepdim=True)) / (X.std(dim=(1, 2), keepdim=True) + 1e-8)
        X = torch.clamp(X, -CLIP_SD, CLIP_SD)
        x = self.bn1(self.temporal(self.pad_t(X[:, None])))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(x)))))
        x = self.bn3(self.point(self.depth(self.pad_s(x))))
        x = self.drop2(self.pool2(F.elu(x)))
        return self.head(x.flatten(1))


class NameMatched:
    """Reorders incoming electrodes into the trained order, zero-filling gaps."""

    def __init__(self, model, channels, in_names, device):
        self.model, self.device = model, device
        lookup = {n.casefold(): i for i, n in enumerate(in_names)}
        self.src = [lookup.get(c.casefold(), -1) for c in channels]
        missing = [c for c, i in zip(channels, self.src) if i < 0]
        print(f"[EEGNet2] matched {len(channels) - len(missing)}/{len(channels)} training channels by "
              f"name among {len(in_names)} inputs; zero-filled: {missing if missing else 'none'}")

    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        out = torch.zeros(X.shape[0], len(self.src), X.shape[-1], device=self.device)
        for dst, src in enumerate(self.src):
            if src >= 0:
                out[:, dst] = X[:, src]
        self.model.eval()
        return self.model(out)


class Solver(CompetSolver):

    name = "EEGNet2-CLIP-Alljoined"

    def load_model(self, meta):
        sub = meta["submission_dir"]
        cfg = json.loads((sub / "eegnet_config.json").read_text())
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        model = EEGNetCLIP(cfg["channels"], cfg["n_outputs"], meta["sfreq"], **cfg["arch"])
        state = torch.load(sub / "weights.pt", map_location=meta["device"], weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or missing:
            raise RuntimeError(f"weights mismatch: missing {missing[:4]}, unexpected {unexpected[:4]}")
        return NameMatched(model.to(meta["device"]).eval(), cfg["channels"], in_names, meta["device"])
