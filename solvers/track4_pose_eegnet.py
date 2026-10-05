"""Track 04 submission: EEGNet regressing 20 joint-angle trajectories from wrist EMG.

The model follows the organisers' `eegnet_pose.py` baseline: braindecode EEGNet with
`final_conv_length=1`, so it emits one joint-angle vector per remaining time step, and
targets standardised by one train-split mean and std kept as buffers. It is trained
with an L1 loss (the metric is a mean absolute error) and the task config's electrode
band rotation, in `track4_02_eegnet.ipynb`.

Input handling repeats the prepare notebook exactly, so training and scoring see the
same data: per-window, per-channel mean removal, resampling from the harness's 2 kHz
to 1 kHz with the same rfft function, and volts to microvolts. EEGNet's output (about
31 steps per second) is linearly upsampled to 200 Hz; the harness then stretches that
to 2 kHz by nearest neighbour. Electrodes are matched by name, and a zip may hold
several seeds whose predictions are averaged.
"""

import json

import torch
import torch.nn.functional as F
from torch import nn

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import EEGNet


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum. Pure torch, since nothing is
    installed at evaluation; the prepare notebook uses the same function."""
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


class DensePose(nn.Module):
    """EEGNet emitting a standardised joint-angle sequence, upsampled to out_steps."""

    def __init__(self, n_chans, n_joints, n_times, out_steps, f1=8, dropout=0.25):
        super().__init__()
        self.out_steps = out_steps
        self.net = EEGNet(
            n_chans=n_chans, n_outputs=n_joints, n_times=n_times,
            final_conv_length=1, F1=f1, drop_prob=dropout,
        )
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, X):                          # standardised (B, J, out_steps)
        out = self.net(X)
        out = out[..., None] if out.ndim == 2 else out
        return F.interpolate(out, size=self.out_steps, mode="linear", align_corners=False)

    def to_radians(self, out):
        return out * self.y_std + self.y_mean


class ChannelMatchedPose:

    def __init__(self, cfg, in_names, in_sfreq, n_joints, device):
        if n_joints not in (None, cfg["n_joints"]):
            raise RuntimeError(f"trained for {cfg['n_joints']} joints, harness asks for {n_joints}")
        self.device = device
        self.cfg = cfg
        self.in_sfreq = float(in_sfreq)
        self.nets = [
            DensePose(len(cfg["channels"]), cfg["n_joints"], cfg["n_times"],
                      cfg["out_steps"], cfg["f1"], cfg["dropout"]).to(device).eval()
            for _ in cfg["weight_files"]
        ]
        lookup = {name.casefold(): i for i, name in enumerate(in_names)}
        self.src = [lookup.get(name.casefold(), -1) for name in cfg["channels"]]
        missing = [n for n, i in zip(cfg["channels"], self.src) if i < 0]
        print(
            f"[EEGNet-pose] matched {len(self.src) - len(missing)}/{len(self.src)} "
            f"training channels by name among {len(in_names)} inputs; "
            f"zero-filled: {missing if missing else 'none'}"
        )
        if len(missing) == len(self.src):
            raise RuntimeError(f"no training channel found in the input: {list(in_names)[:8]}")

    @torch.no_grad()
    def predict(self, X):
        cfg = self.cfg
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        out = torch.zeros(X.shape[0], len(self.src), X.shape[-1], device=self.device)
        for dst, src in enumerate(self.src):
            if src >= 0:
                out[:, dst] = X[:, src]
        out = out - out.mean(-1, keepdim=True)                       # as in prepare
        n_out = int(round(out.shape[-1] * cfg["sfreq"] / self.in_sfreq))
        out = resample_fft(out, n_out) * cfg["emg_scale"]
        n = int(cfg["n_times"])
        out = out[..., :n] if out.shape[-1] >= n else F.pad(out, (0, n - out.shape[-1]))
        preds = [net.to_radians(net(out)) for net in self.nets]
        return torch.stack(preds).mean(0)                           # (B, J, out_steps) radians


class Solver(CompetSolver):

    name = "EEGNet-pose"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        cfg = json.loads((sub / "pose_eegnet_config.json").read_text())
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        model = ChannelMatchedPose(cfg, in_names, meta["sfreq"], meta.get("n_joints"),
                                   meta["device"])
        for net, name in zip(model.nets, cfg["weight_files"]):
            net.load_state_dict(torch.load(sub / name, map_location=meta["device"],
                                           weights_only=True))
        return model
