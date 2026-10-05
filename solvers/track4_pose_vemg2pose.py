"""Track 04 submission: VEMG2Pose, Meta's pretrained emg2pose decoder, optionally fine-tuned.

The model is braindecode's `VEMG2Pose` (Salter et al. 2024): a time-depth-separable
convolutional encoder feeding an LSTM that rolls the hand pose out at 50 Hz. Its
weights start from Meta's released regression checkpoint
(`braindecode/VEMG2Pose-emg2pose`, CC-BY-NC-SA-4.0) and may be fine-tuned in
`track4_03_vemg2pose.ipynb`. They ship in this zip; nothing is downloaded at scoring.

Input handling repeats what the model saw in training: per-window, per-channel mean
removal, a round trip through 1 kHz with the same rfft resampler (the training windows
were stored at 1 kHz), and volts to microvolts, the scale the checkpoint expects. The
model consumes 1,790 samples of lead-in before its first prediction and the harness
gives none, so the window's first 1,790 samples are mirrored in front of it. Electrodes
are matched by name; output is (B, 20, T) radians at the input rate.
"""

import json

import torch

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import VEMG2Pose


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum; the prepare notebook uses the same."""
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


def lead_in(x, n):
    """Mirror the first n samples in front of the window, for the model's left context."""
    return torch.cat([x[..., 1:n + 1].flip(-1), x], dim=-1)


class ChannelMatchedVEMG2Pose:

    def __init__(self, cfg, in_names, in_sfreq, n_joints, device):
        if n_joints not in (None, cfg["n_joints"]):
            raise RuntimeError(f"trained for {cfg['n_joints']} joints, harness asks for {n_joints}")
        self.cfg, self.device = cfg, device
        self.in_sfreq = float(in_sfreq)
        self.net = VEMG2Pose(
            n_outputs=cfg["n_joints"], n_chans=len(cfg["channels"]),
            sfreq=cfg["model_sfreq"], n_times=cfg["model_n_times"],
        ).to(device).eval()
        lookup = {name.casefold(): i for i, name in enumerate(in_names)}
        self.src = [lookup.get(name.casefold(), -1) for name in cfg["channels"]]
        missing = [n for n, i in zip(cfg["channels"], self.src) if i < 0]
        print(
            f"[VEMG2Pose] matched {len(self.src) - len(missing)}/{len(self.src)} "
            f"training channels by name among {len(in_names)} inputs; "
            f"zero-filled: {missing if missing else 'none'}"
        )
        if len(missing) == len(self.src):
            raise RuntimeError(f"no training channel found in the input: {list(in_names)[:8]}")

    @torch.no_grad()
    def predict(self, X):
        cfg = self.cfg
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        x = torch.zeros(X.shape[0], len(self.src), X.shape[-1], device=self.device)
        for dst, src in enumerate(self.src):
            if src >= 0:
                x[:, dst] = X[:, src]
        x = x - x.mean(-1, keepdim=True)
        n_model = int(round(x.shape[-1] * cfg["model_sfreq"] / self.in_sfreq))
        n_stored = int(round(x.shape[-1] * cfg["stored_sfreq"] / self.in_sfreq))
        x = resample_fft(resample_fft(x, n_stored), n_model) * cfg["emg_scale"]
        out = self.net(lead_in(x, self.net.left_context))          # (B, T, J)
        return out.transpose(1, 2)                                  # (B, J, T) radians


class Solver(CompetSolver):

    name = "VEMG2Pose"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        cfg = json.loads((sub / "vemg2pose_config.json").read_text())
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        model = ChannelMatchedVEMG2Pose(cfg, in_names, meta["sfreq"], meta.get("n_joints"),
                                        meta["device"])
        model.net.load_state_dict(torch.load(sub / "weights.pt", map_location=meta["device"],
                                             weights_only=True))
        return model
