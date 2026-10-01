"""Track 03 submission: EEGNet regressing seconds to sleep onset.

The model and loss follow the organisers' `eegnet_reg.py` baseline: one output,
MSE on the latency divided by 600, predictions multiplied back and clamped to
[0, 600] s. The weights come from `track3_02_eegnet.ipynb`, trained on
Sleep-EDF windows made with the harness's own preprocessing. A zip may hold
several copies trained from different seeds; their predictions are averaged.

Inputs are matched to the training electrodes by name (Fpz-Cz, Pz-Oz on the
warm-up data): extra electrodes are ignored, missing ones are filled with zeros,
and it raises if none match. The sealed phase uses a Muse headband with
different electrodes, so this file serves the warm-up only.
"""

import json

import torch
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import EEGNet


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum. Pure torch, since nothing is
    installed at evaluation."""
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


class ChannelMatchedRegressor:

    def __init__(self, cfg, in_names, in_sfreq, device):
        self.device = device
        self.cap = float(cfg["cap_s"])
        self.n_times = int(cfg["n_times"])
        self.sfreq = float(cfg["sfreq"])
        self.in_sfreq = float(in_sfreq)
        self.nets = [
            EEGNet(
                n_chans=len(cfg["channels"]), n_outputs=1, n_times=self.n_times,
                drop_prob=float(cfg["dropout"]),
            ).to(device).eval()
            for _ in cfg.get("weight_files", ["weights.pt"])
        ]

        lookup = {name.casefold(): i for i, name in enumerate(in_names)}
        self.src = [lookup.get(name.casefold(), -1) for name in cfg["channels"]]
        missing = [n for n, i in zip(cfg["channels"], self.src) if i < 0]
        print(
            f"[EEGNet-sleep] matched {len(self.src) - len(missing)}/{len(self.src)} "
            f"training channels by name among {len(in_names)} inputs; "
            f"zero-filled: {missing if missing else 'none'}"
        )
        if len(missing) == len(self.src):
            raise RuntimeError(
                f"no training channel found in the input: {list(in_names)[:8]}"
            )

    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        out = torch.zeros(X.shape[0], len(self.src), X.shape[-1], device=self.device)
        for dst, src in enumerate(self.src):
            if src >= 0:
                out[:, dst] = X[:, src]
        # Match the training rate and window length. At the harness's own
        # settings both are no-ops.
        if self.in_sfreq != self.sfreq:
            out = resample_fft(
                out, int(round(out.shape[-1] * self.sfreq / self.in_sfreq))
            )
        if out.shape[-1] > self.n_times:
            out = out[..., : self.n_times]
        elif out.shape[-1] < self.n_times:
            out = F.pad(out, (0, self.n_times - out.shape[-1]))
        preds = [(net(out).squeeze(-1) * self.cap).clamp(0.0, self.cap)
                 for net in self.nets]
        return torch.stack(preds).mean(0)


class Solver(CompetSolver):

    name = "EEGNet-sleep-onset"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        cfg = json.loads((sub / "sleep_eegnet_config.json").read_text())
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        model = ChannelMatchedRegressor(cfg, in_names, meta["sfreq"], meta["device"])
        for net, name in zip(model.nets, cfg.get("weight_files", ["weights.pt"])):
            state = torch.load(
                sub / name, map_location=meta["device"], weights_only=True
            )
            net.load_state_dict(state)
        return model
