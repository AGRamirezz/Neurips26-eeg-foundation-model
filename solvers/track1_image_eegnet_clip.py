"""Track 01 submission: EEGNet trained with a CLIP retrieval loss.

The model and loss follow the organisers' `eegnet_clip.py` baseline; the weights
come from `track1_07_eegnet.ipynb`, trained on Alljoined-1.6M with the harness's
own preprocessing, so the scoring data is prepared exactly as the training data
was.

EEGNet learns one spatial weight per electrode, in the order it was trained on.
The scoring data may come from a different headset (the warm-up uses a 63-channel
cap) or list the same electrodes in a different order. Inputs are therefore
matched to the training electrodes by name. Electrodes the model never saw are
ignored, and training electrodes absent from the input are filled with zeros:
after the harness's scaling and baseline a real channel sits near zero, so zero
reads as "no information" rather than as signal.
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


class ChannelMatchedEEGNet:

    def __init__(self, cfg, in_names, in_sfreq, device):
        self.device = device
        self.cfg = cfg
        self.n_times = int(cfg["n_times"])
        self.sfreq = float(cfg["sfreq"])
        self.in_sfreq = float(in_sfreq)
        self.net = EEGNet(
            n_chans=len(cfg["channels"]),
            n_outputs=int(cfg["n_outputs"]),
            n_times=self.n_times,
        ).to(device)

        # For each training electrode, the index of the same electrode in the
        # input, or -1 if the input does not have it. Case is folded because
        # montages disagree on it (FP1 vs Fp1).
        lookup = {name.casefold(): i for i, name in enumerate(in_names)}
        self.src = [lookup.get(name.casefold(), -1) for name in cfg["channels"]]
        matched = [n for n, i in zip(cfg["channels"], self.src) if i >= 0]
        missing = [n for n, i in zip(cfg["channels"], self.src) if i < 0]
        print(
            f"[EEGNet-CLIP] matched {len(matched)}/{len(cfg['channels'])} training "
            f"channels by name among {len(in_names)} inputs; "
            f"zero-filled: {missing if missing else 'none'}"
        )
        if not matched:
            raise RuntimeError(
                f"no training channel found in the input: {list(in_names)[:8]}"
            )

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        out = torch.zeros(
            X.shape[0], len(self.src), X.shape[-1], device=self.device
        )
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
        return self.net(out)


class Solver(CompetSolver):

    name = "EEGNet-CLIP-Alljoined"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        cfg = json.loads((sub / "eegnet_config.json").read_text())
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        model = ChannelMatchedEEGNet(cfg, in_names, meta["sfreq"], meta["device"])
        state = torch.load(
            sub / "weights.pt", map_location=meta["device"], weights_only=True
        )
        model.net.load_state_dict(state)
        return model
