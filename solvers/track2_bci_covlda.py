"""Track 02 submission: covariance -> tangent space -> LDA, with a causal per-session reference.

No pretrained network and no GPU. For every window the model:
  1. matches incoming electrodes to the trained ones by name (extras ignored, missing ones zero-filled),
  2. band-filters the full window (FFT mask) in each band of the config,
  3. computes an OAS-shrunk electrode covariance matrix per band,
  4. whitens it against a reference covariance and takes the matrix log (the tangent space),
     listing the upper triangle (off-diagonals x sqrt(2)),
  5. applies a linear classifier (the exported LDA weights) and returns the arg-max class.

The reference is causal: the log-Euclidean mean of the windows BEFORE this one in the same session,
blended with the training prior (weight n0 windows). The only state is a running sum of log-covariances
and a window count; `reset_state` clears it, and the harness calls it at the start of every session.

The feature code matches `track2_03_sealed_eval.ipynb` (Step 2d); keep the two in step. Everything runs on CPU
in float64: a 41 x 41 eigendecomposition is faster there than a GPU round trip.
"""

import json

import torch

from benchmark_utils.base_solver import CompetSolver


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum. Pure torch: nothing is installed at evaluation."""
    n_in = x.shape[-1]
    if n_in == n_out:
        return x
    spec = torch.fft.rfft(x, dim=-1)
    out = torch.zeros(*spec.shape[:-1], n_out // 2 + 1, dtype=spec.dtype)
    keep = min(spec.shape[-1], out.shape[-1])
    out[..., :keep] = spec[..., :keep]
    return torch.fft.irfft(out, n=n_out, dim=-1) * (n_out / n_in)


def band_filter(x, lo, hi, fs):
    """x: (N, C, T). Zero every frequency outside [lo, hi]."""
    T = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(T, d=1.0 / fs)
    return torch.fft.irfft(spec * ((f >= lo) & (f <= hi)).to(x.dtype), n=T, dim=-1)


def oas_cov(x):
    """(N, C, T) -> (N, C, C) covariance, shrunk toward a scaled identity (OAS)."""
    _, C, T = x.shape
    x = x - x.mean(-1, keepdim=True)
    S = x @ x.transpose(1, 2) / T
    alpha = (S ** 2).mean((1, 2))
    mu = S.diagonal(dim1=1, dim2=2).sum(-1) / C
    rho = torch.clamp((alpha + mu ** 2) / ((T + 1.0) * (alpha - mu ** 2 / C)), max=1.0)
    return (1 - rho)[:, None, None] * S + (rho * mu)[:, None, None] * torch.eye(C, dtype=x.dtype)


def sym_fn(M, fn):
    """Apply fn to the eigenvalues of symmetric matrices."""
    w, V = torch.linalg.eigh(M)
    return (V * fn(w).unsqueeze(-2)) @ V.transpose(-1, -2)


def tangent_vec(Chat):
    """SPD (N, C, C) -> (N, C(C+1)/2): matrix log, upper triangle, off-diagonals x sqrt(2)."""
    C = Chat.shape[-1]
    L = sym_fn(Chat, torch.log)
    iu = torch.triu_indices(C, C)
    return L[:, iu[0], iu[1]] * torch.where(iu[0] == iu[1], 1.0, 2.0 ** 0.5)


class CovLDA:

    def __init__(self, cfg, state, in_names, in_sfreq):
        self.cfg = cfg
        self.bands = [(b[0], float(b[1]), float(b[2])) for b in cfg["bands"]]
        self.n0 = float(cfg["n0"])
        self.fs = float(cfg["sfreq"])
        self.in_sfreq = float(in_sfreq)
        self.prior = {name: state[f"prior_{name}"].double() for name, _, _ in self.bands}
        self.coef = state["coef"].double()                      # (classes, features)
        self.intercept = state["intercept"].double()
        lookup = {n.casefold(): i for i, n in enumerate(in_names)}
        self.channels = list(cfg["channels"])
        self.src = [lookup.get(c.casefold(), -1) for c in self.channels]
        missing = [c for c, i in zip(self.channels, self.src) if i < 0]
        print(
            f"[CovLDA-BCI] matched {len(self.channels) - len(missing)}/{len(self.channels)} "
            f"training channels by name among {len(in_names)} inputs; "
            f"zero-filled: {missing if missing else 'none'}; bands {[b[0] for b in self.bands]}"
        )
        if len(missing) == len(self.src):
            raise RuntimeError(f"no training channel found in the input: {list(in_names)[:8]}")
        self.reset_state()

    def reset_state(self):
        C = len(self.channels)
        self.sum = {name: torch.zeros(C, C, dtype=torch.float64) for name, _, _ in self.bands}
        self.count = 0

    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(X).detach().cpu().double()
        x = torch.zeros(X.shape[0], len(self.src), X.shape[-1], dtype=torch.float64)
        for dst, src in enumerate(self.src):
            if src >= 0:
                x[:, dst] = X[:, src]
        if self.in_sfreq != self.fs:
            x = resample_fft(x, int(round(x.shape[-1] * self.fs / self.in_sfreq)))
        out = []
        for i in range(x.shape[0]):                  # windows are handled in order, one at a time
            xi = x[i:i + 1]
            feats = []
            for name, lo, hi in self.bands:
                C = oas_cov(band_filter(xi, lo, hi, self.fs))
                L_ref = (self.n0 * self.prior[name] + self.sum[name]) / (self.n0 + self.count)
                A = sym_fn(L_ref, lambda w: torch.exp(-w / 2))
                feats.append(tangent_vec(A @ C @ A))
                self.sum[name] = self.sum[name] + sym_fn(C, torch.log)[0]
            self.count += 1
            out.append((torch.cat(feats, 1) @ self.coef.T + self.intercept).argmax(-1))
        return torch.cat(out)


class Solver(CompetSolver):

    name = "CovLDA-BCI"

    requirements = []

    def load_model(self, meta):
        sub = meta["submission_dir"]
        cfg = json.loads((sub / "covlda_config.json").read_text())
        if meta.get("n_classes") not in (None, cfg["n_classes"]):
            raise RuntimeError(
                f"trained for {cfg['n_classes']} classes, harness asks for {meta['n_classes']}"
            )
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        state = torch.load(sub / "covlda_state.pt", map_location="cpu", weights_only=True)
        return CovLDA(cfg, state, in_names, meta["sfreq"])
