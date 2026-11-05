"""Heterogeneous electrode/time mask generator for EEG fine-tuning.

Design, from the masking preprint (pierregtch.github.io/eeg-fm-masking) plus our own
choices (marked "ours"). The preprint found that hiding irregular patches of *neighbouring*
electrodes for a short stretch of time works best for pretraining, and that isolated single
tokens (random patch masking) works worst, because neighbours let the model interpolate.
Its mask ratio was 0.55; for fine-tuning we default lower (ours, untested).

A mask is a boolean array over the token grid (electrode, time token); True = hidden.

How it is drawn (ours):
  1. Per sample, draw a few "bumps". Each has a centre electrode, a spatial radius and a
     temporal length, each drawn from a log-normal so sizes are heterogeneous.
  2. The bumps add up to a smooth field over (electrode, time). White noise is mixed in.
  3. The top `ratio` fraction of cells by field value is hidden. The threshold is a rank
     cut, so the hidden fraction is exact and the pattern is non-linear in the field.
  4. `ratio` itself is redrawn per sample (Beta), so masking depth varies too.

`noise` is the dial between blocky and scattered: 0 gives clean blobs, large values
approach independent per-token masking (the weak setting). Bump centres are weighted
against local electrode density so no electrode is systematically hidden more than others.
"""

import numpy as np


class ElectrodeMasker:
    """positions: (C, 3) electrode coordinates in metres (REVE's position bank is in metres,
    head radius about 0.095). n_time: number of time tokens per window."""

    def __init__(self, positions, n_time=1, ratio=0.2, ratio_concentration=8.0,
                 spatial_scale=0.05, temporal_scale=1.0, scale_sigma=0.5,
                 n_bumps=(2, 5), noise=0.3, min_visible_frac=0.4, balance=True):
        pos = np.asarray(positions, dtype=float)
        self.C, self.T = len(pos), int(n_time)
        self.D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)       # (C, C) metres
        self.ratio, self.conc = float(ratio), float(ratio_concentration)
        self.sp, self.tp, self.sig = float(spatial_scale), float(temporal_scale), float(scale_sigma)
        self.n_bumps, self.noise = n_bumps, float(noise)
        self.min_visible = max(1, int(round(min_visible_frac * self.C)))
        if balance:
            density = np.exp(-(self.D ** 2) / (2 * self.sp ** 2)).sum(1)
            w = 1.0 / density
        else:
            w = np.ones(self.C)
        self.centre_p = w / w.sum()

    def _one(self, rng):
        C, T = self.C, self.T
        a, b = self.ratio * self.conc, (1 - self.ratio) * self.conc
        ratio = float(rng.beta(a, b)) if 0 < self.ratio < 1 else self.ratio
        field = np.zeros((C, T))
        t_axis = np.arange(T)
        for _ in range(int(rng.integers(self.n_bumps[0], self.n_bumps[1] + 1))):
            c = rng.choice(C, p=self.centre_p)
            r = self.sp * np.exp(self.sig * rng.standard_normal())
            L = self.tp * np.exp(self.sig * rng.standard_normal())
            t0 = rng.uniform(-0.5, T - 0.5)
            amp = rng.uniform(0.5, 1.5)
            field += amp * np.exp(-self.D[c] ** 2 / (2 * r ** 2))[:, None] \
                * np.exp(-((t_axis - t0) ** 2) / (2 * L ** 2))[None]
        field = (field - field.mean()) / (field.std() + 1e-9)
        field = field + self.noise * rng.standard_normal((C, T)) + 1e-6 * rng.standard_normal((C, T))
        k = int(round(ratio * C * T))
        k = min(k, (C - self.min_visible) * T)
        mask = np.zeros(C * T, dtype=bool)
        mask[np.argsort(-field.ravel())[:k]] = True
        mask = mask.reshape(C, T)
        # every time token keeps at least `min_visible` electrodes
        for t in range(T):
            over = mask[:, t].sum() - (C - self.min_visible)
            if over > 0:
                hidden = np.flatnonzero(mask[:, t])
                mask[hidden[np.argsort(field[hidden, t])[:over]], t] = False
        return mask

    def __call__(self, batch_size, rng=None):
        """(B, C, T) boolean mask, True = hidden. Pass a numpy Generator for repeatability."""
        rng = rng if rng is not None else np.random.default_rng()
        return np.stack([self._one(rng) for _ in range(batch_size)])


def expand_mask(mask, patch_len, n_samples):
    """(B, C, T) token mask -> (B, C, n_samples) sample mask, each token covering patch_len."""
    out = np.repeat(mask, patch_len, axis=-1)[..., :n_samples]
    if out.shape[-1] < n_samples:                       # pad with the last token's value
        out = np.concatenate([out, np.repeat(out[..., -1:], n_samples - out.shape[-1], -1)], -1)
    return out


def apply_mask(x, mask, patch_len):
    """Zero the hidden electrode-tokens of x (B, C, n_samples). Call it AFTER per-window
    standardisation, so zero means "no information", as for an electrode that is absent at
    inference (the solvers zero-fill missing electrodes). Works on numpy arrays or torch
    tensors."""
    m = expand_mask(np.asarray(mask), patch_len, x.shape[-1])
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.masked_fill(torch.as_tensor(m, device=x.device), 0.0)
    except ImportError:
        pass
    return np.where(m, 0.0, x)
