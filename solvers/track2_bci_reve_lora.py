"""Track 02 submission: pretrained REVE with LoRA adapters and a head, averaged over seeds.

The encoder is `brain-bzh/reve-base`. Its weights ship once, in `encoder.pt`, and stay
frozen. Each seed ships only its LoRA adapters (rank r, on the 44 attention projections
`transformer.layers.*.0.{to_qkv,to_out}`, and optionally a weak adapter on the two
feed-forward matrices of each block, `transformer.layers.*.1.net.{1,3}`) and its head, in
`member_<i>.pt`. The head is per-electrode (Track 1's), a pooled-query head, or the plain
average, as the config says. The
members share one copy of the encoder in memory, because the harness deep-copies the
submitted model for every session.

Input handling repeats the training notebook: electrodes matched to the training ones by
name (extras ignored, missing ones zero-filled), the window cropped to the first
`n_times` samples (the notebook's CROP_S), per-window per-channel standardisation, then
resampling from the harness's 120 Hz to REVE's 200 Hz. With `input_norm: window` the model
keeps no state; with `input_norm: session` the windows are scaled by the running power of their own session (causal;
state cleared by `reset_state`). Class probabilities are averaged over members before the
argmax.

The model definition matches `track2_03_sealed_eval.ipynb`; keep the two in step.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import REVE

REVE_SFREQ = 200.0     # REVE's pretraining rate
PATCH = 200            # REVE's patch_size in samples (1 s)
LORA_TARGETS = ("to_qkv", "to_out")
FFN_TARGETS = (".net.1", ".net.3")       # FeedForward.net is Sequential(norm, Linear, GEGLU, Linear)


def resample_fft(x, n_out):
    """Resample along time via the rfft spectrum. Pure torch: nothing is installed at
    evaluation."""
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


def standardize(x):
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(
        x.var(-1, keepdim=True, unbiased=False) + 1e-8
    )


class LoRALinear(nn.Module):
    """Frozen base plus a rank-r update. Names (`base`, `A`, `B`) match training."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale

    @property
    def weight(self):
        """Effective weight, W + scale * B A. braindecode >= 2.0 reads `net[1].weight` of the
        feed-forward GEGLU directly instead of calling the module, which would skip the
        adapter; with this property the adapter is applied there too. (The input dropout of
        this adapter does not apply on that path.)"""
        return self.base.weight + (self.B @ self.A) * self.scale


class PerElectrodeHead(nn.Module):
    """Shrink each electrode's embedding with weights shared across electrodes (one linear
    layer, or a funnel of several), then a final layer with separate weights per electrode.
    norm: a LayerNorm after every funnel linear layer, which keeps the signal size steady.
    init: 'kaiming' gives the funnel layers He starting weights (PyTorch's default shrinks the
    signal at every layer)."""

    def __init__(self, n_chans, embed_dim, dims, n_outputs, dropout=0.0, mid_dropout=0.0,
                 norm=False, init='default'):
        super().__init__()
        dims = [dims] if isinstance(dims, int) else list(dims)
        self.norm = nn.LayerNorm(embed_dim)
        if len(dims) == 1 and not norm:
            self.shrink = nn.Linear(embed_dim, dims[0])        # keeps the older key names
        else:
            layers, prev = [], embed_dim
            for i, d in enumerate(dims):
                layers.append(nn.Linear(prev, d))
                prev = d
                if norm:
                    layers.append(nn.LayerNorm(d))
                if i < len(dims) - 1:
                    layers += [nn.GELU(), nn.Dropout(mid_dropout)]
            self.shrink = nn.Sequential(*layers)
        if init == 'kaiming':
            for m in self.shrink.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_chans * dims[-1], n_outputs)

    def forward(self, tokens):                    # (B, chan, embed)
        z = self.drop(F.gelu(self.shrink(self.norm(tokens))))
        return self.out(z.flatten(1))


class MQPoolHead(nn.Module):
    """K learned queries, each pooling the electrodes by a content-weighted average."""

    def __init__(self, k, embed_dim, n_outputs, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, embed_dim) / embed_dim ** 0.5)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(k * embed_dim, n_outputs)

    def forward(self, tokens):                    # (B, chan, embed)
        z = self.norm(tokens)
        w = torch.einsum("ke,bce->bkc", self.queries, z) / self.queries.shape[-1] ** 0.5
        w = w.softmax(dim=-1)
        pooled = torch.einsum("bkc,bce->bke", w, z)
        return self.out(self.drop(F.gelu(pooled.flatten(1))))


def is_adapter_key(k):
    return k.endswith((".A", ".B")) or k.startswith("head.")


class ReveLoRA(nn.Module):

    def __init__(self, channels, n_classes, in_sfreq, rank, encoder_state,
                 ffn_rank=0, ffn_alpha=0, head_kind="mean", shrink=16, mq_k=2, head_dims=None, head_norm=False):
        super().__init__()
        self.head_kind = head_kind
        self.in_sfreq = float(in_sfreq)
        self.net = REVE(
            n_outputs=n_classes, n_chans=len(channels), n_times=PATCH,
            sfreq=REVE_SFREQ, chs_info=[{"ch_name": c} for c in channels],
        )
        self.net.final_layer = nn.Identity()
        res = self.net.load_state_dict(encoder_state, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"encoder.pt does not fit REVE: {res.unexpected_keys[:4]}")
        for p in self.net.parameters():
            p.requires_grad_(False)
        names = [
            n for n, m in self.net.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in LORA_TARGETS
        ]
        for name in names:
            parent = self.net.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.split(".")[-1],
                    LoRALinear(self.net.get_submodule(name), rank, 2 * rank))
        if ffn_rank:                    # the feed-forward matrices, two per block
            ffn_names = [
                n for n, m in self.net.named_modules()
                if isinstance(m, nn.Linear) and n.startswith("transformer.layers.")
                and n.endswith(FFN_TARGETS)
            ]
            for name in ffn_names:
                parent = self.net.get_submodule(name.rsplit(".", 1)[0])
                setattr(parent, name.split(".")[-1],
                        LoRALinear(self.net.get_submodule(name), ffn_rank, ffn_alpha))
        E = self.net.embed_dim
        if head_kind == "perelec":
            self.head = PerElectrodeHead(len(channels), E, head_dims or shrink, n_classes, norm=head_norm)
        elif head_kind == "mqpool":
            self.head = MQPoolHead(mq_k, E, n_classes)
        else:
            self.head = nn.Linear(E, n_classes)

    def tie_encoder_to(self, other):
        """Share every frozen tensor with `other`, so ensemble members hold one encoder."""
        theirs = dict(other.named_parameters())
        for name, _ in list(self.named_parameters()):
            if is_adapter_key(name):
                continue
            parent = self.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], theirs[name])

    def forward(self, x):
        x = resample_fft(x.float(), int(round(x.shape[-1] * REVE_SFREQ / self.in_sfreq)))
        feats = self.net(x, return_features=True)["features"]     # (B, chan, patch, embed)
        if self.head_kind == "mean":
            return self.head(feats.mean(dim=(1, 2)))
        return self.head(feats.mean(dim=2))                       # average the time patches only


class NameMatchedEnsemble:
    """Reorders incoming electrodes into the trained order, zero-filling gaps, and
    averages the members' class probabilities."""

    def __init__(self, members, channels, in_names, n_times, device,
                 input_norm="window", prior_var=None, n0=5.0):
        self.members, self.device, self.n_times = members, device, int(n_times)
        self.input_norm, self.n0 = input_norm, float(n0)
        self.prior_var = None if prior_var is None else torch.as_tensor(prior_var, dtype=torch.float64)
        lookup = {n.casefold(): i for i, n in enumerate(in_names)}
        self.src = [lookup.get(c.casefold(), -1) for c in channels]
        missing = [c for c, i in zip(channels, self.src) if i < 0]
        print(
            f"[REVE-LoRA-BCI] matched {len(channels) - len(missing)}/{len(channels)} "
            f"training channels by name among {len(in_names)} inputs; "
            f"zero-filled: {missing if missing else 'none'}; "
            f"{len(members)} ensemble members"
        )
        if len(missing) == len(self.src):
            raise RuntimeError(
                f"no training channel found in the input: {list(in_names)[:8]}"
            )
        self.reset_state()

    def reset_state(self):
        # Only the session-scaled model has state: the summed power of the windows seen so far.
        self.var_sum = torch.zeros(len(self.src), dtype=torch.float64)
        self.count = 0

    def _session_scale(self, out):
        """Causal per-session scaling (same as `session_norm` in the training notebook): remove each window's
        mean, divide each electrode by the root of the mean power of the EARLIER windows of the session
        (blended with a prior of n0 windows), then add this window's power to the running sum."""
        scaled = []
        for i in range(out.shape[0]):
            x = out[i].double().cpu()
            x = x - x.mean(-1, keepdim=True)
            scale = torch.sqrt((self.n0 * self.prior_var + self.var_sum) / (self.n0 + self.count))
            scaled.append((x / scale[:, None]).float())
            self.var_sum = self.var_sum + x.var(-1, unbiased=False)
            self.count += 1
        return torch.stack(scaled).to(out.device)

    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        out = torch.zeros(X.shape[0], len(self.src), X.shape[-1], device=self.device)
        for dst, src in enumerate(self.src):
            if src >= 0:
                out[:, dst] = X[:, src]
        if out.shape[-1] > self.n_times:
            out = out[..., : self.n_times]
        elif out.shape[-1] < self.n_times:
            out = F.pad(out, (0, self.n_times - out.shape[-1]))
        out = self._session_scale(out) if self.input_norm == "session" else standardize(out)
        probs = torch.stack([m(out).softmax(-1) for m in self.members]).mean(0)
        return probs.argmax(-1)


class Solver(CompetSolver):

    name = "REVE-LoRA-BCI"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        os.environ["REVE_POSITIONS_PATH"] = str(sub)        # bundled; no download at eval
        cfg = json.loads((sub / "reve_bci_config.json").read_text())
        if meta.get("n_classes") not in (None, cfg["n_classes"]):
            raise RuntimeError(
                f"trained for {cfg['n_classes']} classes, harness asks for "
                f"{meta['n_classes']}"
            )
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        enc_state = torch.load(sub / "encoder.pt", map_location="cpu", weights_only=True)
        members = []
        for fname in cfg["member_files"]:
            m = ReveLoRA(cfg["channels"], cfg["n_classes"], meta["sfreq"],
                         cfg["rank"], enc_state,
                         ffn_rank=cfg.get("ffn_rank", 0), ffn_alpha=cfg.get("ffn_alpha", 0),
                         head_kind=cfg.get("head_kind", "mean"),
                         shrink=cfg.get("head_shrink", 16), mq_k=cfg.get("mq_k", 2),
                         head_dims=cfg.get("head_dims"), head_norm=cfg.get("head_norm", False))
            state = torch.load(sub / fname, map_location="cpu", weights_only=True)
            res = m.load_state_dict(state, strict=False)
            absent = [k for k in res.missing_keys if is_adapter_key(k)]
            if res.unexpected_keys or absent:
                raise RuntimeError(f"{fname} mismatch: "
                                   f"{(res.unexpected_keys or absent)[:4]}")
            if members:
                m.tie_encoder_to(members[0])
            members.append(m)
        members = [m.to(meta["device"]).eval() for m in members]
        return NameMatchedEnsemble(members, cfg["channels"], in_names,
                                   cfg["n_times"], meta["device"],
                                   input_norm=cfg.get("input_norm", "window"),
                                   prior_var=cfg.get("prior_var"), n0=cfg.get("n0", 5.0))
