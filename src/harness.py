"""Local stand-in for the competition harness, so the submission path can be
exercised before the starting kit is in hand.

The server runs a submission inference-only: it imports `submission.py`, calls
`Solver.load_model(meta)`, and feeds `predict(X)` batches of shape `(B, C, T)`
already on `meta["device"]`. Nothing in a NeuralBench training run touches any of
that, so a model can score well locally and still fail to load on upload.

This module mirrors that contract closely enough to catch the failures that
matter: a wrong output shape, a missing weight file, a solver that cannot be
imported, an inference pass that blows the runtime budget. Replace `CompetSolver`
here with the real `compet_core.base_solver.CompetSolver` once the kit ships; the
solver you write should not need to change.

Metric and scoring are numpy-only on purpose, so they can be tested without a GPU
or a torch install.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


class CompetSolver:
    """Stand-in base class matching the documented contract.

    Subclasses implement `load_model(meta)` returning an object with `predict(X)`.
    `fit` runs locally only and is never called by the server.
    """

    name: str = "unnamed"
    requirements: list[str] = []

    def load_model(self, meta: dict[str, Any]):
        raise NotImplementedError

    def fit(self, model, train_loader):  # never runs server-side
        return model


def make_meta(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float,
    weights_dir: Path,
    ch_names: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """The `meta` dict the server hands to `load_model`."""
    return {
        "n_chans": n_chans,
        "n_times": n_times,
        "n_outputs": n_outputs,
        "sfreq": sfreq,
        "ch_names": ch_names or [f"ch{i}" for i in range(n_chans)],
        "chs_info": None,
        "weights_dir": Path(weights_dir),
        "device": device,
    }


def simulated(
    n_samples: int = 64,
    n_chans: int = 32,
    n_times: int = 256,
    dim: int = 1536,
    seed: int = 33,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stand-in data, mirroring the competition's own `Simulated` set.

    Returns EEG `(N, C, T)`, per-trial target embeddings `(N, D)`, and the index of
    each trial's target within the gallery of unique embeddings.

    The signal carries a weak, learnable projection of the target so that a model
    scoring at chance indicates a broken pipeline rather than an impossible task.
    """
    rng = np.random.default_rng(seed)
    gallery = rng.standard_normal((n_samples, dim)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

    mixing = rng.standard_normal((dim, n_chans * n_times)).astype(np.float32) * 0.01
    signal = (gallery @ mixing).reshape(n_samples, n_chans, n_times)
    noise = rng.standard_normal((n_samples, n_chans, n_times)).astype(np.float32)
    X = signal + 0.5 * noise

    return X.astype(np.float32), gallery, np.arange(n_samples)


def top_k_retrieval(pred: np.ndarray, gallery: np.ndarray, truth: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy (%) ranking each prediction against the whole gallery.

    Cosine similarity, matching a retrieval scored in a frozen embedding space.
    """
    if pred.shape[1] != gallery.shape[1]:
        raise ValueError(f"dim mismatch: predictions {pred.shape[1]}, gallery {gallery.shape[1]}")
    p = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    sims = p @ g.T
    k = min(k, gallery.shape[0])
    topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    return 100.0 * np.mean([t in row for t, row in zip(truth, topk)])


def chance_top_k(gallery_size: int, k: int = 5) -> float:
    """Chance accuracy (%). Always report this next to a retrieval score."""
    return 100.0 * min(k, gallery_size) / gallery_size


def load_solver(submission_dir: Path):
    """Import `submission.py` the way the ingestion program does."""
    submission_dir = Path(submission_dir)
    path = submission_dir / "submission.py"
    if not path.exists():
        raise FileNotFoundError(f"no submission.py in {submission_dir}")

    spec = importlib.util.spec_from_file_location("_submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_submission"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "Solver"):
        raise AttributeError("submission.py defines no class named Solver")
    return module.Solver()


@dataclass
class ScoreReport:
    top5: float
    top1: float
    chance_top5: float
    gallery_size: int
    n_samples: int
    load_s: float
    inference_s: float

    def budget_fraction(self, budget_min: float = 60.0) -> float:
        return self.inference_s / (budget_min * 60.0)

    def __str__(self) -> str:
        return (
            f"top5 {self.top5:.2f}%  top1 {self.top1:.2f}%  "
            f"(chance top5 {self.chance_top5:.2f}%, gallery {self.gallery_size:,})\n"
            f"load {self.load_s:.1f}s  inference {self.inference_s:.1f}s "
            f"({self.budget_fraction():.1%} of the 60 min budget)"
        )


def run_local(
    submission_dir: Path,
    X: np.ndarray,
    gallery: np.ndarray,
    truth: np.ndarray,
    meta: dict[str, Any],
    batch_size: int = 32,
    to_batch: Callable[[np.ndarray], Any] | None = None,
) -> ScoreReport:
    """Ingestion plus scoring, end to end, the way the server would run it.

    `to_batch` converts a numpy batch into whatever `predict` expects. The real
    harness passes torch tensors on `meta["device"]`; pass a converter to match.
    """
    solver = load_solver(submission_dir)

    t0 = time.perf_counter()
    model = solver.load_model(meta)
    load_s = time.perf_counter() - t0
    if not hasattr(model, "predict"):
        raise AttributeError(f"{type(model).__name__} exposes no predict()")

    t0 = time.perf_counter()
    chunks = []
    for start in range(0, len(X), batch_size):
        batch = X[start : start + batch_size]
        out = model.predict(to_batch(batch) if to_batch else batch)
        chunks.append(np.asarray(out.detach().cpu() if hasattr(out, "detach") else out))
    pred = np.concatenate(chunks, axis=0)
    inference_s = time.perf_counter() - t0

    expected = (len(X), meta["n_outputs"])
    if pred.shape != expected:
        raise ValueError(f"predict returned {pred.shape}, server expects {expected}")

    return ScoreReport(
        top5=top_k_retrieval(pred, gallery, truth, k=5),
        top1=top_k_retrieval(pred, gallery, truth, k=1),
        chance_top5=chance_top_k(len(gallery)),
        gallery_size=len(gallery),
        n_samples=len(X),
        load_s=load_s,
        inference_s=inference_s,
    )
