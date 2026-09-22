"""Track 04 submission: NeuroPose, output layer zeroed to a neutral pose.

There is no pretrained EMG-to-pose checkpoint anywhere: VEMG2Pose, NeuroPose and
MetaNeuromotorHand are all absent or empty on the Hub. So unlike the EEG tracks
this ships no pretrained encoder, and the point of the run is to prove the
submission path rather than to score.

Why NeuroPose and not the provided `mean_pose.py` floor: that floor ships no
weight file, so it exercises neither `torch.load` nor `load_state_dict`, and its
`ConstantPose` only learns the train mean inside `fit`, which Codabench never
calls. Uploaded bare it predicts zeros while testing almost none of the path.
This predicts the same zeros and tests all of it.

Why NeuroPose and not VEMG2Pose, which is the stronger architecture: VEMG2Pose
consumes 1790 samples of left context before its first prediction, so it cannot
be contract-tested against `Simulated` at 400 samples.

Two details that would otherwise fail silently or loudly:

1. NeuroPose emits `(B, T, n_joints)`; the track wants `(B, n_joints, T)`.
2. Angles are returned in **degrees**, per the objective.

The output layer is zeroed rather than left at random init. Both score the same
— default init already emits within +/-0.6 degrees of zero — but zeroing makes
the prediction exactly a neutral pose and independent of seed, so the score
returned is interpretable as "predict 0 degrees everywhere" and tells us the
scale of the target distribution.
"""

import torch

from benchmark_utils.base_solver import CompetSolver
from braindecode.models import NeuroPose


class PoseRegressor(torch.nn.Module):
    """NeuroPose with the axis order the track expects."""

    def __init__(self, n_chans, n_times, n_joints, sfreq):
        super().__init__()
        self.net = NeuroPose(
            n_chans=n_chans, n_times=n_times, n_outputs=n_joints, sfreq=sfreq
        )

    def forward(self, X):
        return self.net(X).transpose(1, 2)      # (B, T, J) -> (B, J, T)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X)                          # degrees


class Solver(CompetSolver):

    name = "NeuroPose-neutral"

    def load_model(self, meta):
        n_joints = int(meta.get("n_joints") or meta["n_outputs"])
        model = PoseRegressor(
            n_chans=int(meta["n_chans"]),
            n_times=int(meta["n_times"]),
            n_joints=n_joints,
            sfreq=float(meta["sfreq"]),
        )
        state = torch.load(
            meta["submission_dir"] / "weights.pt",
            map_location=meta["device"],
            weights_only=True,
        )
        state = {f"net.{k}": v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in weights.pt: {unexpected[:5]}")
        if missing:
            raise RuntimeError(f"weights missing: {missing[:5]}")
        return model.to(meta["device"]).eval()
