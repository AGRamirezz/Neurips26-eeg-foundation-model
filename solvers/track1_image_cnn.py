"""Track 01 submission: a CNN that reads the raw EEG window as an image.

The window (electrodes x time, 120 Hz) is standardised per window and fed to either a ResNet18
(ResNet18/50/101: ImageNet weights adapted to one input channel, then fine-tuned with LoRA on the
convolutions, merged into plain weights before export) or a small CNN trained from scratch. A small head maps
the pooled features to the 1536-d target embedding. Nothing depends on an install at evaluation
beyond torch: the networks are plain torch modules defined here, with ResNet18 parameter names
identical to torchvision's, so the pretrained weights load by name.

Inputs are matched to the electrodes the model was trained on by name (case-insensitive); extra
electrodes are ignored and missing ones are zero-filled. Row order in the image is the trained
order, so the same window gives the same output whatever order the harness sends it in.

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                            nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, planes, stride):
        super().__init__()
        cout = planes * 4
        self.conv1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                            nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + idt)


RESNETS = {"resnet18": (BasicBlock, (2, 2, 2, 2), 512),
           "resnet50": (Bottleneck, (3, 4, 6, 3), 2048),
           "resnet101": (Bottleneck, (3, 4, 23, 3), 2048)}


class ResNet(nn.Module):
    """torchvision's resnet18/50/101 without the classifier, one input channel; same parameter names."""

    def __init__(self, arch="resnet18"):
        super().__init__()
        block, layers, self.feat_dim = RESNETS[arch]
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin = 64
        for i, (planes, n) in enumerate(zip((64, 128, 256, 512), layers), 1):
            stride = 1 if i == 1 else 2
            blocks = []
            for j in range(n):
                blocks.append(block(cin, planes, stride if j == 0 else 1))
                cin = planes * block.expansion
            setattr(self, f"layer{i}", nn.Sequential(*blocks))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.mean(dim=(2, 3))


class SmallCNN(nn.Module):
    """Four conv blocks (32, 64, 128, 256), from scratch."""
    feat_dim = 256

    def __init__(self):
        super().__init__()
        layers, cin = [], 1
        for cout in (32, 64, 128, 256):
            layers += [nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
                       nn.GELU(), nn.MaxPool2d(2, ceil_mode=True)]
            cin = cout
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x).mean(dim=(2, 3))


class EEGImageNet(nn.Module):
    def __init__(self, channels, n_outputs, in_sfreq, arch="resnet50", shrink=64, dropout=0.4):
        super().__init__()
        self.in_sfreq = float(in_sfreq)
        self.backbone = ResNet(arch) if arch in RESNETS else SmallCNN()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feat_dim, shrink),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(shrink, n_outputs))

    def forward(self, X):                          # (B, n_channels in trained order, T)
        X = X.float()
        n_out = max(int(round(X.shape[-1] * TARGET_SFREQ / self.in_sfreq)), 8)
        X = resample_fft(X, n_out)
        X = (X - X.mean(dim=(1, 2), keepdim=True)) / (X.std(dim=(1, 2), keepdim=True) + 1e-8)
        X = torch.clamp(X, -CLIP_SD, CLIP_SD)
        return self.head(self.backbone(X[:, None]))


class NameMatched:
    """Reorders incoming electrodes into the trained order, zero-filling gaps."""

    def __init__(self, model, channels, in_names, device):
        self.model, self.device = model, device
        lookup = {n.casefold(): i for i, n in enumerate(in_names)}
        self.src = [lookup.get(c.casefold(), -1) for c in channels]
        missing = [c for c, i in zip(channels, self.src) if i < 0]
        print(f"[CNN] matched {len(channels) - len(missing)}/{len(channels)} training channels by "
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

    name = "EEG-image-CNN"

    def load_model(self, meta):
        sub = meta["submission_dir"]
        cfg = json.loads((sub / "cnn_config.json").read_text())
        in_names = meta.get("ch_names") or [c["ch_name"] for c in meta["chs_info"]]
        model = EEGImageNet(cfg["channels"], cfg["n_outputs"], meta["sfreq"], arch=cfg["arch"],
                            shrink=cfg["shrink"], dropout=cfg["dropout"])
        state = torch.load(sub / "weights.pt", map_location=meta["device"], weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or missing:
            raise RuntimeError(f"weights mismatch: missing {missing[:4]}, unexpected {unexpected[:4]}")
        return NameMatched(model.to(meta["device"]).eval(), cfg["channels"], in_names,
                           meta["device"])
