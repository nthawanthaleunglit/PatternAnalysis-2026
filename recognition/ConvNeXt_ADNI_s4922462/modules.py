import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class BaselineCNN(nn.Module):

    # Baseline: four plain conv -> BatchNorm -> ReLU -> max-pool blocks, then global average pooling and a linear head.

    def __init__(self, num_classes=2, in_chans=1, widths=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], in_chans
        for c_out in widths:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, num_classes)

    def embed(self, x):
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.embed(x))


class LayerNorm2d(nn.LayerNorm):

    # LayerNorm over the channel dimension of an (N, C, H, W) tensor.

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DropPath(nn.Module):

    # Stochastic depth: during training, skip the whole residual branch for a random subset of samples.

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.p
        return x * keep / (1 - self.p)


class ConvNeXtBlock(nn.Module):

    # ConvNeXt block (Liu et al., 2022): 7x7 depthwise conv -> LayerNorm -> 1x1 expand 4x -> GELU -> 1x1 project,
    # scaled by a learnable per-channel factor (layer scale) and added back to the input.

    def __init__(self, dim, drop_path=0.0, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)  # to (N, H, W, C) so LayerNorm and Linear act on channels
        y = self.pw2(F.gelu(self.pw1(self.norm(y)))) * self.gamma
        return x + self.drop_path(y.permute(0, 3, 1, 2))


class ConvNeXt(nn.Module):

    # Reduced ConvNeXt: same layout as ConvNeXt-T (depths 3/3/9/3, dims 96..768) but with half the channels and
    # fewer blocks, to suit a dataset of about 900 training patients.
    # Stem: 4x4 stride-4 patchify conv. Between stages: LayerNorm + 2x2 stride-2 conv to halve resolution.

    def __init__(self, num_classes=2, in_chans=1, depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), drop_path=0.1):
        super().__init__()
        self.downsample = nn.ModuleList([nn.Sequential(nn.Conv2d(in_chans, dims[0], 4, stride=4), LayerNorm2d(dims[0]))])
        for i in range(len(dims) - 1):
            self.downsample.append(nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)))

        rates = torch.linspace(0, drop_path, sum(depths)).tolist()  # deeper blocks are dropped more often
        self.stages, k = nn.ModuleList(), 0
        for i, d in enumerate(depths):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i], rates[k + j]) for j in range(d)]))
            k += d

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def embed(self, x):
        for down, stage in zip(self.downsample, self.stages):
            x = stage(down(x))
        return self.norm(x.mean(dim=(2, 3)))  # global average pool, then LayerNorm

    def forward(self, x):
        return self.head(self.embed(x))


class PretrainedConvNeXt(nn.Module):

    # Full-size ConvNeXt-T (depths 3/3/9/3, dims 96..768) with torchvision's ImageNet-1K weights, fine-tuned on ADNI.
    # The stem's three RGB filters are summed into one grey filter, which is the same as feeding the slice as R = G = B.
    # The 1000-class ImageNet head is replaced by a new 2-class head.

    def __init__(self, num_classes=2, drop_path=0.1, pretrained=True):
        super().__init__()
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None,
                            stochastic_depth_prob=drop_path)
        rgb = net.features[0][0]
        grey = nn.Conv2d(1, rgb.out_channels, 4, stride=4)
        with torch.no_grad():
            grey.weight.copy_(rgb.weight.sum(dim=1, keepdim=True))
            grey.bias.copy_(rgb.bias)
        net.features[0][0] = grey
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
        self.net = net

    def embed(self, x):
        # features -> global average pool -> LayerNorm -> flatten, giving a 768-d vector per slice
        return self.net.classifier[:2](self.net.avgpool(self.net.features(x)))

    def forward(self, x):
        return self.net.classifier[2](self.embed(x))


def build_model(name, num_classes=2, drop_path=0.1, pretrained=True):
    # Create a model by name ('cnn', 'convnext' trained from scratch, or 'convnext_pt' fine-tuned from ImageNet).
    if name == "cnn":
        return BaselineCNN(num_classes)
    if name == "convnext":
        return ConvNeXt(num_classes, drop_path=drop_path)
    if name == "convnext_pt":
        return PretrainedConvNeXt(num_classes, drop_path=drop_path, pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}")
