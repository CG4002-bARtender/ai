import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.skip(x))
        return self.pool(out)


class SmallResNet(nn.Module):
    # channels: stem output size, then one entry per residual block.
    # The first two blocks downsample (pool=True); remaining blocks do not.
    def __init__(self, n_classes: int, channels: list = None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 128]

        stem_ch = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = stem_ch
        for i, out_ch in enumerate(channels[1:]):
            pool = (i < 2)  # first two transitions downsample
            blocks.append(ResBlock(in_ch, out_ch, pool=pool))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.n_features = in_ch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.n_features, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.fc(x)
