import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class CloudModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.ModuleList()

        # features.0 / features.1
        self.features.append(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False))
        self.features.append(nn.BatchNorm2d(64))

        # features.4: 64 → 64 (2 blocks)
        stage4 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.features.append(stage4)  # index 2 → checkpoint index 4

        # features.5: 64 → 128 (downsample)
        down5 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=1, bias=False),
            nn.BatchNorm2d(128)
        )
        stage5 = nn.Sequential(
            BasicBlock(64, 128, downsample=down5),
            BasicBlock(128, 128)
        )
        self.features.append(stage5)  # index 3 → checkpoint index 5

        # features.6: 128 → 256
        down6 = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )
        stage6 = nn.Sequential(
            BasicBlock(128, 256, downsample=down6),
            BasicBlock(256, 256)
        )
        self.features.append(stage6)  # index 4 → checkpoint index 6

        # features.7: 256 → 512
        down7 = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512)
        )
        stage7 = nn.Sequential(
            BasicBlock(256, 512, downsample=down7),
            BasicBlock(512, 512)
        )
        self.features.append(stage7)  # index 5 → checkpoint index 7

        # Global pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC = (512 CNN output + 4 meta) = 516
        self.fc = nn.Sequential(
            nn.Linear(516, 128),   # fc.0
            nn.ReLU(),
            nn.Linear(128, 32),    # fc.3
            nn.ReLU(),
            nn.Linear(32, 1)       # fc.5
        )

    def forward(self, x, meta):
        x = self.features[0](x)
        x = self.features[1](x)

        x = self.features[2](x)   # features.4
        x = self.features[3](x)   # features.5
        x = self.features[4](x)   # features.6
        x = self.features[5](x)   # features.7

        x = self.pool(x).flatten(1)

        # concat metadata (already shaped [B,4])
        x = torch.cat([x, meta], dim=1)

        return self.fc(x)
