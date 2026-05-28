import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CHAMPIONS = 58
NUM_CLASSES = NUM_CHAMPIONS + 1
BOARD_HEIGHT = 4
BOARD_WIDTH = 7


class BoardGenerator(nn.Module):
    BOTTLENECK_CHANNELS = 512
    BOTTLENECK_H = 1
    BOTTLENECK_W = 2
    BOTTLENECK_SIZE = BOTTLENECK_CHANNELS * BOTTLENECK_H * BOTTLENECK_W  # 1024

    def __init__(self, input_dim: int = 116):
        super().__init__()
        self.input_dim = input_dim

        self.fc = nn.Linear(input_dim, self.BOTTLENECK_SIZE)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(2, 4),
            stride=(2, 2),
            padding=(0, 1),
        )
        self.bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=NUM_CLASSES,
            kernel_size=(2, 3),
            stride=(2, 2),
            padding=(0, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(-1, self.BOTTLENECK_CHANNELS, self.BOTTLENECK_H, self.BOTTLENECK_W)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        out = F.softmax(x, dim=1)
        return out
