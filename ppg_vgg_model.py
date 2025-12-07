# ppg_vgg_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPGVGGNet(nn.Module):
    """
    VGG-style 1D CNN for arrhythmia classification from 10s PPG (1000 samples).
    Input shape: (batch, 1, 1000)
    Output: logits for 6 classes.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        # VGG-like block: Conv1d + BN + ReLU repeated
        def conv_block(in_ch, out_ch, num_convs):
            layers = []
            for i in range(num_convs):
                layers.append(nn.Conv1d(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1
                ))
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1,   64, 2),   # -> (64, 500)
            conv_block(64,  128, 2),  # -> (128, 250)
            conv_block(128, 256, 3),  # -> (256, 125)
            conv_block(256, 512, 3),  # -> (512, ~62)
            conv_block(512, 512, 3),  # -> (512, ~31)
        )

        # compute flattened size after convs with dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 1000)
            out = self.features(dummy)
            self.flat_dim = out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)          # (B, C, L')
        x = x.view(x.size(0), -1)     # flatten
        x = self.classifier(x)        # (B, num_classes)
        return x


if __name__ == "__main__":
    # quick self-test
    model = PPGVGGNet(num_classes=6)
    x = torch.randn(8, 1, 1000)
    y = model(x)
    print("Output shape:", y.shape)
