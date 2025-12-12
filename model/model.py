import torch
from torch import nn, utils, optim, Tensor
import torch.nn.functional as F
import lightning as L


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.multihead = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.fc = nn.Linear(512, embed_dim)

    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = [ResBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)               # (N, 512, H', W')
        seq = x.flatten(2).transpose(1, 2)          # (N, L, 512)
        seq, _ = self.multihead(seq, seq, seq)      # (N, L, 512)
        x = seq.mean(dim=1)                         # (N, 512)
        x = self.fc(x)                              # (N, embed_dim)
        return x

class LitContrastive(L.LightningModule):
    def __init__(self, embed_dim=512, lr=1e-3, temperature=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Model(embed_dim=embed_dim)
        self.temperature = temperature
        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x_clean, x_aug = batch
        z_clean = F.normalize(self(x_clean), dim=1)
        z_aug = F.normalize(self(x_aug), dim=1)

        z = torch.cat([z_clean, z_aug], dim=0)
        sim = z @ z.t() / self.temperature
        N = z_clean.size(0)
        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        targets = torch.cat([torch.arange(N, 2 * N, device=z.device),
                             torch.arange(0, N, device=z.device)], dim=0)
        loss = F.cross_entropy(sim, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
