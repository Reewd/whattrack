import torch
from torch import nn, utils, optim, Tensor
import torch.nn.functional as F
import lightning as L
import math


class Model(nn.Module):
    def __init__(self, embed_dim: int = 512, patch_size: int = 8, num_heads: int = 8, num_layers: int = 6, hidden_dim: int = 2048):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: convert image patches to embeddings
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))  # max 1000 patches
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 1, H, W) - input spectrogram
        N = x.shape[0]
        
        # Patch embedding: (N, 1, H, W) -> (N, embed_dim, H/P, W/P)
        x = self.patch_embed(x)  # (N, embed_dim, n_patches_h, n_patches_w)
        
        # Flatten patches: (N, embed_dim, n_patches_h, n_patches_w) -> (N, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)  # (N, n_patches, embed_dim)
        
        # Add positional encoding
        n_patches = x.shape[1]
        x = x + self.pos_encoding[:, :n_patches, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(N, -1, -1)  # (N, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (N, 1+n_patches, embed_dim)
        
        # Transformer encoder
        x = self.transformer(x)  # (N, 1+n_patches, embed_dim)
        
        # Extract CLS token embedding
        x = x[:, 0]  # (N, embed_dim)
        
        # Final projection
        x = self.norm(x)
        x = self.head(x)  # (N, embed_dim)
        
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

    def validation_step(self, batch, batch_idx):
        x_clean, x_aug = batch
        z_clean = F.normalize(self(x_clean), dim=1)
        z_aug = F.normalize(self(x_aug), dim=1)
        
        # Positive similarity
        pos_sim = (z_clean * z_aug).sum(dim=1).mean()
        self.log("val_pos_sim", pos_sim, prog_bar=True)
        return pos_sim

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
