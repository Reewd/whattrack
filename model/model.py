import torch
from torch import nn, utils, optim, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torch.nn.functional as F
from torchaudio import transforms as T
import lightning as L
import math


class Model(nn.Module):
    def __init__(self, embed_dim: int = 512, patch_size: int = 8, num_heads: int = 8, num_layers: int = 6, hidden_dim: int = 2048, num_classes: int = None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.spec_transform = T.MelSpectrogram(
            sample_rate=8000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        
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
        
        # Optional classifier head for genre classification
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.classifier = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, n_samples) - input waveform
        N = x.shape[0]

        x = self.spec_transform(x)
        x = torch.log(x + 1e-9).unsqueeze(1)  # Log-mel spectrogram
        
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
        embedding = self.head(x)  # (N, embed_dim)
        
        # If classifier exists, return both embedding and logits
        if self.classifier is not None:
            logits = self.classifier(embedding)  # (N, num_classes)
            return embedding, logits
        
        return embedding

class LitContrastive(L.LightningModule):
    def __init__(self, embed_dim=512, lr=1e-3, temperature=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Model(embed_dim=embed_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature))
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

    def configure_optimizers(self): # type: ignore
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        # Use LinearLR for warmup - increases LR from 0 to target over warmup_steps
        warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps,  # type: ignore
            eta_min=self.lr * 2e-2 # Minimum LR at end
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, scheduler], # type: ignore
            milestones=[warmup_steps]
        )
    

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Update every batch!
                "frequency": 1
            }
        }

class LitGenreClassifier(L.LightningModule):
    def __init__(self, num_classes: int, pretrained_encoder: Model, lr: float = 1e-3, freeze_encoder: bool = True):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_encoder'])
        
        # Use pretrained encoder (frozen by default)
        self.encoder = pretrained_encoder
        
        # Freeze encoder weights to prevent retraining
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Create classifier MLP on top of frozen encoder
        embed_dim = self.encoder.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )
        
        self.num_classes = num_classes
        self.lr = lr

    def forward(self, x):
        # Get embedding from frozen encoder
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            embedding = self.encoder(x)
            # Handle case where encoder might return tuple
            if isinstance(embedding, tuple):
                embedding = embedding[0]
        
        # Classify using trainable MLP
        logits = self.classifier(embedding)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only optimize classifier parameters (encoder is frozen)
        optimizer = optim.AdamW(self.classifier.parameters(), lr=self.lr)
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)
        
        warmup_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
        )
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps,
            eta_min=self.lr * 2e-2
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }