from model.model import LitContrastive
from model.dataset import AudioDataModule
from audio.augmentation import AudioAugmentations
from audio.augmentations.background_noise_mixing import BackgroundNoiseMixing
from audio.augmentations.ir_noise_mixing import ImpulseResponseAugmentation
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor

def main():
    train_path = "dataset/music/train-10k-30s"
    val_path = "dataset/music/val-query-db-500-30s"
    test_path = "dataset/music/test-query-db-500-30s/db"
    aug_dir = "dataset/aug"
    wandb_logger = WandbLogger(project="FDS")
    model = LitContrastive()

    augmentations = AudioAugmentations(
    enabled_augmentations=[
        BackgroundNoiseMixing(
            files_path=f"{aug_dir}/bg",
            train=True,
            snr_range=(0, 10),
            sample_rate=8000
        ),
        ImpulseResponseAugmentation(
            ir_path=f"{aug_dir}/ir",
            train=True,
            sample_rate=8000
        )
    ]
)

    dm = AudioDataModule(train_path=train_path, val_path=val_path, test_path=test_path, num_workers=32, train_augmentations=augmentations, batch_size=300)
    
    # Configure checkpoint callback to save best model based on lowest loss
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    
    # Monitor GPU usage metrics
    device_stats = DeviceStatsMonitor()
    
    trainer = L.Trainer(max_epochs=5, logger=wandb_logger, callbacks=[checkpoint_callback, device_stats])
    dm.setup()
    trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    
    
if __name__ == "__main__":
    main()
