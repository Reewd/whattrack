from model.model import LitContrastive
from model.dataset import AudioDataModule
from audio.augmentation import AudioAugmentations
from audio.augmentations.background_noise_mixing import BackgroundNoiseMixing
from audio.augmentations.ir_noise_mixing import ImpulseResponseAugmentation
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
import argparse
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    argparser.add_argument('--train-path', type=str, default='dataset/music/train-10k-30s', help='Path to training dataset')
    argparser.add_argument('--val-path', type=str, default='dataset/music/val-query-db-500-30s', help='Path to validation dataset')
    argparser.add_argument('--test-path', type=str, default='dataset/music/test-query-db-500-30s/db', help='Path to test dataset')  
    argparser.add_argument('--aug-dir', type=str, default='dataset/aug', help='Path to augmentation directory')
    argparser.add_argument('--batch-size', type=int, default=300, help='Batch size for training')
    argparser.add_argument('--num-workers', type=int, default=32, help='Number of workers for data loading')
    argparser.add_argument('--max-epochs', type=int, default=5, help='Maximum number of training epochs')
    args = argparser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
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

    dm = AudioDataModule(train_path=train_path, val_path=val_path, test_path=test_path, num_workers=args.num_workers, train_augmentations=augmentations, batch_size=args.batch_size)
    
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
    
    trainer = L.Trainer(max_epochs=args.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, device_stats])
    dm.setup()
    trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    
    
if __name__ == "__main__":
    main()
