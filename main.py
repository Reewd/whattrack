from model.model import LitContrastive
from model.dataset import AudioDataModule
from audio.augmentation import AudioAugmentations
from audio.augmentations import *
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
import argparse
import torch
import warnings

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    argparser.add_argument('--train-path', type=str, default='dataset/music/train-10k-30s', help='Path to training dataset')
    argparser.add_argument('--val-path', type=str, default='dataset/music/val-query-db-500-30s', help='Path to validation dataset')
    argparser.add_argument('--test-path', type=str, default='dataset/music/test-query-db-500-30s/db', help='Path to test dataset')  
    argparser.add_argument('--aug-dir', type=str, default='dataset/aug', help='Path to augmentation directory')
    argparser.add_argument('--batch-size', type=int, default=300, help='Batch size for training')
    argparser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading')
    argparser.add_argument('--max-epochs', type=int, default=5, help='Maximum number of training epochs')
    argparser.add_argument('--faster-h100', action='store_true', help='Use faster H100 optimizations')
    argparser.add_argument('--suppress-warnings', action='store_true', help='Suppress warnings during training')
    argparser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    argparser.add_argument('--run-name', type=str, default=None, help='WandB run name')
    argparser.add_argument('--prefetch-factor', type=int, default=16, help='Number of batches to prefetch per worker')
    argparser.add_argument('--skip-training', action='store_true', help='Skip training and only run evaluation')
    args = argparser.parse_args()

    if args.faster_h100:
        print("Using faster H100 optimizations")
        torch.set_float32_matmul_precision('high')

    if args.suppress_warnings:
        print("Suppressing warnings")
        warnings.filterwarnings("ignore")

    return args

def main():
    args = parse_args()

    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    aug_dir = args.aug_dir

    wandb_logger = WandbLogger(project="FDS", name=args.run_name)
    model = LitContrastive(lr=args.lr)

    augmentations = AudioAugmentations(
        enabled_augmentations=[
            BackgroundNoiseMixing(
                files_path=f"{aug_dir}/bg",
                train=True,
                snr_range=(0, 15),
                amp_range=(0.1, 1.4),
                sample_rate=8000
            ),
            ImpulseResponseAugmentation(
                ir_path=f"{aug_dir}/ir",
                train=True,
                sample_rate=8000
            ),
            PitchJitterAugmentation(
                steps_range=(-2, 2),
                sample_rate=8000,
                train=True
            ),
            VolumeAugmentation(
                gain_range=(-4.5, 4.5),
                scale_range=(0.5, 1.5),
                clipping=True,
                sample_rate=8000,
                train=True
            ),
            ReverbAugmentation(
                reverb_amount=(5, 30),
                room_scale=(30, 80),
                damping=(30, 70),
                wet_dry_mix=(20, 50),
                sample_rate=8000,
                train=True
            ),
            #BandPassFilterAugmentation(
            #    lower_range=(300, 500),
            #    upper_range=(4000, 6000),
            #    filter_order=4,
            #    sample_rate=8000,
            #    train=True,
            #),
        ]
    )

    print("Setting up data module...")
    dm = AudioDataModule(
        train_path=train_path, 
        val_path=val_path, 
        test_path=test_path, 
        num_workers=args.num_workers, 
        train_augmentations=augmentations, 
        batch_size=args.batch_size, 
        val_augmentations=augmentations,
        test_augmentations=augmentations,
        prefetch_factor=args.prefetch_factor,
        sample_duration_s=2,
        # hop_duration_s=1
    )
    
    # Configure checkpoint callback to save best model based on lowest loss
    run_name = args.run_name if args.run_name is not None else "default_run"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_pos_sim',
        dirpath='checkpoints',
        filename= run_name + '-best-model-{epoch:02d}-{train_loss:.4f}-{val_pos_sim:.4f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    lr_callback = LearningRateMonitor(logging_interval='step')
    
    # Monitor GPU usage metrics
    device_stats = DeviceStatsMonitor()
    
    print("Starting training...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs, 
        logger=wandb_logger, 
        callbacks=[checkpoint_callback, device_stats, lr_callback], 
        num_sanity_val_steps=0,
        timeout=600  # 10 minute timeout per batch
    ) # type: ignore
    dm.setup()
    
    if not args.skip_training:
        try:
            print("Fitting model (this may take a while)...")
            trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        model = LitContrastive.load_from_checkpoint(args.checkpoint)

    trainer.test(model=model, dataloaders=dm.test_dataloader())
    
    
if __name__ == "__main__":
    main()
