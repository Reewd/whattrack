from model.model import LitContrastive, LitGenreClassifier
from model.dataset import AudioDataModule
from model.fma_dataset import FMADataModule
from audio.augmentation import AudioAugmentations
from audio.augmentations import PitchJitterAugmentation, TempoJitterAugmentation, BackgroundNoiseMixing, ImpulseResponseAugmentation
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
import argparse
import torch
import warnings


def make_train_trainer(args, checkpoint_callback, logger, device_stats, lr_callback):
    callbacks = [cb for cb in (checkpoint_callback, device_stats, lr_callback) if cb]
    return L.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks
    )


def make_eval_trainer(device_stats):
    return L.Trainer(logger=False, callbacks=[device_stats], enable_checkpointing=False)


def build_encoder_datamodule(args, augmentations):
    dm = AudioDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        num_workers=args.num_workers,
        train_augmentations=augmentations,
        batch_size=args.batch_size,
        val_augmentations=augmentations,
        prefetch_factor=args.prefetch_factor,
        sample_duration_s=args.sample_duration,
    )
    dm.setup()
    return dm


def build_classifier_datamodule(args):
    dm = FMADataModule(
        audio_path='fma_medium',
        metadata_path='fma_metadata',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_duration_s=args.sample_duration,
        sample_rate=8000,
        subset='medium',
        prefetch_factor=args.prefetch_factor
    )
    dm.setup()
    return dm


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training')
    argparser.add_argument('--mode', choices=['train-encoder', 'train-classifier', 'eval-encoder', 'eval-classifier'], default='train-encoder', help='What to run')
    argparser.add_argument('--train-path', type=str, default='dataset/music/train-10k-30s', help='Path to training dataset')
    argparser.add_argument('--val-path', type=str, default='dataset/music/val-query-db-500-30s', help='Path to validation dataset')
    argparser.add_argument('--test-path', type=str, default='dataset/music/test-query-db-500-30s/db', help='Path to test dataset')  
    argparser.add_argument('--aug-dir', type=str, default='dataset/aug', help='Path to augmentation directory')
    argparser.add_argument('--batch-size', type=int, default=300, help='Batch size for training')
    argparser.add_argument('--num-workers', type=int, default=32, help='Number of workers for data loading')
    argparser.add_argument('--max-epochs', type=int, default=5, help='Maximum number of training epochs')
    argparser.add_argument('--faster-h100', action='store_true', help='Use faster H100 optimizations')
    argparser.add_argument('--suppress-warnings', action='store_true', help='Suppress warnings during training')
    argparser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    argparser.add_argument('--run-name', type=str, default=None, help='WandB run name')
    argparser.add_argument('--prefetch-factor', type=int, default=4, help='Number of batches to prefetch per worker')
    argparser.add_argument('--encoder-checkpoint', type=str, default=None, help='Path to encoder checkpoint (for classifier training or encoder eval)')
    argparser.add_argument('--classifier-checkpoint', type=str, default=None, help='Path to classifier checkpoint (for classifier eval)')
    argparser.add_argument('--freeze-encoder', action='store_true', default=True, help='Freeze encoder during classifier training')
    argparser.add_argument('--classifier-lr', type=float, default=1e-3, help='Learning rate for classifier')
    argparser.add_argument('--sample-duration', type=float, default=1.0, help='Duration of audio samples in seconds')
    argparser.add_argument('--eval-test', action='store_true', help='Evaluate on the test split after training')
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

    wandb_logger = WandbLogger(project="FDS", name=args.run_name) if args.mode.startswith('train') else False
    print("Setting up data module...")

    # Shared callbacks
    lr_callback = LearningRateMonitor(logging_interval='step')
    device_stats = DeviceStatsMonitor()

    if args.mode == 'train-encoder':
        augmentations = AudioAugmentations(
            enabled_augmentations=[
                BackgroundNoiseMixing(
                    files_path=f"{args.aug_dir}/bg",
                    train=True,
                    snr_range=(0, 10),
                    sample_rate=8000
                ),
                ImpulseResponseAugmentation(
                    ir_path=f"{args.aug_dir}/ir",
                    train=True,
                    sample_rate=8000
                ),
            ]
        )

        dm = build_encoder_datamodule(args, augmentations)

        model = LitContrastive(lr=args.lr)
        print("Training encoder with contrastive learning...")

        run_name = args.run_name if args.run_name is not None else "contrastive"
        checkpoint_callback = ModelCheckpoint(
            monitor='val_pos_sim',
            dirpath='checkpoints',
            filename=run_name + '-{epoch:02d}-{train_loss:.4f}-{val_pos_sim:.4f}',
            save_top_k=1,
            mode='max',
            save_last=True
        )

        trainer = make_train_trainer(args, checkpoint_callback, wandb_logger, device_stats, lr_callback)
        trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader(), ckpt_path=args.resume_from_checkpoint)
        if args.eval_test:
            print("Running test evaluation...")
            trainer.test(model=model, dataloaders=dm.test_dataloader())

    elif args.mode == 'train-classifier':
        if not args.encoder_checkpoint:
            raise ValueError("--encoder-checkpoint is required for train-classifier")

        dm = build_classifier_datamodule(args)
        print(f"Number of genre classes: {dm.num_classes}")

        print(f"Loading pretrained encoder from {args.encoder_checkpoint}...")
        contrastive_model = LitContrastive.load_from_checkpoint(args.encoder_checkpoint)
        model = LitGenreClassifier(
            num_classes=dm.num_classes,
            pretrained_encoder=contrastive_model.encoder,
            lr=args.classifier_lr,
            freeze_encoder=args.freeze_encoder
        )
        print(f"Training classifier with {dm.num_classes} classes (encoder frozen: {args.freeze_encoder})")

        run_name = args.run_name if args.run_name is not None else "classifier"
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath='checkpoints',
            filename=run_name + '-{epoch:02d}-{val_acc:.4f}',
            save_top_k=1,
            mode='max',
            save_last=True
        )

        trainer = make_train_trainer(args, checkpoint_callback, wandb_logger, device_stats, lr_callback)
        trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader(), ckpt_path=args.resume_from_checkpoint)
        if args.eval_test:
            print("Running test evaluation...")
            trainer.test(model=model, dataloaders=dm.test_dataloader())

    elif args.mode == 'eval-encoder':
        if not args.encoder_checkpoint:
            raise ValueError("--encoder-checkpoint is required for eval-encoder")

        dm = build_encoder_datamodule(args, augmentations=None)

        print(f"Loading encoder from {args.encoder_checkpoint} for evaluation...")
        model = LitContrastive.load_from_checkpoint(args.encoder_checkpoint)

        trainer = make_eval_trainer(device_stats)
        print("Validating encoder...")
        trainer.validate(model=model, dataloaders=dm.val_dataloader())
        print("Testing encoder...")
        trainer.test(model=model, dataloaders=dm.test_dataloader())

    elif args.mode == 'eval-classifier':
        if not args.classifier_checkpoint:
            raise ValueError("--classifier-checkpoint is required for eval-classifier")

        dm = build_classifier_datamodule(args)

        print(f"Loading classifier from {args.classifier_checkpoint} for evaluation...")
        # Use a fresh encoder shell; weights are restored from checkpoint
        model = LitGenreClassifier.load_from_checkpoint(
            args.classifier_checkpoint,
            pretrained_encoder=LitContrastive().encoder
        )

        trainer = make_eval_trainer(device_stats)
        print("Validating classifier...")
        trainer.validate(model=model, dataloaders=dm.val_dataloader())
        print("Testing classifier...")
        trainer.test(model=model, dataloaders=dm.test_dataloader())
    
    
if __name__ == "__main__":
    main()
