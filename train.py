from model.dataset import AudioDataModule
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
import argparse

def train(args: argparse.Namespace, dm: AudioDataModule, model: L.LightningModule, wandb_logger: WandbLogger) -> None:
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
    trainer = L.Trainer(max_epochs=args.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, device_stats, lr_callback]) # type: ignore
    dm.setup()
    trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    