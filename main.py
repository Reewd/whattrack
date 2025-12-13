from model.model import LitContrastive
from model.dataset import AudioDataModule
import lightning as L
from pytorch_lightning.loggers import WandbLogger

def main():
    train_path = "dataset/music/train-10k-30s"
    val_path = "dataset/music/val-query-db-500-30s"
    test_path = "dataset/music/test-query-db-500-30s/db"
    wandb_logger = WandbLogger(project="FDS")
    model = LitContrastive()
    dm = AudioDataModule(train_path=train_path, val_path=val_path, test_path=test_path)
    trainer = L.Trainer(max_epochs=10, logger=wandb_logger)
    dm.setup()
    trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    
    
if __name__ == "__main__":
    main()
