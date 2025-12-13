from model.model import LitContrastive
from model.dataset import AudioDataModule
import lightning as L

def main():
    train_path = "dataset/music/train-10k-30s"
    val_path = "dataset/music/val-query-db-500-30s"
    test_path = "dataset/music/test-query-db-500-30s/db"
    model = LitContrastive()
    dm = AudioDataModule(train_path=train_path, val_path=val_path, test_path=test_path)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, dm)
    
    
if __name__ == "__main__":
    main()
