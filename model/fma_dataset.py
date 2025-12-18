import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torchaudio
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np


class FMAGenreDataset(Dataset):
    """
    FMA dataset for genre classification.
    
    Returns (audio, label) pairs for supervised learning.
    Reads genre labels from FMA metadata tracks.csv.
    """
    
    def __init__(
        self,
        audio_path: str,
        metadata_path: str,
        sample_duration_s: float = 2.0,
        sample_rate: int = 8000,
        subset: str = 'small',
        split: str = 'training',
        genre_to_idx: Optional[Dict[str, int]] = None
    ):
        self.audio_path = Path(audio_path)
        self.metadata_path = Path(metadata_path)
        self.sample_rate = sample_rate
        self.sample_duration_s = sample_duration_s
        self.n_samples = int(sample_duration_s * sample_rate)
        self.subset = subset
        self.split = split
        self.genre_to_idx_override = genre_to_idx
        
        # Load metadata and create samples
        self._load_metadata()
        self.samples = self._create_samples_list()
        
        print(f"Loaded {len(self.samples)} samples from FMA {subset} {split} split")
        print(f"Number of genres: {self.num_classes}")
        print(f"Genres: {list(self.genre_to_idx.keys())}")
        print(f"Genre distribution: {self._get_genre_distribution()}")
    
    def _load_metadata(self):
        """Load FMA metadata CSV and extract genre information."""
        tracks_file = self.metadata_path / 'tracks.csv'
        
        # Load tracks metadata (multi-index CSV from FMA)
        tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
        
        # Filter by subset and split
        subset_mask = tracks['set', 'subset'] <= self.subset
        split_mask = tracks['set', 'split'] == self.split
        filtered_tracks = tracks[subset_mask & split_mask]
        
        # Extract top-level genre
        self.track_genres = filtered_tracks['track', 'genre_top'].copy()
        
        # Remove tracks without genre labels
        self.track_genres = self.track_genres.dropna()
        
        if self.genre_to_idx_override is None:
            # Create genre to index mapping from this split (train expected)
            unique_genres = sorted(self.track_genres.unique())
            self.genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
        else:
            # Reuse provided mapping and drop any genre not present
            self.genre_to_idx = self.genre_to_idx_override
            self.track_genres = self.track_genres[self.track_genres.isin(self.genre_to_idx.keys())]

        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}
        self.num_classes = len(self.genre_to_idx)
    
    def _create_samples_list(self) -> List[Tuple[Path, int]]:
        """Create list of (audio_path, genre_label) tuples."""
        samples = []
        
        for track_id, genre in self.track_genres.items():
            # FMA directory structure: audio_path/XXX/XXXXXX.mp3
            # where XXX is first 3 digits of 6-digit track_id
            track_id_str = f"{track_id:06d}"
            subdir = track_id_str[:3]
            audio_file = self.audio_path / subdir / f"{track_id_str}.mp3"
            
            if not audio_file.exists():
                # Try .wav extension as fallback
                audio_file = self.audio_path / subdir / f"{track_id_str}.wav"
            
            if audio_file.exists():
                # Skip samples whose genre is not in the provided mapping
                if genre in self.genre_to_idx:
                    genre_idx = self.genre_to_idx[genre]
                    samples.append((audio_file, genre_idx))
        
        return samples
    
    def _get_genre_distribution(self) -> Dict[str, int]:
        """Get distribution of genres in the dataset."""
        distribution = {}
        for _, label in self.samples:
            genre = self.idx_to_genre[label]
            distribution[genre] = distribution.get(genre, 0) + 1
        return distribution
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        Returns (audio, label) pair.
        
        Returns:
            audio: (n_samples,) - audio waveform
            label: int - genre class index
        """
        file_path, label = self.samples[idx]
        
        try:
            # Load audio file
            waveform, sr = torchaudio.load(file_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            audio = waveform.squeeze(0)
            
            # Random crop to sample_duration_s if longer (training)
            # Center crop for validation/test
            if audio.shape[0] > self.n_samples:
                if self.split == 'training':
                    # Random crop during training
                    max_start = audio.shape[0] - self.n_samples
                    start = np.random.randint(0, max_start)
                else:
                    # Center crop during validation/test
                    start = (audio.shape[0] - self.n_samples) // 2
                audio = audio[start:start + self.n_samples]
            elif audio.shape[0] < self.n_samples:
                # Pad if shorter
                padding = self.n_samples - audio.shape[0]
                audio = torch.nn.functional.pad(audio, (0, padding))
            
            return audio, label
        except Exception as e:
            # Return silence as fallback to avoid training interruption
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(self.n_samples), label


class FMADataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FMA genre classification.
    """
    
    def __init__(
        self,
        audio_path: str = 'fma_small',
        metadata_path: str = 'fma_metadata',
        batch_size: int = 128,
        num_workers: int = 4,
        sample_duration_s: float = 2.0,
        sample_rate: int = 8000,
        subset: str = 'small',
        prefetch_factor: int = 4
    ):
        super().__init__()
        self.audio_path = audio_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_duration_s = sample_duration_s
        self.sample_rate = sample_rate
        self.subset = subset
        self.prefetch_factor = prefetch_factor
        
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = FMAGenreDataset(
                audio_path=self.audio_path,
                metadata_path=self.metadata_path,
                sample_duration_s=self.sample_duration_s,
                sample_rate=self.sample_rate,
                subset=self.subset,
                split='training'
            )

            # Share a single label mapping across splits to keep class indices aligned
            shared_genre_to_idx = self.train_dataset.genre_to_idx
            
            self.val_dataset = FMAGenreDataset(
                audio_path=self.audio_path,
                metadata_path=self.metadata_path,
                sample_duration_s=self.sample_duration_s,
                sample_rate=self.sample_rate,
                subset=self.subset,
                split='validation',
                genre_to_idx=shared_genre_to_idx
            )
            
            # Store number of classes
            self.num_classes = self.train_dataset.num_classes
        
        if stage == "test" or stage is None:
            self.test_dataset = FMAGenreDataset(
                audio_path=self.audio_path,
                metadata_path=self.metadata_path,
                sample_duration_s=self.sample_duration_s,
                sample_rate=self.sample_rate,
                subset=self.subset,
                split='test',
                genre_to_idx=getattr(self, 'train_dataset', None).genre_to_idx if hasattr(self, 'train_dataset') else None
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                collate_fn=self._collate_fn
            )
        return None
    
    @staticmethod
    def _collate_fn(batch):
        """Collate function for classification datasets."""
        audios = []
        labels = []
        
        for audio, label in batch:
            audios.append(audio)
            labels.append(label)
        
        audios_batch = torch.stack(audios)  # (batch, n_samples)
        labels_batch = torch.tensor(labels, dtype=torch.long)  # (batch,)
        
        return audios_batch, labels_batch


if __name__ == "__main__":
    # Test the dataset
    print("Testing FMA Genre Dataset...")
    
    dataset = FMAGenreDataset(
        audio_path='fma_small',
        metadata_path='fma_metadata',
        sample_duration_s=2.0,
        sample_rate=8000,
        subset='small',
        split='training'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        audio, label = dataset[i]
        genre = dataset.idx_to_genre[label]
        print(f"\nSample {i}:")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Label: {label} ({genre})")
    
    # Test datamodule
    print("\n\nTesting FMA DataModule...")
    dm = FMADataModule(
        audio_path='fma_small',
        metadata_path='fma_metadata',
        batch_size=32,
        num_workers=4
    )
    
    dm.setup()
    train_loader = dm.train_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    
    # Test one batch
    batch_audio, batch_labels = next(iter(train_loader))
    print(f"Batch audio shape: {batch_audio.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
