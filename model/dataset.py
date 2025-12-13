import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from os import walk
from abc import ABC, abstractmethod
from audio import AudioAugmentations
import torchaudio
from typing import Tuple
import numpy as np

class AbstractAudioDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, path: str, augPath: str | None = None):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method")


class AudioDataset(AbstractAudioDataset):
    """
    Audio fingerprinting dataset with anchor-positive pairs.
    
    Returns batches of (anchors, positives) where:
    - Anchors: Clean audio segments
    - Positives: Augmented versions from same song with slight time offsets
    """
    
    def __init__(
        self, 
        path: str, 
        augPath: str | None = None, 
        augmentations: AudioAugmentations | None = None, 
        sample_duration_s: float = 1.0,
        hop_duration_s: float = 0.5,
        sample_rate: int = 8000,
        n_positives_per_anchor: int = 1,
        offset_margin_ratio: float = 0.4,
        train: bool = True
    ):
        self.path = Path(path)
        self.augPath = Path(augPath) if augPath is not None else None
        self.sample_rate = sample_rate
        self.sample_duration_s = sample_duration_s
        self.hop_duration_s = hop_duration_s
        self.n_positives_per_anchor = n_positives_per_anchor
        self.offset_margin_ratio = offset_margin_ratio
        self.train = train
        self.augmentations = augmentations
        
        # Collect audio files
        self.audio_files = []
        for (dirpath, _, filenames) in walk(self.path):
            for filename in filenames:
                if filename.endswith('.wav'):
                    self.audio_files.append(Path(dirpath) / filename)
        
        assert len(self.audio_files) > 0, f"No .wav files found in {self.path}"
        
        # Create segment list: [[file_path, seg_idx, offset_min, offset_max], ...]
        self.segments = self._create_segment_list()
        
        print(f"Loaded {len(self.audio_files)} audio files with {len(self.segments)} segments")
    
    def _create_segment_list(self):
        """
        Create list of all possible segments from audio files.
        Similar to get_fns_seg_list() in neural-audio-fp.
        """
        segments = []
        n_samples_per_segment = int(self.sample_duration_s * self.sample_rate)
        n_samples_per_hop = int(self.hop_duration_s * self.sample_rate)
        
        for audio_file in self.audio_files:
            # Get audio length
            info = torchaudio.info(audio_file) # type: ignore
            n_frames = info.num_frames
            
            # Calculate number of segments
            if n_frames > n_samples_per_segment:
                n_segs = (n_frames - n_samples_per_segment + n_samples_per_hop) // n_samples_per_hop
            else:
                n_segs = 1
            
            n_segs = int(n_segs)
            residual_frames = max(0, n_frames - ((n_segs - 1) * n_samples_per_hop + n_samples_per_segment))
            
            # Create segment entries
            for seg_idx in range(n_segs):
                offset_min = -n_samples_per_hop if seg_idx > 0 else 0
                offset_max = n_samples_per_hop if seg_idx < n_segs - 1 else residual_frames
                
                segments.append([audio_file, seg_idx, offset_min, offset_max])
        
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (anchor, positives) pair.
        
        Returns:
            anchor: (1, n_samples) - clean audio segment
            positives: (n_positives, n_samples) - augmented versions with time offsets
        """
        file_path, seg_idx, offset_min, offset_max = self.segments[idx]
        
        # Calculate base time position
        base_start_sample = seg_idx * int(self.hop_duration_s * self.sample_rate)
        n_samples = int(self.sample_duration_s * self.sample_rate)
        
        # Calculate allowed offset range
        offset_margin = int(self.hop_duration_s * self.offset_margin_ratio * self.sample_rate)
        anchor_offset_min = max(offset_min, -offset_margin)
        anchor_offset_max = min(offset_max, offset_margin)
        
        # Load anchor with random offset (if training)
        if self.train and anchor_offset_min != anchor_offset_max:
            anchor_offset = np.random.randint(anchor_offset_min, anchor_offset_max + 1)
        else:
            anchor_offset = 0
        
        anchor_start = base_start_sample + anchor_offset
        anchor = self._load_audio_segment(file_path, anchor_start, n_samples)
        
        # Load positives with different random offsets
        positives = []
        if self.n_positives_per_anchor > 0:
            pos_offset_min = max((anchor_offset - offset_margin), offset_min)
            pos_offset_max = min((anchor_offset + offset_margin), offset_max)
            
            for _ in range(self.n_positives_per_anchor):
                if self.train and pos_offset_min != pos_offset_max:
                    pos_offset = np.random.randint(pos_offset_min, pos_offset_max + 1)
                else:
                    pos_offset = 0
                
                pos_start = base_start_sample + pos_offset
                pos_audio = self._load_audio_segment(file_path, pos_start, n_samples)
                
                # Apply augmentations to positive
                if self.augmentations is not None:
                    pos_audio = self.augmentations.apply(pos_audio)
                
                positives.append(pos_audio)
        
        # Stack positives
        if len(positives) > 0:
            positives_tensor = torch.stack(positives)  # (n_pos, n_samples)
        else:
            positives_tensor = torch.empty(0, n_samples)
        
        return anchor, positives_tensor
    
    def _load_audio_segment(self, file_path: Path, start_sample: int, n_samples: int) -> torch.Tensor:
        """Load a specific segment from audio file."""
        # Load audio
        waveform, sr = torchaudio.load(file_path)
        assert sr == self.sample_rate, f"Sample rate mismatch: expected {self.sample_rate}, got {sr}"
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Extract segment
        end_sample = start_sample + n_samples
        
        # Handle boundary conditions
        if start_sample < 0:
            start_sample = 0
        if end_sample > waveform.shape[1]:
            end_sample = waveform.shape[1]
        
        segment = waveform[:, start_sample:end_sample]
        
        # Pad if necessary
        if segment.shape[1] < n_samples:
            padding = n_samples - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, padding))
        
        return segment.squeeze(0)  # Return (n_samples,)


def collate_fn(batch):
    """
    Custom collate function to handle anchor-positive pairs.
    
    Input: List of (anchor, positives) tuples
    Output: (anchors_batch, positives_batch)
        anchors_batch: (batch_size, 1, n_samples)
        positives_batch: (batch_size * n_pos, 1, n_samples)
    """
    anchors = []
    positives = []
    
    for anchor, pos in batch:
        anchors.append(anchor)
        if len(pos) > 0:
            positives.extend(pos)
    
    # Stack and add channel dimension
    anchors_batch = torch.stack(anchors).unsqueeze(1)  # (batch_size, 1, n_samples)
    
    if len(positives) > 0:
        positives_batch = torch.stack(positives).unsqueeze(1)  # (batch_size*n_pos, 1, n_samples)
    else:
        positives_batch = torch.empty(0, 1, anchors_batch.shape[2])
    
    return anchors_batch, positives_batch


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_path: str, 
        val_path: str | None = None,
        test_path: str | None = None,
        batch_size: int = 60,
        num_workers: int = 4,
        train_aug_path: str | None = None,
        val_aug_path: str | None = None,
        sample_duration_s: float = 1.0,
        hop_duration_s: float = 0.5,
        sample_rate: int = 8000,
        n_positives_per_anchor: int = 1,
        train_augmentations: AudioAugmentations | None = None,
        val_augmentations: AudioAugmentations | None = None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_aug_path = train_aug_path
        self.val_aug_path = val_aug_path
        self.sample_duration_s = sample_duration_s
        self.hop_duration_s = hop_duration_s
        self.sample_rate = sample_rate
        self.n_positives_per_anchor = n_positives_per_anchor
        self.train_augmentations = train_augmentations
        self.val_augmentations = val_augmentations
        
    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = AudioDataset(
                path=self.train_path,
                augPath=self.train_aug_path,
                augmentations=self.train_augmentations,
                sample_duration_s=self.sample_duration_s,
                hop_duration_s=self.hop_duration_s,
                sample_rate=self.sample_rate,
                n_positives_per_anchor=self.n_positives_per_anchor,
                train=True
            )
            
            if self.val_path:
                self.val_dataset = AudioDataset(
                    path=self.val_path,
                    augPath=self.val_aug_path,
                    augmentations=self.val_augmentations,
                    sample_duration_s=self.sample_duration_s,
                    hop_duration_s=self.hop_duration_s,
                    sample_rate=self.sample_rate,
                    n_positives_per_anchor=self.n_positives_per_anchor,
                    train=False  # No random offsets for validation
                )
        
        if stage == "test" or stage is None:
            if self.test_path:
                self.test_dataset = AudioDataset(
                    path=self.test_path,
                    augmentations=None,  # No augmentations for test
                    sample_duration_s=self.sample_duration_s,
                    hop_duration_s=self.hop_duration_s,
                    sample_rate=self.sample_rate,
                    n_positives_per_anchor=0,  # No positives for test DB
                    train=False
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                collate_fn=collate_fn,
                pin_memory=True
            )
        return None
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                collate_fn=collate_fn,
                pin_memory=True
            )
        return None


if __name__ == "__main__":
    
    dataset_dir = "/home/francescodb/Documenti/Uni/FDS/Final Project/whattrack/dataset/neural-audio-fp-dataset/music/train-10k-30s/fma_small_8k_plus_medium_2k"
    aug_dir = "/home/francescodb/Documenti/Uni/FDS/Final Project/whattrack/dataset/neural-audio-fp-dataset/aug"
    from audio.augmentations import BackgroundNoiseMixing, ImpulseResponseAugmentation
    # Create augmentations
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
    
    # Create dataset
    dataset = AudioDataset(
        path=dataset_dir, 
        augPath=aug_dir,
        augmentations=augmentations,
        n_positives_per_anchor=1,
        train=True,
    )
    
    print(f"Number of segments: {len(dataset)}")
    print(f"\nTesting batch generation:")
    
    for i in range(3):
        anchor, positives = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Anchor shape: {anchor.shape}")
        print(f"  Positives shape: {positives.shape}")
    
    # Test dataloader
    print(f"\nTesting DataLoader:")
    dataloader = DataLoader(dataset, batch_size=60, shuffle=True, collate_fn=collate_fn, num_workers=12, persistent_workers=True, pin_memory=True)
    anchors_batch, positives_batch = next(iter(dataloader))
    print(f"Anchors batch shape: {anchors_batch.shape}")  # (60, 1, 8000)
    print(f"Positives batch shape: {positives_batch.shape}")  # (60, 1, 8000)