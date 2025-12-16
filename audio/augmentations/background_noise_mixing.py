from .augmentation_abc import AudioAugmentation
from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import Tensor
import numpy as np
from pathlib import Path
from os import path, listdir, walk
from random import choice, randint
import torchaudio

class BackgroundNoiseMixing(AudioAugmentation):
    """
    Background noise mixing augmentation for audio fingerprinting.
    
    Mixes clean audio with background noise at a random SNR (Signal-to-Noise Ratio).
    Inspired by the neural-audio-fp implementation.
    
    Args:
        noise_files: List of paths to background noise audio files
        snr_range: Tuple of (min_snr, max_snr) in decibels. Default: (0, 10)
        amp_range: Tuple of (min_amp, max_amp) for amplitude scaling. Default: (0.1, 1.0)
        sample_rate: Audio sample rate. Default: 8000
    """
    
    def __init__(
        self,
        files_path: str,
        snr_range: Tuple[float, float] = (0.0, 10.0),
        amp_range: Tuple[float, float] = (0.1, 1.0),
        train: bool = True,
        sample_rate: int = 8000
    ):
        
        files_dir = Path(files_path)
        assert path.exists(files_dir), f"Noise path {files_dir} does not exist."
        assert path.exists(files_dir / "tr") if train else path.exists(files_dir / "ts"), \
            f"Noise subdirectory {'tr' if train else 'ts'} does not exist in {files_dir}."

        files_dir = files_dir / ("tr" if train else "ts")
        
        # walk through directory and collect .wav files
        noise_files = []
        for (dirpath, dirnames, filenames) in walk(files_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    noise_files.append(Path(dirpath) / filename)

        assert len(noise_files) > 0, f"No .wav files found in {files_dir}."

        self.noise_files = noise_files
        self.snr_range = snr_range
        self.amp_range = amp_range
        self.sample_rate = sample_rate
        
        # Pre-load all noise files into memory for efficiency
        print(f"Pre-loading {len(noise_files)} background noise files...")
        self._noise_cache = self._preload_noise_files()
        memory_mb = sum(len(n) for n in self._noise_cache) * 4 / 1024 / 1024
        print(f"Loaded {len(self._noise_cache)} noise files (~{memory_mb:.1f} MB)")
    
    @property
    def name(self) -> str:
        return "BackgroundNoiseMixing"
    
    def _preload_noise_files(self) -> list[Tensor]:
        """
        Pre-load all noise files into memory.
        
        Returns:
            List of pre-loaded noise tensors
        """
        noise_cache = []
        for noise_file in self.noise_files:
            waveform, sr = torchaudio.load(noise_file)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            noise_cache.append(waveform)
        
        return noise_cache
    
    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply background noise mixing to an audio segment.
        
        Args:
            audio_segment: Input audio tensor of shape (samples,) or (batch, samples)
            
        Returns:
            Augmented audio tensor with same shape as input
        """
        # Handle batch dimension
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = audio_segment.shape[0]
        augmented_batch = []
        
        for i in range(batch_size):
            audio = audio_segment[i]
            
            # Load random background noise
            noise = self._load_random_noise(len(audio))
            
            # Sample random SNR
            snr_db = self._random_uniform(self.snr_range[0], self.snr_range[1])
            
            # Mix audio with noise
            mixed = self._mix_with_snr(audio, noise, snr_db)
            
            # Apply random amplitude scaling (log-scale)
            amp_factor = self._random_log_scale(self.amp_range[0], self.amp_range[1])
            mixed = mixed * amp_factor
            
            augmented_batch.append(mixed)
        
        result = torch.stack(augmented_batch)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def _load_random_noise(self, target_length: int) -> Tensor:
        """
        Load a random segment of background noise from pre-loaded cache.
        
        Args:
            target_length: Number of samples needed
            
        Returns:
            Noise segment tensor of shape (target_length,)
        """
        
        # Randomly select a pre-loaded noise file
        waveform = choice(self._noise_cache)
        
        # Random offset and extract target_length samples
        if len(waveform) > target_length:
            max_offset = len(waveform) - target_length
            offset = np.random.randint(0, max_offset + 1)
            noise = waveform[offset:offset + target_length]
        else:
            # Pad if noise is shorter than target
            noise = torch.nn.functional.pad(
                waveform, 
                (0, target_length - len(waveform))
            )
        
        return noise
    
    def _mix_with_snr(self, audio: Tensor, noise: Tensor, snr_db: float) -> Tensor:
        """
        Mix audio with noise at specified SNR.
        
        Args:
            audio: Clean audio signal
            noise: Noise signal
            snr_db: Desired SNR in decibels
            
        Returns:
            Mixed audio signal (max-normalized)
        """
        # Ensure same length
        min_len = min(len(audio), len(noise))
        audio = audio[:min_len]
        noise = noise[:min_len]
        
        # Normalize by RMSE (root mean square energy)
        audio_rmse = torch.sqrt(torch.mean(audio ** 2))
        noise_rmse = torch.sqrt(torch.mean(noise ** 2))
        
        # Avoid division by zero
        if audio_rmse == 0 or noise_rmse == 0:
            return self._max_normalize(audio + noise)
        
        audio_normalized = audio / audio_rmse
        noise_normalized = noise / noise_rmse
        
        # Calculate magnitude from SNR
        magnitude = 10 ** (snr_db / 20.0)
        
        # Mix: magnitude * audio + noise
        mixed = magnitude * audio_normalized + noise_normalized
        
        # Max-normalize to prevent clipping
        return self._max_normalize(mixed)
    
    def _max_normalize(self, audio: Tensor) -> Tensor:
        """Max-normalize audio to [-1, 1] range."""
        max_val = torch.max(torch.abs(audio))
        if max_val == 0:
            return audio
        return audio / max_val
    
    def _random_uniform(self, min_val: float, max_val: float) -> float:
        """Generate random value uniformly in [min_val, max_val]."""
        return np.random.uniform(min_val, max_val)
    
    def _random_log_scale(self, min_val: float, max_val: float) -> float:
        """
        Generate random value on log scale.
        
        This ensures uniform distribution in log space, giving equal probability
        to values across orders of magnitude.
        """
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_val = np.random.uniform(log_min, log_max)
        return 10 ** log_val
    

if __name__ == "__main__":
    # Load a sample audio file
    sample_audio_path = "dataset/neural-audio-fp-dataset/music/train-10k-30s/fma_small_8k_plus_medium_2k/000/000002.wav"
    waveform, sr = torchaudio.load(sample_audio_path)
    waveform = waveform.mean(dim=0)  # Convert to mono if stereo

    # Initialize augmentation
    noise_files_path = "dataset/neural-audio-fp-dataset/aug/bg"
    augmenter = BackgroundNoiseMixing(
        files_path=noise_files_path,
        snr_range=(0, 10),
        amp_range=(0.1, 1.0),
        train=True,
        sample_rate=sr
    )

    # Apply augmentation
    augmented_audio = augmenter.apply(waveform)

    # Save augmented audio for inspection
    torchaudio.save("augmented_audio.wav", augmented_audio.unsqueeze(0), sr)