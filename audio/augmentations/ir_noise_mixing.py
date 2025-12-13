from abc import ABC, abstractmethod
from pathlib import Path
from os import walk
from typing import Optional
import torch
from torch import Tensor
import numpy as np
from random import choice
import torchaudio
from .augmentation_abc import AudioAugmentation


class ImpulseResponseAugmentation(AudioAugmentation):
    """
    Impulse Response (IR) augmentation for simulating room acoustics.
    
    Applies convolution with room impulse responses to simulate reverberation
    and room characteristics. This mimics how audio sounds in different environments
    (concert halls, rooms, studios, etc.).
    
    Args:
        ir_path: Path to directory containing impulse response files
                 Expected structure: ir_path/tr/ or ir_path/ts/
        train: If True, use 'tr' subdirectory, else use 'ts' subdirectory
        max_ir_length: Maximum IR length in samples to use (for efficiency). Default: 600
        sample_rate: Audio sample rate. Default: 8000
    """
    
    def __init__(
        self,
        ir_path: str | Path,
        train: bool = True,
        max_ir_length: int = 600,  # 75ms at 8kHz
        sample_rate: int = 8000
    ):
        self.ir_path = Path(ir_path)
        self.train = train
        self.max_ir_length = max_ir_length
        self.sample_rate = sample_rate
        
        # Load IR file paths
        self.ir_files = self._load_ir_file_paths()
        
        # Cache for loaded IRs (optional optimization)
        self._ir_cache = {}
    
    def _load_ir_file_paths(self) -> list[Path]:
        """
        Automatically parse IR directory structure and collect .wav files.
        
        Returns:
            List of paths to IR files
        """
        assert self.ir_path.exists(), \
            f"IR path {self.ir_path} does not exist."
        
        subdir = "tr" if self.train else "ts"
        assert (self.ir_path / subdir).exists(), \
            f"IR subdirectory '{subdir}' does not exist in {self.ir_path}."
        
        ir_dir = self.ir_path / subdir
        
        # Walk through directory and collect .wav files
        ir_files = []
        for (dirpath, dirnames, filenames) in walk(ir_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    ir_files.append(Path(dirpath) / filename)
        
        assert len(ir_files) > 0, \
            f"No .wav files found in {ir_dir}."
        
        print(f"Loaded {len(ir_files)} IR files from {ir_dir}")
        return ir_files
    
    @property
    def name(self) -> str:
        return "ImpulseResponseAugmentation"
    
    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply impulse response convolution to audio segment.
        
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
            
            # Load random IR
            ir = self._load_random_ir()
            
            # Apply IR via FFT convolution
            augmented = self._convolve_fft(audio, ir)
            
            augmented_batch.append(augmented)
        
        result = torch.stack(augmented_batch)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def _load_random_ir(self) -> Tensor:
        """
        Load a random impulse response.
        
        Returns:
            IR tensor of shape (ir_length,), truncated to max_ir_length
        """
        
        # Randomly select an IR file
        ir_file = choice(self.ir_files)
        
        # Check cache first
        ir_file_str = str(ir_file)
        if ir_file_str in self._ir_cache:
            return self._ir_cache[ir_file_str]
        
        # Load audio
        waveform, sr = torchaudio.load(ir_file)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Truncate to max_ir_length for efficiency
        if len(waveform) > self.max_ir_length:
            waveform = waveform[:self.max_ir_length]
        
        # Cache the IR (optional)
        self._ir_cache[ir_file_str] = waveform
        
        return waveform
    
    def _convolve_fft(self, audio: Tensor, ir: Tensor) -> Tensor:
        """
        Convolve audio with impulse response using FFT.
        
        This is more efficient than direct convolution for longer IRs.
        Matches the implementation from ir_aug_batch() in neural-audio-fp.
        
        Args:
            audio: Audio signal tensor
            ir: Impulse response tensor
            
        Returns:
            Convolved audio (max-normalized)
        """
        # Determine FFT length (max of audio and IR lengths)
        fft_length = max(len(audio), len(ir))
        
        # FFT of both signals
        audio_fft = torch.fft.fft(audio, n=fft_length)
        ir_fft = torch.fft.fft(ir, n=fft_length)
        
        # Multiply in frequency domain (equivalent to convolution in time domain)
        convolved_fft = audio_fft * ir_fft
        
        # IFFT back to time domain
        convolved = torch.fft.ifft(convolved_fft)
        
        # Take real part and truncate to original audio length
        convolved = convolved.real[:len(audio)]
        
        # Max-normalize
        return self._max_normalize(convolved)
    
    def _max_normalize(self, audio: Tensor) -> Tensor:
        """Max-normalize audio to [-1, 1] range."""
        max_val = torch.max(torch.abs(audio))
        if max_val == 0:
            return audio
        return audio / max_val


if __name__ == "__main__":
    
    # Load test audio file
    test_file = "000002.wav"
    audio_segment, sr = torchaudio.load(test_file)
    
    # Convert to mono if stereo
    if audio_segment.shape[0] > 1:
        audio_segment = audio_segment.mean(dim=0)
    else:
        audio_segment = audio_segment.squeeze(0)
    
    # Resample to 8kHz if necessary
    if sr != 8000:
        resampler = torchaudio.transforms.Resample(sr, 8000)
        audio_segment = resampler(audio_segment)
    
    # Truncate or pad to 1 second (8000 samples)
    #if len(audio_segment) > 8000:
    #    audio_segment = audio_segment[:8000]
    #else:
    #    audio_segment = torch.nn.functional.pad(
    #        audio_segment, 
    #        (0, 8000 - len(audio_segment))
    #    )
    
    print(f"Loaded audio from {test_file}")
    print(f"Audio shape: {audio_segment.shape}")
    print(f"Sample rate: 8000 Hz")
    
    # Create training IR augmentation instance
    ir_augmentation = ImpulseResponseAugmentation(
        ir_path="dataset/neural-audio-fp-dataset/aug/ir",
        train=True,  # Uses 'tr' subdirectory
        max_ir_length=600,
        sample_rate=8000
    )
    
    # Apply IR augmentation
    augmented = ir_augmentation.apply(audio_segment)
    
    print(f"\nAugmentation applied: {ir_augmentation.name}")
    print(f"Original shape: {audio_segment.shape}")
    print(f"Augmented shape: {augmented.shape}")
    
    # Optionally save the augmented audio
    torchaudio.save("000002_augmented.wav", augmented.unsqueeze(0), 8000)
    print(f"\nAugmented audio saved to: 000002_augmented.wav")