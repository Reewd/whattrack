from .augmentation_abc import AudioAugmentation
from torch import Tensor
import torch
import numpy as np
from typing import Tuple


class VolumeAugmentation(AudioAugmentation):
    """
    Volume augmentation for audio fingerprinting.
    
    Applies random volume scaling to simulate different recording levels,
    microphone distances, or playback volumes while preserving audio content.
    
    Args:
        gain_range: Tuple of (min_gain_db, max_gain_db) in decibels.
                   Controls volume change. Default: (-12, 12) for ±12dB
                   Negative values reduce volume, positive values increase it.
        scale_range: Tuple of (min_scale, max_scale) as linear amplitude scale.
                    Alternative to gain_range if specified. Default: None
                    If provided, overrides gain_range.
        clipping: Whether to clip (prevent) audio values exceeding [-1, 1]. Default: True
        sample_rate: Audio sample rate. Default: 8000
        train: Whether to enable randomness. Default: True
    """
    
    def __init__(
        self,
        gain_range: Tuple[float, float] = (-12, 12),
        scale_range: Tuple[float, float] | None = None,
        clipping: bool = True,
        sample_rate: int = 8000,
        train: bool = True,
    ) -> None:
        self.gain_range = gain_range
        self.scale_range = scale_range
        self.clipping = clipping
        self.sample_rate = sample_rate
        self.train = train
    
    @property
    def name(self) -> str:
        return "VolumeAugmentation"
    
    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply volume augmentation to an audio segment.
        
        Supports input shapes (samples,) or (batch, samples).
        Returns the same shape as input.
        
        Args:
            audio_segment: Input audio tensor
            
        Returns:
            Augmented audio tensor with volume scaling applied
        """
        # Normalize input shape to (batch, samples)
        squeeze_output = False
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)
            squeeze_output = True
        
        batch, n_samples = audio_segment.shape
        augmented = []
        
        for i in range(batch):
            x = audio_segment[i]
            
            # Sample random volume parameter if in training mode
            if self.train:
                scale_factor = self._get_random_scale()
            else:
                # Use middle value when not training
                scale_factor = self._get_fixed_scale()
            
            # Apply volume scaling
            y = x * scale_factor
            
            # Optionally clip to prevent distortion
            if self.clipping:
                y = torch.clamp(y, min=-1.0, max=1.0)
            
            augmented.append(y)
        
        out = torch.stack(augmented)
        if squeeze_output:
            out = out.squeeze(0)
        return out
    
    def _get_random_scale(self) -> float:
        """
        Generate a random scale factor for volume adjustment.
        
        Returns:
            Linear amplitude scale factor
        """
        if self.scale_range is not None:
            # Use linear scale range
            scale = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        else:
            # Convert dB range to linear scale
            # Formula: linear_scale = 10^(gain_db / 20)
            gain_db = float(np.random.uniform(self.gain_range[0], self.gain_range[1]))
            scale = 10.0 ** (gain_db / 20.0)
        
        return max(0.01, scale)  # Prevent zero or negative scales
    
    def _get_fixed_scale(self) -> float:
        """
        Get a fixed scale factor (middle value) for deterministic inference.
        
        Returns:
            Linear amplitude scale factor
        """
        if self.scale_range is not None:
            scale = (self.scale_range[0] + self.scale_range[1]) / 2
        else:
            # Use middle value of dB range
            gain_db = (self.gain_range[0] + self.gain_range[1]) / 2
            scale = 10.0 ** (gain_db / 20.0)
        
        return max(0.01, scale)
