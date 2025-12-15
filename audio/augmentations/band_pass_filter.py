from .augmentation_abc import AudioAugmentation
from torch import Tensor
import torch
import numpy as np
from typing import Tuple
import torchaudio


class BandPassFilterAugmentation(AudioAugmentation):
    """
    Band-pass filter augmentation for audio fingerprinting.
    
    Applies a band-pass filter that preserves frequencies within a specific band
    while attenuating frequencies outside that band. Filters out everything
    below the lower bound and everything above the upper bound.
    
    Simulates effects like:
    - Telephone/phone call quality (300-3400 Hz typical)
    - Radio transmission with limited bandwidth
    - Specific frequency range recordings
    - Old recording equipment with limited frequency response
    - Communication systems with bandwidth constraints
    
    Args:
        lower_range: Tuple of (min_freq_hz, max_freq_hz) for sampling the lower cutoff.
                    Default: (100, 300) Hz
                    The lower bound is sampled from this range.
        upper_range: Tuple of (min_freq_hz, max_freq_hz) for sampling the upper cutoff.
                    Default: (2000, 4000) Hz
                    The upper bound is sampled from this range.
        filter_order: Order of the Butterworth filter (steepness). Default: 4
                     Higher order = steeper rolloff but more computational cost
        sample_rate: Audio sample rate. Default: 8000
        train: Whether to enable randomness. Default: True
        
    Note:
        Ensure that lower_range values are always less than upper_range values
        to create a valid band-pass filter.
    """
    
    def __init__(
        self,
        lower_range: Tuple[float, float] = (300, 500),
        upper_range: Tuple[float, float] = (3000, 6000),
        filter_order: int = 4,
        sample_rate: int = 8000,
        train: bool = True,
    ) -> None:
        self.lower_range = lower_range
        self.upper_range = upper_range
        self.filter_order = filter_order
        self.sample_rate = sample_rate
        self.train = train
        
        # Validate frequency ranges
        assert lower_range[0] > 0, "Lower range minimum must be > 0 Hz"
        assert lower_range[1] < sample_rate / 2, \
            f"Lower range maximum must be < Nyquist frequency ({sample_rate / 2} Hz)"
        assert upper_range[0] > 0, "Upper range minimum must be > 0 Hz"
        assert upper_range[1] < sample_rate / 2, \
            f"Upper range maximum must be < Nyquist frequency ({sample_rate / 2} Hz)"
        assert lower_range[0] < lower_range[1], \
            "Lower range: minimum must be less than maximum"
        assert upper_range[0] < upper_range[1], \
            "Upper range: minimum must be less than maximum"
        assert lower_range[1] < upper_range[0], \
            "Lower range maximum must be less than upper range minimum for valid band-pass"
    
    @property
    def name(self) -> str:
        return "BandPassFilterAugmentation"
    
    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply band-pass filter augmentation to an audio segment.
        
        Supports input shapes (samples,) or (batch, samples).
        Returns the same shape as input.
        
        Args:
            audio_segment: Input audio tensor
            
        Returns:
            Augmented audio tensor with band-pass filter applied
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
            
            # Sample random cutoff frequencies if in training mode
            if self.train:
                lower_cutoff = float(np.random.uniform(
                    self.lower_range[0], 
                    self.lower_range[1]
                ))
                upper_cutoff = float(np.random.uniform(
                    self.upper_range[0], 
                    self.upper_range[1]
                ))
            else:
                # Use middle values when not training
                lower_cutoff = (self.lower_range[0] + self.lower_range[1]) / 2
                upper_cutoff = (self.upper_range[0] + self.upper_range[1]) / 2
            
            # Ensure lower < upper (should be guaranteed by validation, but double-check)
            if lower_cutoff >= upper_cutoff:
                lower_cutoff = upper_cutoff - 100  # Safety margin
            
            # Apply band-pass filter
            y = self._apply_bandpass_filter(x, lower_cutoff, upper_cutoff)
            
            # Ensure length == n_samples (pad or trim)
            if y.shape[0] < n_samples:
                y = torch.nn.functional.pad(y, (0, n_samples - y.shape[0]))
            elif y.shape[0] > n_samples:
                y = y[:n_samples]
            
            augmented.append(y)
        
        out = torch.stack(augmented)
        if squeeze_output:
            out = out.squeeze(0)
        return out
    
    def _apply_bandpass_filter(
        self, 
        audio: Tensor, 
        lower_cutoff: float, 
        upper_cutoff: float
    ) -> Tensor:
        """
        Apply band-pass filter to audio.
        
        Filters out everything below lower_cutoff and above upper_cutoff.
        
        Args:
            audio: Input audio tensor of shape (samples,)
            lower_cutoff: Lower cutoff frequency in Hz
            upper_cutoff: Upper cutoff frequency in Hz
            
        Returns:
            Filtered audio tensor
        """
        # Add channel dimension for torchaudio
        audio_ch = audio.unsqueeze(0)
        
        try:
            # Use torchaudio's bandpass_biquad filter
            filtered = torchaudio.functional.bandpass_biquad(
                audio_ch,
                self.sample_rate,
                (lower_cutoff + upper_cutoff) / 2,  # Central frequency
                (upper_cutoff - lower_cutoff) / ((lower_cutoff + upper_cutoff) / 2)  # Q factor
            )
            return filtered.squeeze(0)
            
        except Exception:
            # Fallback: use sox_effects if available
            try:
                # SoX bandpass filter: bandpass frequency bandwidth[q|o|h]
                central_freq = (lower_cutoff + upper_cutoff) / 2
                bandwidth = upper_cutoff - lower_cutoff
                
                filtered_ch, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio_ch,
                    self.sample_rate,
                    effects=[["bandpass", str(central_freq), str(bandwidth)]]
                )
                return filtered_ch.squeeze(0)
                
            except Exception:
                # Final fallback: apply high-pass then low-pass
                return self._simple_bandpass_fallback(audio, lower_cutoff, upper_cutoff)
    
    def _simple_bandpass_fallback(
        self,
        audio: Tensor,
        lower_cutoff: float,
        upper_cutoff: float
    ) -> Tensor:
        """
        Fallback band-pass filter using cascade of high-pass and low-pass.
        
        Args:
            audio: Input audio tensor
            lower_cutoff: Lower cutoff frequency
            upper_cutoff: Upper cutoff frequency
            
        Returns:
            Filtered audio tensor
        """
        # First apply high-pass to remove frequencies below lower_cutoff
        highpassed = self._simple_highpass(audio, lower_cutoff)
        
        # Then apply low-pass to remove frequencies above upper_cutoff
        bandpassed = self._simple_lowpass(highpassed, upper_cutoff)
        
        return bandpassed
    
    def _simple_highpass(self, audio: Tensor, cutoff_freq: float) -> Tensor:
        """Simple high-pass filter using subtraction method."""
        # Estimate kernel size from cutoff frequency
        normalized_cutoff = cutoff_freq / (self.sample_rate / 2)
        kernel_size = max(3, int(10 * (1 - normalized_cutoff)))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Create averaging kernel for low-pass
        kernel = torch.ones(1, 1, kernel_size, device=audio.device) / kernel_size
        
        # Reshape audio for conv1d
        audio_reshaped = audio.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution with padding
        padding = (kernel_size - 1) // 2
        lowpassed = torch.nn.functional.conv1d(
            audio_reshaped,
            kernel,
            padding=padding
        )
        
        # High-pass = Original - Low-pass
        highpassed = audio_reshaped - lowpassed
        
        return highpassed.squeeze(0).squeeze(0)
    
    def _simple_lowpass(self, audio: Tensor, cutoff_freq: float) -> Tensor:
        """Simple low-pass filter using moving average."""
        # Estimate kernel size from cutoff frequency
        normalized_cutoff = cutoff_freq / (self.sample_rate / 2)
        kernel_size = max(3, int(10 * (1 - normalized_cutoff)))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Create averaging kernel
        kernel = torch.ones(1, 1, kernel_size, device=audio.device) / kernel_size
        
        # Reshape audio for conv1d
        audio_reshaped = audio.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution with padding
        padding = (kernel_size - 1) // 2
        filtered = torch.nn.functional.conv1d(
            audio_reshaped,
            kernel,
            padding=padding
        )
        
        return filtered.squeeze(0).squeeze(0)
