from .augmentation_abc import AudioAugmentation
from torch import Tensor
import torch
import numpy as np
from typing import Tuple
import torchaudio


class LowPassFilterAugmentation(AudioAugmentation):
    """
    Low-pass filter augmentation for audio fingerprinting.
    
    Applies a low-pass filter that attenuates high frequencies while preserving
    low frequencies. Simulates effects like:
    - Poor quality or low-bandwidth audio recordings
    - Distance/propagation effects
    - Muffled/distant sound
    - Telephone or speech codec limitations
    - Microphone limitations
    
    Args:
        cutoff_freq_range: Tuple of (min_freq_hz, max_freq_hz) for filter cutoff frequency.
                          Default: (1000, 4000) Hz for attenuating high frequencies
                          Typical speech range is 80-8000 Hz, so cutoff in 1000-4000 Hz
                          range removes high-frequency detail.
        filter_order: Order of the Butterworth filter (steepness). Default: 4
                     Higher order = steeper rolloff but more computational cost
        sample_rate: Audio sample rate. Default: 8000
        train: Whether to enable randomness. Default: True
    """
    
    def __init__(
        self,
        cutoff_freq_range: Tuple[float, float] = (1000, 4000),
        filter_order: int = 4,
        sample_rate: int = 8000,
        train: bool = True,
        p: float = 0.5
    ) -> None:
        self.cutoff_freq_range = cutoff_freq_range
        self.filter_order = filter_order
        self.sample_rate = sample_rate
        self.train = train
        self.p = p
        
        # Validate cutoff frequencies
        assert cutoff_freq_range[0] > 0, "Minimum cutoff frequency must be > 0 Hz"
        assert cutoff_freq_range[1] < sample_rate / 2, \
            f"Maximum cutoff frequency must be < Nyquist frequency ({sample_rate / 2} Hz)"
        assert cutoff_freq_range[0] < cutoff_freq_range[1], \
            "Minimum cutoff must be less than maximum cutoff"
    
    @property
    def name(self) -> str:
        return "LowPassFilterAugmentation"
    
    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply low-pass filter augmentation to an audio segment.
        
        Supports input shapes (samples,) or (batch, samples).
        Returns the same shape as input.
        
        Args:
            audio_segment: Input audio tensor
            
        Returns:
            Augmented audio tensor with low-pass filter applied
        """
        # Normalize input shape to (batch, samples)
        if torch.rand(1).item() > self.p:
            return audio_segment

        squeeze_output = False
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)
            squeeze_output = True
        
        batch, n_samples = audio_segment.shape
        augmented = []
        
        for i in range(batch):
            x = audio_segment[i]
            
            # Sample random cutoff frequency if in training mode
            if self.train:
                cutoff_freq = float(np.random.uniform(
                    self.cutoff_freq_range[0], 
                    self.cutoff_freq_range[1]
                ))
            else:
                # Use middle value when not training
                cutoff_freq = (self.cutoff_freq_range[0] + self.cutoff_freq_range[1]) / 2
            
            # Apply low-pass filter
            y = self._apply_lowpass_filter(x, cutoff_freq)
            
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
    
    def _apply_lowpass_filter(self, audio: Tensor, cutoff_freq: float) -> Tensor:
        """
        Apply Butterworth low-pass filter to audio.
        
        Args:
            audio: Input audio tensor of shape (samples,)
            cutoff_freq: Cutoff frequency in Hz
            
        Returns:
            Filtered audio tensor
        """
        # Add channel dimension for torchaudio
        audio_ch = audio.unsqueeze(0)
        
        try:
            # Use torchaudio's lowpass_biquad filter (simpler) or sox_effects
            # Using lowpass_biquad is lightweight and effective
            filtered = torchaudio.functional.lowpass_biquad(
                audio_ch,
                self.sample_rate,
                cutoff_freq
            )
            return filtered.squeeze(0)
            
        except Exception:
            # Fallback: use sox_effects if available
            try:
                # SoX lowpass filter: lowpass frequency [width[q|o|h]]
                # Normalize cutoff frequency to be a valid value
                normalized_cutoff = min(cutoff_freq, self.sample_rate / 2 - 1)
                
                filtered_ch, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio_ch,
                    self.sample_rate,
                    effects=[["lowpass", str(normalized_cutoff)]]
                )
                return filtered_ch.squeeze(0)
                
            except Exception:
                # Final fallback: simple moving average filter
                return self._simple_lowpass_fallback(audio, cutoff_freq)
    
    def _simple_lowpass_fallback(self, audio: Tensor, cutoff_freq: float) -> Tensor:
        """
        Fallback low-pass filter using moving average.
        Creates a simple low-pass effect by averaging neighboring samples.
        
        Args:
            audio: Input audio tensor
            cutoff_freq: Cutoff frequency (used to determine filter kernel size)
            
        Returns:
            Filtered audio tensor
        """
        # Estimate kernel size from cutoff frequency
        # Lower cutoff = more aggressive filtering = larger kernel
        normalized_cutoff = cutoff_freq / (self.sample_rate / 2)
        # Invert relationship: lower normalized cutoff -> higher kernel size
        kernel_size = max(3, int(10 * (1 - normalized_cutoff)))
        # Make kernel size odd for symmetric filtering
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Create simple averaging kernel (moving average)
        kernel = torch.ones(1, 1, kernel_size, device=audio.device) / kernel_size
        
        # Reshape audio for conv1d: (1, 1, samples)
        audio_reshaped = audio.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution with padding to maintain size
        padding = (kernel_size - 1) // 2
        filtered = torch.nn.functional.conv1d(
            audio_reshaped,
            kernel,
            padding=padding
        )
        
        # Reshape back to (samples,)
        return filtered.squeeze(0).squeeze(0)
