from .augmentation_abc import AudioAugmentation
from torch import Tensor
import torch
import numpy as np
import torchaudio
from typing import Tuple


class ReverbAugmentation(AudioAugmentation):
    """
    Reverb augmentation for audio fingerprinting.
    
    Applies reverberation effects to simulate different acoustic environments
    (rooms, halls, etc.) without requiring external impulse response files.
    Uses torchaudio's built-in reverb effects via SoX.
    
    Args:
        reverb_amount: Tuple of (min_amount, max_amount) in percentage (0-100).
                      Controls the amount of reverb/wet signal. Default: (30, 60)
        room_scale: Tuple of (min_scale, max_scale) in percentage (0-100).
                   Simulates room size. Default: (30, 80)
        damping: Tuple of (min_damp, max_damp) in percentage (0-100).
                Controls high-frequency damping. Default: (30, 70)
        wet_dry_mix: Tuple of (min_wet, max_wet) in percentage (0-100).
                    Ratio of wet (reverbed) to dry signal. Default: (20, 50)
        sample_rate: Audio sample rate. Default: 8000
        train: Whether to enable randomness. Default: True
    """
    
    def __init__(
        self,
        reverb_amount: Tuple[float, float] = (30, 60),
        room_scale: Tuple[float, float] = (30, 80),
        damping: Tuple[float, float] = (30, 70),
        wet_dry_mix: Tuple[float, float] = (20, 50),
        sample_rate: int = 8000,
        train: bool = True,
    ) -> None:
        self.reverb_amount = reverb_amount
        self.room_scale = room_scale
        self.damping = damping
        self.wet_dry_mix = wet_dry_mix
        self.sample_rate = sample_rate
        self.train = train
    
    @property
    def name(self) -> str:
        return "ReverbAugmentation"
    
    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply reverb augmentation to an audio segment.
        
        Supports input shapes (samples,) or (batch, samples).
        Returns the same shape as input.
        
        Args:
            audio_segment: Input audio tensor
            
        Returns:
            Augmented audio tensor with reverb applied
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
            
            # Sample random reverb parameters if in training mode
            if self.train:
                reverb_amount = float(np.random.uniform(self.reverb_amount[0], self.reverb_amount[1]))
                room_scale = float(np.random.uniform(self.room_scale[0], self.room_scale[1]))
                damping = float(np.random.uniform(self.damping[0], self.damping[1]))
                wet_dry_mix = float(np.random.uniform(self.wet_dry_mix[0], self.wet_dry_mix[1]))
            else:
                # Use middle values when not training
                reverb_amount = (self.reverb_amount[0] + self.reverb_amount[1]) / 2
                room_scale = (self.room_scale[0] + self.room_scale[1]) / 2
                damping = (self.damping[0] + self.damping[1]) / 2
                wet_dry_mix = (self.wet_dry_mix[0] + self.wet_dry_mix[1]) / 2
            
            # Apply reverb using SoX reverb effect
            y = self._apply_reverb(x, reverb_amount, room_scale, damping, wet_dry_mix)
            
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
    
    def _apply_reverb(
        self,
        audio: Tensor,
        reverb_amount: float,
        room_scale: float,
        damping: float,
        wet_dry_mix: float,
    ) -> Tensor:
        """
        Apply reverb effect using SoX or fallback method.
        
        Args:
            audio: Input audio tensor of shape (samples,)
            reverb_amount: Reverb amount in percentage (0-100)
            room_scale: Room size in percentage (0-100)
            damping: Damping factor in percentage (0-100)
            wet_dry_mix: Wet/dry mix in percentage (0-100)
            
        Returns:
            Reverbed audio tensor
        """
        # Add channel dimension for sox_effects
        audio_ch = audio.unsqueeze(0)
        
        try:
            # Use SoX reverb effect
            # reverb effect syntax: reverb [reverberance [HF-damping [room-scale [stereo-depth [pre-delay [wet-gain [dry-gain [wet-only]]]]]]]
            # Map parameters to reasonable ranges for reverb effect
            reverberance = reverb_amount  # 0-100
            hf_damping = damping  # 0-100
            room_scale_param = room_scale / 100.0 * 100  # Convert to 0-100 range
            
            effects = [
                ["reverb", str(reverberance), str(hf_damping), str(room_scale_param)]
            ]
            
            audio_reverbed, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio_ch, self.sample_rate, effects=effects
            )
            
            # Mix wet and dry signals based on wet_dry_mix parameter
            wet_gain = wet_dry_mix / 100.0
            dry_gain = (100.0 - wet_dry_mix) / 100.0
            
            result = dry_gain * audio + wet_gain * audio_reverbed.squeeze(0)
            return result
            
        except Exception:
            # Fallback: apply simple algorithmic reverb using delay-based approach
            return self._simple_reverb_fallback(
                audio, reverb_amount, room_scale, wet_dry_mix
            )
    
    def _simple_reverb_fallback(
        self,
        audio: Tensor,
        reverb_amount: float,
        room_scale: float,
        wet_dry_mix: float,
    ) -> Tensor:
        """
        Fallback reverb implementation using delays and feedback.
        Creates a simple reverb-like effect by layering delayed copies.
        
        Args:
            audio: Input audio tensor
            reverb_amount: Reverb intensity (0-100)
            room_scale: Room size factor (affects delay times)
            wet_dry_mix: Wet/dry signal mix
            
        Returns:
            Audio with reverb-like effect applied
        """
        device = audio.device
        n_samples = audio.shape[0]
        
        # Calculate reverb parameters based on room scale and reverb amount
        # Larger room = longer decay times
        decay_factor = min(0.8, reverb_amount / 100.0)
        
        # Base delay time proportional to room scale (in milliseconds)
        base_delay_ms = 10 + (room_scale / 100.0) * 50  # 10-60 ms
        base_delay_samples = int(base_delay_ms * self.sample_rate / 1000.0)
        
        # Create multiple delayed copies with decreasing amplitude
        reverb = torch.zeros_like(audio)
        
        # Add 3-4 delayed copies
        num_delays = int(2 + (reverb_amount / 100.0) * 2)
        for delay_idx in range(1, num_delays + 1):
            delay_samples = base_delay_samples * delay_idx
            if delay_samples >= n_samples:
                break
            
            # Amplitude decreases with each delay
            amp = (decay_factor ** delay_idx)
            
            # Create delayed signal
            delayed = torch.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * amp
            
            reverb = reverb + delayed
        
        # Normalize and mix
        reverb_max = torch.max(torch.abs(reverb))
        if reverb_max > 0:
            reverb = reverb / (reverb_max + 1e-8)
        
        # Mix wet and dry signals
        wet_gain = wet_dry_mix / 100.0
        dry_gain = (100.0 - wet_dry_mix) / 100.0
        
        result = dry_gain * audio + wet_gain * reverb
        
        # Prevent clipping
        max_val = torch.max(torch.abs(result))
        if max_val > 1.0:
            result = result / max_val
        
        return result
