from .augmentation_abc import AudioAugmentation
from torch import Tensor
import torch
import numpy as np
import torchaudio


class TempoJitterAugmentation(AudioAugmentation):
    """
    Random tempo jitter augmentation.

    Applies a small random tempo change (speed up or slow down)
    using SoX tempo effect. Pitch is preserved by SoX's tempo algorithm.

    Args:
        factor_range: (min_factor, max_factor), where 1.0 means no change.
                      e.g., (0.95, 1.05) for ±5% tempo change.
        sample_rate: Audio sample rate. Default: 8000
        train: Whether to enable randomness
    """

    def __init__(
        self,
        factor_range: tuple[float, float] = (0.95, 1.05),
        sample_rate: int = 8000,
        train: bool = True,
    ) -> None:
        self.factor_range = factor_range
        self.sample_rate = sample_rate
        self.train = train

    @property
    def name(self) -> str:
        return "TempoJitterAugmentation"

    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply random tempo jitter.

        Accepts (samples,) or (batch, samples). Returns same shape.
        """
        squeeze_output = False
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)
            squeeze_output = True

        batch, n_samples = audio_segment.shape
        augmented = []

        for i in range(batch):
            x = audio_segment[i]
            factor = float(np.random.uniform(self.factor_range[0], self.factor_range[1])) if self.train else 1.0

            if factor == 1.0:
                y = x
            else:
                # SoX tempo effect: preserves pitch, changes duration.
                # Input to sox_effects is (channels, time)
                x_ch = x.unsqueeze(0)
                try:
                    y_ch, sr = torchaudio.sox_effects.apply_effects_tensor(
                        x_ch, self.sample_rate,
                        effects=[["tempo", str(factor)]]
                    )
                    y = y_ch.squeeze(0)
                except Exception:
                    # Fallback: simple resample-based speed change (will alter pitch slightly)
                    new_sr = int(self.sample_rate * factor)
                    resample_to = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=new_sr)
                    y = resample_to(x)

                # After tempo change, duration differs; resample back to original length
                if y.shape[0] != n_samples:
                    y = self._fit_length(y, n_samples)

            augmented.append(y)

        out = torch.stack(augmented)
        if squeeze_output:
            out = out.squeeze(0)
        return out

    def _fit_length(self, y: Tensor, target_len: int) -> Tensor:
        """Resample y uniformly to target_len samples."""
        # Create time indices and interpolate to target length
        src_len = y.shape[0]
        if src_len == target_len:
            return y
        # Use linear interpolation via grid_sample on 1D treated as 2D
        # Simpler: use torchaudio resample with fractional rate
        rate = src_len / float(target_len)
        new_sr = max(1, int(self.sample_rate / rate))
        resampler = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=new_sr)  # placeholder to satisfy type
        # Instead, do manual interpolation
        idx = torch.linspace(0, src_len - 1, target_len, device=y.device)
        idx_floor = torch.floor(idx).long()
        idx_ceil = torch.clamp(idx_floor + 1, max=src_len - 1)
        w = idx - idx_floor.float()
        return (1 - w) * y[idx_floor] + w * y[idx_ceil]