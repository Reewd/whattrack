from .augmentation_abc import AudioAugmentation
from torch import Tensor
import torch
import numpy as np
import torchaudio


class PitchJitterAugmentation(AudioAugmentation):
	"""
	Random pitch jitter augmentation.

	Applies a small random pitch shift to the audio segment measured in semitone steps.
	This simulates minor detuning or playback speed variations while preserving duration.

	Args:
		steps_range: Tuple (min_steps, max_steps) in semitones. e.g., (-0.5, 0.5)
		sample_rate: Audio sample rate. Default: 8000
		train: Whether to enable randomness (kept for API symmetry)
	"""

	def __init__(
		self,
		steps_range: tuple[float, float] = (-0.5, 0.5),
		sample_rate: int = 8000,
		train: bool = True,
	) -> None:
		self.steps_range = steps_range
		self.sample_rate = sample_rate
		self.train = train

	@property
	def name(self) -> str:
		return "PitchJitterAugmentation"

	def apply(self, audio_segment: Tensor) -> Tensor:
		"""
		Apply random pitch jitter to an audio segment.

		Supports input shapes (samples,) or (batch, samples).
		Returns the same shape as input.
		"""
		# Normalize input shape to (batch, samples)
		squeeze_output = False
		if audio_segment.dim() == 1:
			audio_segment = audio_segment.unsqueeze(0)
			squeeze_output = True

		batch, n_samples = audio_segment.shape
		augmented = []

		# Random steps per item
		for i in range(batch):
			x = audio_segment[i]
			steps = float(np.random.uniform(self.steps_range[0], self.steps_range[1])) if self.train else 0.0

			if steps == 0.0:
				y = x
			else:
				# torchaudio.functional.pitch_shift preserves length
				# Input should be (channels, time); use mono channel
				x_ch = x.unsqueeze(0)  # (1, time)
				try:
					y = torchaudio.functional.pitch_shift(x_ch, self.sample_rate, int(steps)).squeeze(0)
				except Exception:
					# Fallback: simple resample-based shift (changes formant slightly)
					# Compute rate factor from semitone steps: factor = 2^(steps/12)
					factor = 2 ** (steps / 12.0)
					new_sr = int(self.sample_rate * factor)
					resample_to = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=new_sr)
					resample_back = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=self.sample_rate)
					y = resample_back(resample_to(x))

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
