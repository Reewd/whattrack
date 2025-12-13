from typing import Sequence
from .augmentations.augmentation_abc import AudioAugmentation
from torch import Tensor

class AudioAugmentations:
    def __init__(self, enabled_augmentations: Sequence[AudioAugmentation]) -> None:
        self.enabled_augmentations = enabled_augmentations

    def apply(self, audio_segment: Tensor) -> Tensor:
        """
        Apply all enabled augmentations to the audio segment.
        
        Args:
            audio_segment: Input audio segment tensor
        
        Returns:
            Augmented audio segment tensor
        """
        augmented_segment = audio_segment.clone()
        for augmentation in self.enabled_augmentations:
            augmented_segment = augmentation.apply(augmented_segment)
        return augmented_segment