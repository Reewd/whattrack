from abc import ABC, abstractmethod
from torch import Tensor

class AudioAugmentation(ABC):
    @abstractmethod
    def apply(self, audio_segment: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


