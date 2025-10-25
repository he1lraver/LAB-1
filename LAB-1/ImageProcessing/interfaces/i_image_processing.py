from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class IImageProcessing(ABC):
    """
    Интерфейс для реализации методов обработки изображений.
    """

    @abstractmethod
    def convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        pass
