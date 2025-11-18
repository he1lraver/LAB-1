from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class IImageProcessing(ABC):
    """
    Интерфейс для реализации методов обработки изображений.
    Определяет набор методов для различных операций над изображениями.
    """
    
    @abstractmethod
    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Свёртка изображения с ядром."""
        pass
    
    @abstractmethod
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Преобразование RGB в grayscale."""
        pass
    
    @abstractmethod
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Обнаружение границ."""
        pass
    
    @abstractmethod
    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """Обнаружение углов."""
        pass
    
    @abstractmethod
    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """Обнаружение окружностей."""
        pass