"""
Абстрактный класс CatImage и его реализации для цветных и ч/б изображений.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Dict, Any
import cv2
from PIL import Image


class CatImage(ABC):
    """
    Абстрактный класс для представления изображения животного.
    """
    
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        self._image_data = image_data
        self._image_url = image_url
        self._breed = breed
    
    @property
    def image_data(self) -> np.ndarray:
        return self._image_data
    
    @property
    def image_url(self) -> str:
        return self._image_url
    
    @property
    def breed(self) -> str:
        return self._breed
    
    @property
    @abstractmethod
    def is_grayscale(self) -> bool:
        pass
    
    @abstractmethod
    def to_grayscale(self) -> 'GrayscaleCatImage':
        pass
    
    def _simple_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Упрощенное обнаружение границ для избежания циклических импортов."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Простой оператор Собеля
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        if magnitude.max() > 0:
            return (magnitude / magnitude.max() * 255).astype(np.uint8)
        return np.zeros_like(gray, dtype=np.uint8)
    
    def custom_edge_detection(self) -> np.ndarray:
        """Выделение контуров пользовательским методом."""
        return self._simple_edge_detection(self._image_data)
    
    def library_edge_detection(self) -> np.ndarray:
        """Выделение контуров библиотечным методом (Canny)."""
        if len(self._image_data.shape) == 3:
            gray_image = cv2.cvtColor(self._image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = self._image_data
            
        edges = cv2.Canny(gray_image, 100, 200)
        return edges
    
    def __add__(self, other: 'CatImage') -> np.ndarray:
        if self._image_data.shape != other.image_data.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")
        return (self._image_data.astype(float) + other.image_data.astype(float)).astype(np.uint8)
    
    def __sub__(self, other: 'CatImage') -> np.ndarray:
        if self._image_data.shape != other.image_data.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")
        return np.clip(self._image_data.astype(float) - other.image_data.astype(float), 0, 255).astype(np.uint8)
    
    def __str__(self) -> str:
        return f"CatImage(breed={self._breed}, shape={self._image_data.shape}, url={self._image_url})"


class ColorCatImage(CatImage):
    """Реализация для цветных изображений."""
    
    @property
    def is_grayscale(self) -> bool:
        return False
    
    def to_grayscale(self) -> 'GrayscaleCatImage':
        if len(self._image_data.shape) != 3:
            raise ValueError("Изображение уже в оттенках серого")
        
        gray_data = np.dot(self._image_data[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return GrayscaleCatImage(gray_data, self._image_url, self._breed)


class GrayscaleCatImage(CatImage):
    """Реализация для ч/б изображений."""
    
    @property
    def is_grayscale(self) -> bool:
        return True
    
    def to_grayscale(self) -> 'GrayscaleCatImage':
        return self