"""
abstract_image.py

Абстрактные классы для работы с изображениями животных с логированием.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from .image_processing import ImageProcessing
from image_processor.logging_config import get_logger



class CatImage(ABC):
    """
    Абстрактный класс для представления изображения животного с логированием.
    """

    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        """
        Инициализирует объект изображения.

        Args:
            image_data: Данные изображения (numpy array)
            image_url: URL изображения
            breed: Порода животного
        """
        self._image_data = image_data
        self._image_url = image_url
        self._breed = breed
        self._image_processor = ImageProcessing()
        self.logger = get_logger()

        self.logger.debug(f"CatImage created: breed={breed}, shape={image_data.shape}, url={image_url[:50]}...")

    @property
    def image_data(self) -> np.ndarray:
        """Возвращает данные изображения."""
        return self._image_data

    @property
    def image_url(self) -> str:
        """Возвращает URL изображения."""
        return self._image_url

    @property
    def breed(self) -> str:
        """Возвращает породу животного."""
        return self._breed

    @property
    @abstractmethod
    def is_grayscale(self) -> bool:
        """Абстрактное свойство: является ли изображение черно-белым."""
        pass

    @abstractmethod
    def to_grayscale(self) -> 'GrayscaleCatImage':
        """Абстрактный метод: преобразовать в черно-белое изображение."""
        pass

    @abstractmethod
    def to_color(self) -> 'ColorCatImage':
        """Абстрактный метод: преобразовать в цветное изображение."""
        pass

    def get_dimensions(self) -> tuple:
        """
        Возвращает размеры изображения.

        Returns:
            Кортеж (width, height)
        """
        height, width = self._image_data.shape[:2]
        self.logger.debug(f"Image dimensions: {width}x{height}")
        return width, height

    def custom_edge_detection(self) -> np.ndarray:
        """
        Выделение контуров пользовательским методом (Собель) с логированием.
        """
        self.logger.debug(f"Custom edge detection started for breed: {self._breed}")
        result = self._image_processor.edge_detection(self._image_data)
        self.logger.debug(f"Custom edge detection completed")
        return result

    def library_edge_detection(self) -> np.ndarray:
        """
        Выделение контуров библиотечным методом (Кэнни) с логированием.
        """
        self.logger.debug(f"Library edge detection started for breed: {self._breed}")
        result = self._image_processor.library_edge_detection(self._image_data)
        self.logger.debug(f"Library edge detection completed")
        return result

    def corner_detection(self) -> np.ndarray:
        """
        Обнаружение углов на изображении (Харрис) с логированием.
        """
        self.logger.debug(f"Corner detection started for breed: {self._breed}")
        result = self._image_processor.corner_detection(self._image_data)
        self.logger.debug(f"Corner detection completed")
        return result

    def circle_detection(self) -> np.ndarray:
        """
        Обнаружение окружностей на изображении (Хаф) с логированием.
        """
        self.logger.debug(f"Circle detection started for breed: {self._breed}")
        result = self._image_processor.circle_detection(self._image_data)
        self.logger.debug(f"Circle detection completed")
        return result

    def convolution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применение свертки с заданным ядром с логированием.
        """
        self.logger.debug(f"Convolution started. Kernel shape: {kernel.shape}")
        result = self._image_processor.convolution(self._image_data, kernel)
        self.logger.debug(f"Convolution completed")
        return result

    def _prepare_data_for_operation(self, data: np.ndarray) -> np.ndarray:
        """
        Подготовка данных изображения для арифметических операций.
        """
        if len(data.shape) == 2:
            self.logger.debug("Converting grayscale to RGB (3 channels)")
            return np.stack([data] * 3, axis=-1)
        elif len(data.shape) == 3 and data.shape[2] == 1:
            self.logger.debug("Converting single channel to RGB")
            return np.repeat(data, 3, axis=2)
        else:
            return data

    def __add__(self, other: Union['CatImage', np.ndarray]) -> 'CatImage':
        """
        Сложение изображений (перегрузка оператора +) с логированием.
        """
        self.logger.debug(f"Addition operation started")

        if isinstance(other, CatImage):
            other_data = other.image_data
            breed_name = f"{self._breed}+{other.breed}"
        elif isinstance(other, np.ndarray):
            other_data = other
            breed_name = f"{self._breed}+processed"
        else:
            self.logger.error(f"Unsupported type for addition: {type(other)}")
            raise TypeError(f"Unsupported type for addition: {type(other)}")

        if self._image_data.shape[:2] != other_data.shape[:2]:
            self.logger.error(f"Image dimensions mismatch: {self._image_data.shape[:2]} vs {other_data.shape[:2]}")
            raise ValueError(f"Image dimensions mismatch")

        self_data = self._prepare_data_for_operation(self._image_data)
        other_data = self._prepare_data_for_operation(other_data)

        result_data = np.clip(
            self_data.astype(float) + other_data.astype(float),
            0,
            255
        ).astype(np.uint8)

        self.logger.debug(f"Addition operation completed. Result shape: {result_data.shape}")
        return self.__class__(result_data, f"combined_{self._breed}", breed_name)

    def __sub__(self, other: Union['CatImage', np.ndarray]) -> 'CatImage':
        """
        Вычитание изображений (перегрузка оператора -) с логированием.
        """
        self.logger.debug(f"Subtraction operation started")

        if isinstance(other, CatImage):
            other_data = other.image_data
            breed_name = f"{self._breed}-{other.breed}"
        elif isinstance(other, np.ndarray):
            other_data = other
            breed_name = f"{self._breed}-processed"
        else:
            self.logger.error(f"Unsupported type for subtraction: {type(other)}")
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")

        if self._image_data.shape[:2] != other_data.shape[:2]:
            self.logger.error(f"Image dimensions mismatch")
            raise ValueError(f"Image dimensions mismatch")

        self_data = self._prepare_data_for_operation(self._image_data)
        other_data = self._prepare_data_for_operation(other_data)

        result_data = np.clip(
            self_data.astype(float) - other_data.astype(float),
            0,
            255
        ).astype(np.uint8)

        self.logger.debug(f"Subtraction operation completed. Result shape: {result_data.shape}")
        return self.__class__(result_data, f"subtracted_{self._breed}", breed_name)

    def __str__(self) -> str:
        """Строковое представление объекта."""
        return f"CatImage(breed={self._breed}, shape={self._image_data.shape})"


class ColorCatImage(CatImage):
    """
    Конкретный класс для цветных изображений (RGB) с логированием.
    """

    @property
    def is_grayscale(self) -> bool:
        """Цветное изображение не является черно-белым."""
        return False

    def to_grayscale(self) -> 'GrayscaleCatImage':
        """
        Преобразование из RGB в черно-белое с логированием.
        """
        self.logger.debug(f"Converting ColorCatImage to grayscale: {self._breed}")
        gray_data = self._image_processor.rgb_to_grayscale(self._image_data)
        return GrayscaleCatImage(gray_data, self._image_url, self._breed)

    def to_color(self) -> 'ColorCatImage':
        """Уже цветное, возвращаем себя."""
        return self


class GrayscaleCatImage(CatImage):
    """
    Конкретный класс для черно-белых изображений с логированием.
    """

    @property
    def is_grayscale(self) -> bool:
        """Черно-белое изображение является черно-белым."""
        return True

    def to_grayscale(self) -> 'GrayscaleCatImage':
        """Уже черно-белое, возвращаем себя."""
        return self

    def to_color(self) -> 'ColorCatImage':
        """
        Преобразование из черно-белого в RGB с логированием.
        """
        self.logger.debug(f"Converting GrayscaleCatImage to color: {self._breed}")
        if len(self._image_data.shape) == 2:
            color_data = np.stack([self._image_data] * 3, axis=-1)
        else:
            color_data = self._image_data
        return ColorCatImage(color_data, self._image_url, self._breed)