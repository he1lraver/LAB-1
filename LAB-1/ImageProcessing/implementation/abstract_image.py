from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from implementation.image_processing import ImageProcessing


class CatImage(ABC):
    """
    Абстрактный класс для представления изображения животного.
    Определяет интерфейс для работы с цветными и черно-белыми изображениями.
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
        return width, height
    
    def custom_edge_detection(self) -> np.ndarray:
        """
        Выделение контуров пользовательским методом (Собель).
        
        Returns:
            Бинарное изображение с контурами
        """
        return self._image_processor.edge_detection(self._image_data)
    
    def library_edge_detection(self) -> np.ndarray:
        """
        Выделение контуров библиотечным методом (Кэнни).
        
        Returns:
            Бинарное изображение с контурами
        """
        return self._image_processor.library_edge_detection(self._image_data)
    
    def corner_detection(self) -> np.ndarray:
        """
        Обнаружение углов на изображении (Харрис).
        
        Returns:
            RGB изображение с отмеченными углами (красные крестики)
        """
        return self._image_processor.corner_detection(self._image_data)
    
    def circle_detection(self) -> np.ndarray:
        """
        Обнаружение окружностей на изображении (Хаф).
        
        Returns:
            RGB изображение с отмеченными окружностями (зелёные кружки)
        """
        return self._image_processor.circle_detection(self._image_data)
    
    def convolution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применение свертки с заданным ядром.
        
        Args:
            kernel: Ядро для свёртки
            
        Returns:
            Результат свёртки
        """
        return self._image_processor.convolution(self._image_data, kernel)
    
    def _prepare_data_for_operation(self, data: np.ndarray) -> np.ndarray:
        """
        Подготовка данных изображения для арифметических операций.
        Приводит изображения к одинаковому количеству каналов.
        
        Args:
            data: Массив изображения
            
        Returns:
            Приведенный массив изображения (3 канала - RGB)
        """
        # Если изображение одноканальное (grayscale), преобразуем в RGB
        if len(data.shape) == 2:
            return np.stack([data] * 3, axis=-1)
        elif len(data.shape) == 3 and data.shape[2] == 1:
            return np.repeat(data, 3, axis=2)
        else:
            return data
    
    def __add__(self, other: Union['CatImage', np.ndarray]) -> 'CatImage':
        """
        Сложение изображений (перегрузка оператора +).
        
        Args:
            other: Другое изображение (CatImage) или numpy массив 
                   (например, результат edge_detection)
        
        Returns:
            Новый объект CatImage того же типа с результатом сложения
            
        Raises:
            TypeError: Если тип other не поддерживается
            ValueError: Если размеры изображений не совпадают
            
        Examples:
            result = cat_image1 + cat_image2
            edges = cat_image.custom_edge_detection()
            result = cat_image + edges
        """
        if isinstance(other, CatImage):
            other_data = other.image_data
            breed_name = f"{self._breed}+{other.breed}"
        elif isinstance(other, np.ndarray):
            other_data = other
            breed_name = f"{self._breed}+processed"
        else:
            raise TypeError(f"Неподдерживаемый тип для сложения: {type(other)}")
        
        # Проверка размеров
        if self._image_data.shape[:2] != other_data.shape[:2]:
            raise ValueError(
                f"Размеры изображений не совпадают: {self._image_data.shape[:2]} vs {other_data.shape[:2]}"
            )
        
        # Приведение к одинаковому количеству каналов
        self_data = self._prepare_data_for_operation(self._image_data)
        other_data = self._prepare_data_for_operation(other_data)
        
        # Выполнение сложения с клиппингом значений
        result_data = np.clip(
            self_data.astype(float) + other_data.astype(float), 
            0, 
            255
        ).astype(np.uint8)
        
        return self.__class__(result_data, f"combined_{self._breed}", breed_name)
    
    def __sub__(self, other: Union['CatImage', np.ndarray]) -> 'CatImage':
        """
        Вычитание изображений (перегрузка оператора -).
        
        Args:
            other: Другое изображение (CatImage) или numpy массив 
                   (например, результат edge_detection)
        
        Returns:
            Новый объект CatImage того же типа с результатом вычитания
            
        Raises:
            TypeError: Если тип other не поддерживается
            ValueError: Если размеры изображений не совпадают
            
        Examples:
            result = cat_image1 - cat_image2
            edges = cat_image.custom_edge_detection()
            result = cat_image - edges
        """
        if isinstance(other, CatImage):
            other_data = other.image_data
            breed_name = f"{self._breed}-{other.breed}"
        elif isinstance(other, np.ndarray):
            other_data = other
            breed_name = f"{self._breed}-processed"
        else:
            raise TypeError(f"Неподдерживаемый тип для вычитания: {type(other)}")
        
        # Проверка размеров
        if self._image_data.shape[:2] != other_data.shape[:2]:
            raise ValueError(
                f"Размеры изображений не совпадают: {self._image_data.shape[:2]} vs {other_data.shape[:2]}"
            )
        
        # Приведение к одинаковому количеству каналов
        self_data = self._prepare_data_for_operation(self._image_data)
        other_data = self._prepare_data_for_operation(other_data)
        
        # Выполнение вычитания с клиппингом значений
        result_data = np.clip(
            self_data.astype(float) - other_data.astype(float), 
            0, 
            255
        ).astype(np.uint8)
        
        return self.__class__(result_data, f"subtracted_{self._breed}", breed_name)
    
    def __str__(self) -> str:
        """Строковое представление объекта."""
        return f"CatImage(breed={self._breed}, shape={self._image_data.shape}, url={self._image_url})"


class ColorCatImage(CatImage):
    """
    Конкретный класс для цветных изображений (RGB).
    Реализует все абстрактные методы базового класса.
    """
    
    @property
    def is_grayscale(self) -> bool:
        """Цветное изображение не является черно-белым."""
        return False
    
    def to_grayscale(self) -> 'GrayscaleCatImage':
        """
        Преобразование из RGB в черно-белое.
        
        Returns:
            GrayscaleCatImage объект
        """
        gray_data = self._image_processor.rgb_to_grayscale(self._image_data)
        return GrayscaleCatImage(gray_data, self._image_url, self._breed)
    
    def to_color(self) -> 'ColorCatImage':
        """
            Уже цветное, возвращаем себя (полиморфизм).
        
        Returns:
            Сам объект ColorCatImage
        """
        return self


class GrayscaleCatImage(CatImage):
    """
    Конкретный класс для черно-белых изображений.
    Реализует все абстрактные методы базового класса.
    """
    
    @property
    def is_grayscale(self) -> bool:
        """Черно-белое изображение является черно-белым."""
        return True
    
    def to_grayscale(self) -> 'GrayscaleCatImage':
        """
        Уже черно-белое, возвращаем себя (полиморфизм).
        
        Returns:
            Сам объект GrayscaleCatImage
        """
        return self
    
    def to_color(self) -> 'ColorCatImage':
        """
        Преобразование из черно-белого в RGB.
        
        Returns:
            ColorCatImage объект
        """
        if len(self._image_data.shape) == 2:
            # Дублируем один канал три раза
            color_data = np.stack([self._image_data] * 3, axis=-1)
        else:
            color_data = self._image_data
        return ColorCatImage(color_data, self._image_url, self._breed)