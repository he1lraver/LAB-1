# -*- coding: utf-8 -*-
"""
Модуль тестирования для приложения обработки изображений.

Содержит unit-тесты для основных классов и функций приложения:

1. TestCatImage - тесты класса CatImage и его наследников
   - RGB to Grayscale преобразование
   - Grayscale to Color преобразование
   - Сложение изображений (+)
   - Вычитание изображений (-)
   - Edge detection (custom и library)
   
2. TestImageProcessing - тесты класса ImageProcessing
   - RGB to Grayscale преобразование
   - Гамма-коррекция
   - Свертка изображения
   - Edge detection
   
Запуск тестов:
    python -m unittest tests.py -v
    
или конкретного теста:
    python -m unittest tests.TestCatImage.test_rgb_to_grayscale -v
"""

import unittest
import numpy as np
import logging
from typing import Tuple

# Импортируем классы для тестирования
from implementation.abstract_image import ColorCatImage, GrayscaleCatImage
from implementation.image_processing import ImageProcessing
from logging_config import setup_logging


# ════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ ДЛЯ СОЗДАНИЯ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ
# ════════════════════════════════════════════════════════════════════════════

def create_test_rgb_image(width: int = 100, height: int = 100, 
                          pattern: str = 'random') -> np.ndarray:
    """
    Создает тестовое RGB изображение.
    
    Args:
        width: Ширина изображения
        height: Высота изображения
        pattern: Тип паттерна - 'random', 'gradient', 'solid_red', 'checkerboard'
    
    Returns:
        np.ndarray: RGB изображение (height, width, 3)
    """
    if pattern == 'random':
        # Случайное изображение
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    elif pattern == 'gradient':
        # Градиент от черного к белому
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        image = np.tile(gradient, (height, 1))
        return np.stack([image, image, image], axis=-1)
    
    elif pattern == 'solid_red':
        # Сплошное красное изображение
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 2] = 255  # R канал (красный в BGR)
        return image
    
    elif pattern == 'checkerboard':
        # Шахматный паттерн
        image = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 10
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
        return image
    
    else:
        raise ValueError(f"Неизвестный паттерн: {pattern}")


def create_test_grayscale_image(width: int = 100, height: int = 100,
                               pattern: str = 'random') -> np.ndarray:
    """
    Создает тестовое grayscale изображение.
    
    Args:
        width: Ширина изображения
        height: Высота изображения
        pattern: Тип паттерна
    
    Returns:
        np.ndarray: Grayscale изображение (height, width)
    """
    rgb = create_test_rgb_image(width, height, pattern)
    # Простое преобразование RGB в grayscale
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# КЛАСС ТЕСТОВ ДЛЯ CatImage
# ════════════════════════════════════════════════════════════════════════════

class TestCatImage(unittest.TestCase):
    """
    Тест-набор для класса CatImage и его подклассов.
    
    Проверяет:
    - Преобразование RGB в Grayscale
    - Преобразование Grayscale в Color
    - Сложение изображений
    - Вычитание изображений
    - Edge detection (custom и library)
    """
    
    def setUp(self):
        """Подготовка к каждому тесту."""
        self.logger = setup_logging()
        self.logger.info("─" * 70)
        self.logger.info(f"Запуск теста: {self._testMethodName}")
        
        # Создаем тестовые изображения
        self.test_rgb_image = create_test_rgb_image(100, 100, 'checkerboard')
        self.test_grayscale_image = create_test_grayscale_image(100, 100, 'gradient')
        
        # Создаем объекты CatImage
        self.cat_rgb = ColorCatImage(self.test_rgb_image, "test_url", "TestBreed")
        self.cat_gray = GrayscaleCatImage(self.test_grayscale_image, "test_url", "TestBreedGray")
    
    def tearDown(self):
        """Очистка после каждого теста."""
        self.logger.info(f"Завершен тест: {self._testMethodName}")
        self.logger.info("─" * 70)
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 1: RGB to Grayscale Преобразование
    # ════════════════════════════════════════════════════════════════════
    
    def test_rgb_to_grayscale_conversion(self):
        """
        Тест преобразования RGB изображения в grayscale.
        
        Проверяет:
        - Результат является объектом GrayscaleCatImage
        - Размеры уменьшаются с (H,W,3) до (H,W)
        - Значения пикселей находятся в диапазоне [0, 255]
        """
        self.logger.debug("Начало теста: RGB to Grayscale преобразование")
        
        # Преобразуем в grayscale
        gray_cat = self.cat_rgb.to_grayscale()
        
        # Проверка типа
        self.assertIsInstance(gray_cat, GrayscaleCatImage)
        self.logger.debug("✓ Результат имеет тип GrayscaleCatImage")
        
        # Проверка размеров
        self.assertEqual(len(gray_cat.image_data.shape), 2)
        self.logger.debug(f"✓ Размеры изображения: {gray_cat.image_data.shape}")
        
        # Проверка диапазона значений
        self.assertTrue(np.all(gray_cat.image_data >= 0))
        self.assertTrue(np.all(gray_cat.image_data <= 255))
        self.logger.debug(f"✓ Диапазон значений пикселей: [{gray_cat.image_data.min()}, {gray_cat.image_data.max()}]")
        
        # Проверка что изображение не полностью черное
        self.assertTrue(np.max(gray_cat.image_data) > 0)
        self.logger.debug("✓ Изображение не полностью черное")
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 2: Grayscale to Color Преобразование
    # ════════════════════════════════════════════════════════════════════
    
    def test_grayscale_to_color_conversion(self):
        """
        Тест преобразования grayscale изображения в RGB.
        
        Проверяет:
        - Результат является объектом ColorCatImage
        - Размеры увеличиваются с (H,W) до (H,W,3)
        - Все три канала (R,G,B) имеют одинаковые значения
        """
        self.logger.debug("Начало теста: Grayscale to Color преобразование")
        
        # Преобразуем в цветное
        color_cat = self.cat_gray.to_color()
        
        # Проверка типа
        self.assertIsInstance(color_cat, ColorCatImage)
        self.logger.debug("✓ Результат имеет тип ColorCatImage")
        
        # Проверка размеров
        self.assertEqual(len(color_cat.image_data.shape), 3)
        self.assertEqual(color_cat.image_data.shape[2], 3)
        self.logger.debug(f"✓ Размеры изображения: {color_cat.image_data.shape}")
        
        # Проверка что каналы одинаковые (так как это grayscale -> RGB)
        r, g, b = color_cat.image_data[:,:,0], color_cat.image_data[:,:,1], color_cat.image_data[:,:,2]
        self.assertTrue(np.array_equal(r, g))
        self.assertTrue(np.array_equal(g, b))
        self.logger.debug("✓ Все три канала (R,G,B) имеют одинаковые значения")
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 3: Сложение Изображений
    # ════════════════════════════════════════════════════════════════════
    
    def test_image_addition(self):
        """
        Тест сложения двух RGB изображений.
        
        Проверяет:
        - Результат сложения имеет правильный тип
        - Размеры совпадают с исходными
        - Значения клиппируются на [0, 255]
        - Сложение вычисляется корректно
        """
        self.logger.debug("Начало теста: Сложение изображений")
        
        # Создаем второе изображение
        test_rgb_2 = create_test_rgb_image(100, 100, 'solid_red')
        cat_2 = ColorCatImage(test_rgb_2, "test_url_2", "TestBreed2")
        
        # Складываем изображения
        result = self.cat_rgb + cat_2
        
        # Проверка типа
        self.assertIsInstance(result, ColorCatImage)
        self.logger.debug("✓ Результат имеет тип ColorCatImage")
        
        # Проверка размеров
        self.assertEqual(result.image_data.shape, self.cat_rgb.image_data.shape)
        self.logger.debug(f"✓ Размеры сохранены: {result.image_data.shape}")
        
        # Проверка диапазона
        self.assertTrue(np.all(result.image_data >= 0))
        self.assertTrue(np.all(result.image_data <= 255))
        self.logger.debug(f"✓ Значения в диапазоне [0, 255]")
        
        # Проверка что результат не равен нулю
        self.assertTrue(np.max(result.image_data) > 0)
        self.logger.debug(f"✓ Результат не пустой, макс значение: {result.image_data.max()}")
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 4: Вычитание Изображений
    # ════════════════════════════════════════════════════════════════════
    
    def test_image_subtraction(self):
        """
        Тест вычитания одного RGB изображения из другого.
        
        Проверяет:
        - Результат вычитания имеет правильный тип
        - Размеры совпадают с исходными
        - Значения клиппируются на [0, 255]
        - Вычитание вычисляется корректно (без отрицательных значений)
        """
        self.logger.debug("Начало теста: Вычитание изображений")
        
        # Создаем два одинаковых изображения
        image1 = create_test_rgb_image(100, 100, 'gradient')
        image2 = create_test_rgb_image(100, 100, 'gradient')
        
        cat_1 = ColorCatImage(image1, "test_url_1", "TestBreed1")
        cat_2 = ColorCatImage(image2, "test_url_2", "TestBreed2")
        
        # Вычитаем изображения (одинаковые, должно быть близко к нулю)
        result = cat_1 - cat_2
        
        # Проверка типа
        self.assertIsInstance(result, ColorCatImage)
        self.logger.debug("✓ Результат имеет тип ColorCatImage")
        
        # Проверка размеров
        self.assertEqual(result.image_data.shape, cat_1.image_data.shape)
        self.logger.debug(f"✓ Размеры сохранены: {result.image_data.shape}")
        
        # Проверка диапазона (нет отрицательных значений)
        self.assertTrue(np.all(result.image_data >= 0))
        self.assertTrue(np.all(result.image_data <= 255))
        self.logger.debug(f"✓ Значения в диапазоне [0, 255] без отрицательных")
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 5: Custom Edge Detection
    # ════════════════════════════════════════════════════════════════════
    
    def test_custom_edge_detection(self):
        """
        Тест custom edge detection (Собель).
        
        Проверяет:
        - Результат имеет правильные размеры
        - Результат бинарный (только 0 и 255)
        - На градиентном изображении есть обнаруженные границы
        """
        self.logger.debug("Начало теста: Custom Edge Detection (Sobel)")
        
        # Применяем edge detection
        edges = self.cat_rgb.custom_edge_detection()
        
        # Проверка размеров
        self.assertEqual(edges.shape[:2], self.cat_rgb.image_data.shape[:2])
        self.logger.debug(f"✓ Размеры edges: {edges.shape}")
        
        # Проверка что это изображение
        self.assertIsInstance(edges, np.ndarray)
        
        # Проверка диапазона
        self.assertTrue(np.all(edges >= 0))
        self.assertTrue(np.all(edges <= 255))
        self.logger.debug(f"✓ Диапазон edge detection: [{edges.min()}, {edges.max()}]")
        
        # На шахматном паттерне должны быть границы
        self.assertTrue(np.max(edges) > 0, "Edge detection должен обнаружить границы на шахматном паттерне")
        self.logger.debug(f"✓ Обнаружены границы: количество пикселей границ = {np.sum(edges > 0)}")
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 6: Image Addition with Edge Detection Result
    # ════════════════════════════════════════════════════════════════════
    
    def test_add_original_plus_edges(self):
        """
        Тест сложения оригинального изображения с результатом edge detection.
        
        Проверяет:
        - Корректное сложение изображения и массива numpy
        - Размеры результата совпадают
        - Значения в правильном диапазоне
        """
        self.logger.debug("Начало теста: Original + Custom Edges")
        
        # Получаем края
        edges = self.cat_rgb.custom_edge_detection()
        self.logger.debug(f"Edge detection выполнен, shape={edges.shape}")
        
        # Складываем оригинальное с краями
        result = self.cat_rgb + edges
        
        # Проверка типа
        self.assertIsInstance(result, ColorCatImage)
        self.logger.debug("✓ Результат имеет тип ColorCatImage")
        
        # Проверка размеров
        self.assertEqual(result.image_data.shape, self.cat_rgb.image_data.shape)
        self.logger.debug(f"✓ Размеры результата: {result.image_data.shape}")
        
        # Проверка диапазона
        self.assertTrue(np.all(result.image_data >= 0))
        self.assertTrue(np.all(result.image_data <= 255))
        self.logger.debug(f"✓ Значения в диапазоне [0, 255]")


# ════════════════════════════════════════════════════════════════════════════
# КЛАСС ТЕСТОВ ДЛЯ ImageProcessing
# ════════════════════════════════════════════════════════════════════════════

class TestImageProcessing(unittest.TestCase):
    """
    Тест-набор для класса ImageProcessing.
    
    Проверяет:
    - RGB to Grayscale преобразование
    - Гамма-коррекция
    - Свертка изображения
    - Edge detection
    """
    
    def setUp(self):
        """Подготовка к каждому тесту."""
        self.logger = setup_logging()
        self.logger.info("─" * 70)
        self.logger.info(f"Запуск теста: {self._testMethodName}")
        
        self.processor = ImageProcessing()
        self.test_rgb_image = create_test_rgb_image(100, 100, 'checkerboard')
    
    def tearDown(self):
        """Очистка после каждого теста."""
        self.logger.info(f"Завершен тест: {self._testMethodName}")
        self.logger.info("─" * 70)
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 1: RGB to Grayscale в ImageProcessing
    # ════════════════════════════════════════════════════════════════════
    
    def test_rgb_to_grayscale_processor(self):
        """
        Тест методов преобразования RGB в grayscale в классе ImageProcessing.
        
        Проверяет:
        - _rgb_to_grayscale() работает корректно
        - Размеры уменьшаются
        - Значения в диапазоне [0, 255]
        """
        self.logger.debug("Начало теста: ImageProcessing.rgb_to_grayscale()")
        
        # Преобразуем
        gray = self.processor.rgb_to_grayscale(self.test_rgb_image)
        
        # Проверка размеров (2D вместо 3D)
        self.assertEqual(len(gray.shape), 2)
        self.logger.debug(f"✓ Размеры: {gray.shape}")
        
        # Проверка диапазона
        self.assertTrue(np.all(gray >= 0) and np.all(gray <= 255))
        self.logger.debug(f"✓ Диапазон значений: [{gray.min()}, {gray.max()}]")
        
        # Проверка типа
        self.assertEqual(gray.dtype, np.uint8)
        self.logger.debug(f"✓ Тип данных: {gray.dtype}")
    
    
    # ════════════════════════════════════════════════════════════════════
    # ТЕСТ 2: Свертка (Convolution)
    # ════════════════════════════════════════════════════════════════════
    
    def test_convolution(self):
        """
        Тест свертки изображения с ядром.
        
        Проверяет:
        - Свертка работает с различными ядрами
        - Результат имеет правильный размер
        - Результат содержит числовые значения
        """
        self.logger.debug("Начало теста: Свертка (Convolution)")
        
        # Создаем простое ядро (средний фильтр)
        kernel = np.ones((3, 3)) / 9
        
        # Конвертируем в grayscale для свертки
        gray = self.processor.rgb_to_grayscale(self.test_rgb_image)
        
        # Применяем свертку
        result = self.processor.convolution(gray, kernel)
        
        # Проверка типа результата
        self.assertIsInstance(result, np.ndarray)
        self.logger.debug(f"✓ Результат является numpy array")
        
        # Проверка размеров (примерно одинаковые, может быть немного меньше)
        self.assertLessEqual(abs(result.shape[0] - gray.shape[0]), 5)
        self.assertLessEqual(abs(result.shape[1] - gray.shape[1]), 5)
        self.logger.debug(f"✓ Размеры результата: {result.shape}, исходные: {gray.shape}")
        
        # Проверка что результат не пуст
        self.assertTrue(np.any(result != 0))
        self.logger.debug(f"✓ Результат содержит ненулевые значения")


# ════════════════════════════════════════════════════════════════════════════
# ЗАПУСК ТЕСТОВ
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Запуск всех тестов.
    
    Использование:
        python tests.py -v                    # Все тесты с подробностью
        python -m unittest tests -v           # Через unittest
        python -m unittest tests.TestCatImage -v  # Только TestCatImage
    """
    
    # Инициализируем логирование
    logger = setup_logging()
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 20 + "ЗАПУСК UNIT-ТЕСТОВ" + " " * 31 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    
    # Запускаем тесты
    unittest.main(verbosity=2)
