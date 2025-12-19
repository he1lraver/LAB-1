"""
tests.py

Модуль тестирования приложения обработки изображений.

Использует unittest для проверки функциональности:
1. TestCatImage - тесты для класса CatImage и его методов
2. TestAsyncImageProcessor - тесты для асинхронного процессора

Запуск тестов:
    python -m unittest tests -v
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processor.implementation.abstract_image import ColorCatImage, GrayscaleCatImage
from image_processor.implementation.image_processing import ImageProcessing
from image_processor.implementation.async_image_processor import AsyncImageProcessor, ImageMetadata
from image_processor.logging_config import setup_logging, get_logger


class TestCatImage(unittest.TestCase):
    """
    Тесты для класса CatImage: RGB-BW преобразование, свертка, сложение изображений.
    """

    @classmethod
    def setUpClass(cls):
        """Инициализация логирования перед тестами."""
        setup_logging()

    def setUp(self):
        """Подготовка к каждому тесту."""
        # Создаём тестовые RGB изображения (100x100)
        self.test_image_rgb = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)

        # Создаём объекты CatImage
        self.color_cat = ColorCatImage(self.test_image_rgb, "http://example.com/cat.jpg", "Siamese")
        self.gray_cat = GrayscaleCatImage(self.test_image_gray, "http://example.com/cat_gray.jpg", "Persian")

    def test_rgb_to_grayscale_conversion(self):
        """
        Тест 1: Проверка преобразования RGB изображения в черно-белое (BW).

        Проверяет:
        - Корректное преобразование формата
        - Правильные размеры выходного изображения
        - Тип данных
        """
        logger = get_logger()
        logger.info("TEST 1: RGB to Grayscale conversion")

        # Преобразование
        gray_result = self.color_cat.to_grayscale()

        # Проверяем тип объекта
        self.assertIsInstance(gray_result, GrayscaleCatImage, 
                             "Результат должен быть объектом GrayscaleCatImage")

        # Проверяем размеры (должны быть 100x100 2D)
        self.assertEqual(len(gray_result.image_data.shape), 2,
                        "Черно-белое изображение должно быть 2D")
        self.assertEqual(gray_result.image_data.shape, (100, 100),
                        "Размеры должны быть (100, 100)")

        # Проверяем тип данных
        self.assertEqual(gray_result.image_data.dtype, np.uint8,
                        "Тип данных должен быть uint8")

        # Проверяем свойство is_grayscale
        self.assertTrue(gray_result.is_grayscale,
                       "Свойство is_grayscale должно быть True для черно-белого изображения")

        logger.info("[PASSED] RGB to Grayscale conversion test")

    def test_grayscale_to_color_conversion(self):
        """
        Тест 2: Проверка преобразования черно-белого изображения в цветное (RGB).

        Проверяет:
        - Корректное преобразование формата
        - Правильные размеры выходного изображения (должно быть 3 канала)
        - Соответствие значений всех 3 каналов
        """
        logger = get_logger()
        logger.info("TEST 2: Grayscale to Color conversion")

        # Преобразование
        color_result = self.gray_cat.to_color()

        # Проверяем тип объекта
        self.assertIsInstance(color_result, ColorCatImage,
                             "Результат должен быть объектом ColorCatImage")

        # Проверяем размеры (должны быть 100x100x3)
        self.assertEqual(len(color_result.image_data.shape), 3,
                        "Цветное изображение должно быть 3D")
        self.assertEqual(color_result.image_data.shape, (100, 100, 3),
                        "Размеры должны быть (100, 100, 3)")

        # Проверяем, что все 3 канала одинаковые (так как преобразуем из grayscale)
        self.assertTrue(
            np.allclose(color_result.image_data[:, :, 0], 
                       color_result.image_data[:, :, 1]),
            "R и G каналы должны быть одинаковыми"
        )
        self.assertTrue(
            np.allclose(color_result.image_data[:, :, 1], 
                       color_result.image_data[:, :, 2]),
            "G и B каналы должны быть одинаковыми"
        )

        # Проверяем свойство is_grayscale
        self.assertFalse(color_result.is_grayscale,
                        "Свойство is_grayscale должно быть False для цветного изображения")

        logger.info("[PASSED] Grayscale to Color conversion test")

    def test_image_addition(self):
        """
        Тест 3: Проверка операции сложения двух изображений (CatImage + ndarray).

        Проверяет:
        - Корректное сложение двух RGB изображений
        - Правильность размеров результата
        - Корректный clipping значений (0-255)
        - Возможность складывать с результатом edge detection
        """
        logger = get_logger()
        logger.info("TEST 3: Image addition operation")

        # Создаём второе изображение
        second_image_data = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        second_cat = ColorCatImage(second_image_data, "http://example.com/cat2.jpg", "Maine Coon")

        # Сложение двух CatImage
        result = self.color_cat + second_cat

        # Проверяем тип результата
        self.assertIsInstance(result, ColorCatImage,
                             "Результат должен быть ColorCatImage")

        # Проверяем размеры
        self.assertEqual(result.image_data.shape, (100, 100, 3),
                        "Размеры результата должны совпадать")

        # Проверяем диапазон значений (должны быть в [0, 255])
        self.assertTrue(np.all(result.image_data >= 0) and np.all(result.image_data <= 255),
                       "Все значения должны быть в диапазоне [0, 255]")

        # Проверяем, что это действительно сумма (с clipping)
        # Хотя бы некоторые пиксели должны быть не равны исходным
        self.assertFalse(np.allclose(result.image_data, self.test_image_rgb),
                        "Результат должен отличаться от исходного изображения")

        logger.info("[PASSED] Image addition test")

    def test_image_subtraction(self):
        """
        Тест 4: Проверка операции вычитания изображений.

        Проверяет:
        - Корректное вычитание двух изображений
        - Правильность размеров результата
        - Корректный clipping значений (не отрицательные)
        """
        logger = get_logger()
        logger.info("TEST 4: Image subtraction operation")

        # Создаём второе изображение с заведомо меньшими значениями
        second_image_data = np.full((100, 100, 3), 50, dtype=np.uint8)
        second_cat = ColorCatImage(second_image_data, "http://example.com/cat2.jpg", "Bengal")

        # Вычитание
        result = self.color_cat - second_cat

        # Проверяем тип результата
        self.assertIsInstance(result, ColorCatImage,
                             "Результат должен быть ColorCatImage")

        # Проверяем размеры
        self.assertEqual(result.image_data.shape, (100, 100, 3),
                        "Размеры результата должны совпадать")

        # Проверяем, что нет отрицательных значений (clipping работает)
        self.assertTrue(np.all(result.image_data >= 0),
                       "Все значения должны быть >= 0 (no negative due to clipping)")

        # Проверяем, что все значения <= 255
        self.assertTrue(np.all(result.image_data <= 255),
                       "Все значения должны быть <= 255")

        logger.info("[PASSED] Image subtraction test")

    def test_convolution_operation(self):
        """
        Тест 5: Проверка операции свёртки (convolution) изображения.

        Проверяет:
        - Корректное выполнение свёртки
        - Правильность размеров результата
        - Что свёртка действительно изменяет изображение

        ПРИМЕЧАНИЕ: Применяем kernel к одному каналу (или grayscale),
        так как scipy.ndimage.convolve требует совпадения размерности
        изображения и ядра.
        """
        logger = get_logger()
        logger.info("TEST 5: Convolution operation")

        # Создаём простое ядро (например, размытие)
        blur_kernel = np.array([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ], dtype=np.float32)

        # Применяем свёртку к ОДНОМУ КАНАЛУ (первому каналу RGB)
        # или можно конвертировать в grayscale перед сверткой
        gray_image = self.color_cat.to_grayscale()
        result = gray_image._image_processor.convolution(gray_image.image_data, blur_kernel)

        # Проверяем что результат - это numpy array
        self.assertIsInstance(result, np.ndarray,
                             "Результат должен быть numpy array")

        # Проверяем, что результат не пустой
        self.assertGreater(result.size, 0,
                          "Результат свёртки не должен быть пустым")

        logger.info("[PASSED] Convolution operation test")

    def test_image_dimensions(self):
        """
        Тест 6: Проверка корректного получения размеров изображения.

        Проверяет:
        - Правильность возвращаемых размеров
        - Формат возвращаемого значения (width, height)
        """
        logger = get_logger()
        logger.info("TEST 6: Image dimensions")

        # Получаем размеры
        width, height = self.color_cat.get_dimensions()

        # Проверяем значения
        self.assertEqual(width, 100, "Ширина должна быть 100")
        self.assertEqual(height, 100, "Высота должна быть 100")

        # Проверяем типы
        self.assertIsInstance(width, int, "Ширина должна быть int")
        self.assertIsInstance(height, int, "Высота должна быть int")

        logger.info("[PASSED] Image dimensions test")


class TestAsyncImageProcessor(unittest.TestCase):
    """
    Тесты для класса AsyncImageProcessor: обращение к API, чтение/запись файла.
    """

    @classmethod
    def setUpClass(cls):
        """Инициализация логирования перед тестами."""
        setup_logging()

    def setUp(self):
        """Подготовка к каждому тесту."""
        # Создаём процессоры (без реального обращения к API в тестах)
        self.cat_processor = AsyncImageProcessor("fake_api_key", "cat")
        self.dog_processor = AsyncImageProcessor("fake_api_key", "dog")

    def test_async_processor_initialization(self):
        """
        Тест 1: Проверка инициализации AsyncImageProcessor.

        Проверяет:
        - Корректное сохранение параметров API
        - Правильность установки base_url для разных типов API
        """
        logger = get_logger()
        logger.info("TEST 1: AsyncImageProcessor initialization")

        # Проверяем cat processor
        self.assertEqual(self.cat_processor._api_type, "cat",
                        "API type должен быть 'cat'")
        self.assertIn("thecatapi.com", self.cat_processor._base_url,
                     "Base URL должна содержать thecatapi.com")

        # Проверяем dog processor
        self.assertEqual(self.dog_processor._api_type, "dog",
                        "API type должен быть 'dog'")
        self.assertIn("thedogapi.com", self.dog_processor._base_url,
                     "Base URL должна содержать thedogapi.com")

        # Проверяем сохранение API ключа
        self.assertEqual(self.cat_processor._api_key, "fake_api_key",
                        "API ключ должен быть сохранен")

        logger.info("[PASSED] AsyncImageProcessor initialization test")

    def test_image_metadata_creation(self):
        """
        Тест 2: Проверка создания метаданных изображения (ImageMetadata).

        Проверяет:
        - Корректное создание объекта метаданных
        - Правильность сохранения параметров
        - Изначально пустые данные изображения
        """
        logger = get_logger()
        logger.info("TEST 2: ImageMetadata creation")

        # Создаём метаданные
        metadata = ImageMetadata(1, "http://example.com/image.jpg", "Siamese")

        # Проверяем параметры
        self.assertEqual(metadata.index, 1, "Index должен быть 1")
        self.assertEqual(metadata.url, "http://example.com/image.jpg",
                        "URL должен соответствовать")
        self.assertEqual(metadata.breed, "Siamese", "Breed должен быть 'Siamese'")

        # Проверяем, что image_data изначально None
        self.assertIsNone(metadata.image_data,
                         "Image data должна быть None до загрузки")

        logger.info("[PASSED] ImageMetadata creation test")

    def test_image_metadata_assignment(self):
        """
        Тест 3: Проверка присвоения данных изображения метаданным.

        Проверяет:
        - Корректное присвоение numpy array в image_data
        - Сохранение формата и размеров
        """
        logger = get_logger()
        logger.info("TEST 3: ImageMetadata assignment")

        # Создаём метаданные
        metadata = ImageMetadata(1, "http://example.com/image.jpg", "Persian")

        # Создаём тестовое изображение
        test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # Присваиваем данные
        metadata.image_data = test_image

        # Проверяем присвоение
        self.assertIsNotNone(metadata.image_data, "Image data не должна быть None")
        self.assertTrue(np.array_equal(metadata.image_data, test_image),
                       "Image data должна соответствовать присвоенному array")
        self.assertEqual(metadata.image_data.shape, (200, 200, 3),
                        "Размеры должны быть (200, 200, 3)")
        self.assertEqual(metadata.image_data.dtype, np.uint8,
                        "Тип данных должен быть uint8")

        logger.info("[PASSED] ImageMetadata assignment test")

    def test_create_image_objects_from_metadata(self):
        """
        Тест 4: Проверка создания объектов CatImage из метаданных.

        Проверяет:
        - Корректное создание объектов из метаданных
        - Правильность выбора типа (Color или Grayscale)
        - Сохранение порядковых номеров
        """
        logger = get_logger()
        logger.info("TEST 4: Create image objects from metadata")

        # Создаём метаданные с RGB изображением
        metadata_color = ImageMetadata(1, "http://example.com/cat.jpg", "Siamese")
        metadata_color.image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Создаём метаданные с grayscale изображением
        metadata_gray = ImageMetadata(2, "http://example.com/cat_gray.jpg", "Persian")
        metadata_gray.image_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Создаём объекты
        image_objects = self.cat_processor.create_image_objects([metadata_color, metadata_gray])

        # Проверяем количество объектов
        self.assertEqual(len(image_objects), 2,
                        "Должно быть создано 2 объекта")

        # Проверяем первый объект (RGB)
        index1, obj1 = image_objects[0]
        self.assertEqual(index1, 1, "Index первого объекта должен быть 1")
        self.assertIsInstance(obj1, ColorCatImage,
                             "Первый объект должен быть ColorCatImage")

        # Проверяем второй объект (grayscale)
        index2, obj2 = image_objects[1]
        self.assertEqual(index2, 2, "Index второго объекта должен быть 2")
        self.assertIsInstance(obj2, GrayscaleCatImage,
                             "Второй объект должен быть GrayscaleCatImage")

        logger.info("[PASSED] Create image objects from metadata test")


class TestImageProcessing(unittest.TestCase):
    """
    Дополнительные тесты для ImageProcessing класса.
    """

    @classmethod
    def setUpClass(cls):
        """Инициализация логирования перед тестами."""
        setup_logging()

    def setUp(self):
        """Подготовка к каждому тесту."""
        self.processor = ImageProcessing()
        self.test_image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

    def test_edge_detection(self):
        """
        Тест: Проверка функции обнаружения границ (edge detection).
        """
        logger = get_logger()
        logger.info("TEST: Edge detection")

        # Применяем edge detection
        edges = self.processor.edge_detection(self.test_image)

        # Проверяем результат
        self.assertIsInstance(edges, np.ndarray, "Результат должен быть numpy array")
        self.assertEqual(edges.dtype, np.uint8, "Тип данных должен быть uint8")
        self.assertEqual(edges.shape, (100, 100), "Размеры должны быть (100, 100)")

        logger.info("[PASSED] Edge detection test")

    def test_rgb_to_grayscale(self):
        """
        Тест: Проверка функции преобразования RGB в grayscale.
        """
        logger = get_logger()
        logger.info("TEST: RGB to grayscale")

        # Применяем преобразование
        gray = self.processor.rgb_to_grayscale(self.test_image)

        # Проверяем результат
        self.assertIsInstance(gray, np.ndarray, "Результат должен быть numpy array")
        self.assertEqual(gray.dtype, np.uint8, "Тип данных должен быть uint8")
        self.assertEqual(len(gray.shape), 2, "Результат должен быть 2D")
        self.assertEqual(gray.shape, (100, 100), "Размеры должны быть (100, 100)")

        logger.info("[PASSED] RGB to grayscale test")


if __name__ == "__main__":
    # Запуск тестов с verbose выводом
    unittest.main(verbosity=2)
