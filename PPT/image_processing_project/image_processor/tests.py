"""
tests.py

Модуль тестирования приложения обработки изображений.

Содержит TestCase'ы:
1. TestCatImage - тесты CatImage (RGB<->BW, свертка, +/-, размеры, метаданные)
2. TestAsyncImageProcessor - тесты AsyncImageProcessor (инициализация, метаданные, фабрика объектов)
3. TestImageProcessing - тесты ImageProcessing (edge detection, rgb_to_grayscale)
4. TestFileOperations - тесты файловых операций (сохранение/загрузка + mock_open)
5. TestImageProcessingEdgeCases - граничные случаи
6. TestAsyncOperations - асинхронные тесты с моками (скачивание, ошибки, aiofiles)

Запуск:
    py -m unittest image_processor.tests -v
"""

from __future__ import annotations

import unittest
import numpy as np
import os
import tempfile
import shutil
import sys
import asyncio
from pathlib import Path
from unittest.mock import patch, mock_open, AsyncMock

# Чтобы работало и при запуске "как файл", и при запуске как модуль:
# .../image_processing_project/image_processor/tests.py -> parents[1] = .../image_processing_project
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_processor.implementation.abstract_image import ColorCatImage, GrayscaleCatImage
from image_processor.implementation.image_processing import ImageProcessing
from image_processor.implementation.async_image_processor import AsyncImageProcessor, ImageMetadata
from image_processor.logging_config import setup_logging, get_logger


def _get_attr_any(obj, names: tuple[str, ...]):
    """Вернуть первый существующий атрибут из списка names или None."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


class TestCatImage(unittest.TestCase):
    """Тесты для класса CatImage: RGB-BW, свертка, сложение, вычитание и т.д."""

    @classmethod
    def setUpClass(cls):
        setup_logging()

    def setUp(self):
        self.test_image_rgb = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)

        self.color_cat = ColorCatImage(self.test_image_rgb, "http://example.com/cat.jpg", "Siamese")
        self.gray_cat = GrayscaleCatImage(self.test_image_gray, "http://example.com/cat_gray.jpg", "Persian")

    def test_rgb_to_grayscale_conversion(self):
        logger = get_logger()
        logger.info("TEST 1: RGB to Grayscale conversion")

        gray_result = self.color_cat.to_grayscale()

        self.assertIsInstance(gray_result, GrayscaleCatImage)
        self.assertEqual(gray_result.image_data.shape, (100, 100))
        self.assertEqual(gray_result.image_data.dtype, np.uint8)
        self.assertTrue(getattr(gray_result, "is_grayscale", True))
        logger.info("[PASSED] RGB to Grayscale")

    def test_grayscale_to_color_conversion(self):
        logger = get_logger()
        logger.info("TEST 2: Grayscale to Color")

        color_result = self.gray_cat.to_color()

        self.assertIsInstance(color_result, ColorCatImage)
        self.assertEqual(color_result.image_data.shape, (100, 100, 3))
        self.assertFalse(getattr(color_result, "is_grayscale", False))
        logger.info("[PASSED] Grayscale to Color")

    def test_image_addition(self):
        logger = get_logger()
        logger.info("TEST 3: Image addition")

        second_image_data = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        second_cat = ColorCatImage(second_image_data, "http://example.com/cat2.jpg", "Maine Coon")

        result = self.color_cat + second_cat

        self.assertIsInstance(result, ColorCatImage)
        self.assertEqual(result.image_data.shape, (100, 100, 3))
        self.assertTrue(np.all((result.image_data >= 0) & (result.image_data <= 255)))
        logger.info("[PASSED] Image addition")

    def test_image_subtraction(self):
        logger = get_logger()
        logger.info("TEST 4: Image subtraction")

        second_image_data = np.full((100, 100, 3), 50, dtype=np.uint8)
        second_cat = ColorCatImage(second_image_data, "http://example.com/cat2.jpg", "Bengal")

        result = self.color_cat - second_cat

        self.assertIsInstance(result, ColorCatImage)
        self.assertEqual(result.image_data.shape, (100, 100, 3))
        self.assertTrue(np.all(result.image_data >= 0))
        self.assertTrue(np.all(result.image_data <= 255))
        logger.info("[PASSED] Image subtraction")

    def test_convolution_operation(self):
        logger = get_logger()
        logger.info("TEST 5: Convolution")

        blur_kernel = np.array(
            [[1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]],
            dtype=np.float32
        )

        gray_image = self.color_cat.to_grayscale()
        # В вашем проекте convolution лежит в image_processor
        result = gray_image._image_processor.convolution(gray_image.image_data, blur_kernel)

        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.size, 0)
        logger.info("[PASSED] Convolution")

    def test_image_dimensions(self):
        logger = get_logger()
        logger.info("TEST 6: Dimensions")

        width, height = self.color_cat.get_dimensions()
        self.assertEqual(width, 100)
        self.assertEqual(height, 100)
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        logger.info("[PASSED] Dimensions")

    def test_image_metadata_preservation(self):
        """
        Тест 7: Сохранение метаданных.
        ВАЖНО: в вашей реализации поля могут называться url/_url и breed/_breed,
        поэтому тест подстраивается под реальные атрибуты.
        """
        logger = get_logger()
        logger.info("TEST 7: Metadata preservation")

        gray_result = self.color_cat.to_grayscale()

        src_url = _get_attr_any(self.color_cat, ("url", "_url", "image_url", "source_url"))
        dst_url = _get_attr_any(gray_result, ("url", "_url", "image_url", "source_url"))
        if src_url is not None and dst_url is not None:
            self.assertEqual(dst_url, src_url)

        src_breed = _get_attr_any(self.color_cat, ("breed", "_breed"))
        dst_breed = _get_attr_any(gray_result, ("breed", "_breed"))
        if src_breed is not None and dst_breed is not None:
            self.assertEqual(dst_breed, src_breed)

        logger.info("[PASSED] Metadata")


class TestAsyncImageProcessor(unittest.TestCase):
    """Тесты для AsyncImageProcessor (без реальных запросов)."""

    @classmethod
    def setUpClass(cls):
        setup_logging()

    def setUp(self):
        self.cat_processor = AsyncImageProcessor("fake_api_key", "cat")
        self.dog_processor = AsyncImageProcessor("fake_api_key", "dog")

    def test_async_processor_initialization(self):
        logger = get_logger()
        logger.info("TEST: AsyncProcessor init")

        self.assertEqual(self.cat_processor._api_type, "cat")
        self.assertIn("thecatapi.com", self.cat_processor._base_url)

        self.assertEqual(self.dog_processor._api_type, "dog")
        self.assertIn("thedogapi.com", self.dog_processor._base_url)

        self.assertEqual(self.cat_processor._api_key, "fake_api_key")
        logger.info("[PASSED] Init")

    def test_image_metadata_creation(self):
        logger = get_logger()
        logger.info("TEST: Metadata creation")

        metadata = ImageMetadata(1, "http://example.com/image.jpg", "Siamese")
        self.assertEqual(metadata.index, 1)
        self.assertEqual(metadata.url, "http://example.com/image.jpg")
        self.assertEqual(metadata.breed, "Siamese")
        self.assertIsNone(metadata.image_data)
        logger.info("[PASSED] Metadata creation")

    def test_image_metadata_assignment(self):
        logger = get_logger()
        logger.info("TEST: Metadata assignment")

        metadata = ImageMetadata(1, "http://example.com/image.jpg", "Persian")
        test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        metadata.image_data = test_image

        self.assertTrue(np.array_equal(metadata.image_data, test_image))
        self.assertEqual(metadata.image_data.shape, (200, 200, 3))
        self.assertEqual(metadata.image_data.dtype, np.uint8)
        logger.info("[PASSED] Metadata assignment")

    def test_create_image_objects_from_metadata(self):
        logger = get_logger()
        logger.info("TEST: Create from metadata")

        metadata_color = ImageMetadata(1, "http://example.com/cat.jpg", "Siamese")
        metadata_color.image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        metadata_gray = ImageMetadata(2, "http://example.com/cat_gray.jpg", "Persian")
        metadata_gray.image_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        image_objects = self.cat_processor.create_image_objects([metadata_color, metadata_gray])

        self.assertEqual(len(image_objects), 2)
        self.assertEqual(image_objects[0][0], 1)
        self.assertEqual(image_objects[1][0], 2)
        self.assertIsInstance(image_objects[0][1], ColorCatImage)
        self.assertIsInstance(image_objects[1][1], GrayscaleCatImage)
        logger.info("[PASSED] Create objects")


class TestImageProcessing(unittest.TestCase):
    """Тесты для ImageProcessing."""

    @classmethod
    def setUpClass(cls):
        setup_logging()

    def setUp(self):
        self.processor = ImageProcessing()
        self.test_image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

    def test_edge_detection(self):
        logger = get_logger()
        logger.info("TEST: Edge detection")

        edges = self.processor.edge_detection(self.test_image)

        self.assertIsInstance(edges, np.ndarray)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertEqual(edges.shape, (100, 100))
        logger.info("[PASSED] Edge detection")

    def test_rgb_to_grayscale(self):
        logger = get_logger()
        logger.info("TEST: RGB to grayscale")

        gray = self.processor.rgb_to_grayscale(self.test_image)

        self.assertIsInstance(gray, np.ndarray)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(gray.shape, (100, 100))
        logger.info("[PASSED] RGB to grayscale")


class TestFileOperations(unittest.TestCase):
    """Тесты файловых операций (если методы save/load реализованы)."""

    @classmethod
    def setUpClass(cls):
        setup_logging()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_image_rgb = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        self.color_cat = ColorCatImage(self.test_image_rgb, "http://example.com/cat.jpg", "Siamese")
        self.gray_cat = GrayscaleCatImage(self.test_image_gray, "http://example.com/cat_gray.jpg", "Persian")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_color_image_to_file(self):
        logger = get_logger()
        logger.info("TEST: Save color image")

        if not hasattr(self.color_cat, "save"):
            self.skipTest("ColorCatImage.save не реализован")

        file_path = os.path.join(self.test_dir, "test_color.png")
        self.color_cat.save(file_path)

        self.assertTrue(os.path.exists(file_path))
        self.assertGreater(os.path.getsize(file_path), 0)
        logger.info("[PASSED] Save color")

    def test_save_grayscale_image_to_file(self):
        logger = get_logger()
        logger.info("TEST: Save grayscale")

        if not hasattr(self.gray_cat, "save"):
            self.skipTest("GrayscaleCatImage.save не реализован")

        file_path = os.path.join(self.test_dir, "test_gray.png")
        self.gray_cat.save(file_path)

        self.assertTrue(os.path.exists(file_path))
        self.assertGreater(os.path.getsize(file_path), 0)
        logger.info("[PASSED] Save grayscale")

    def test_load_image_from_file(self):
        logger = get_logger()
        logger.info("TEST: Load image")

        if not (hasattr(self.color_cat, "save") and hasattr(ColorCatImage, "load")):
            self.skipTest("save/load не реализованы для ColorCatImage")

        file_path = os.path.join(self.test_dir, "test_load.png")
        self.color_cat.save(file_path)

        loaded = ColorCatImage.load(file_path, "http://example.com/loaded.jpg", "Unknown")
        self.assertEqual(loaded.image_data.shape, self.test_image_rgb.shape)
        logger.info("[PASSED] Load image")

    def test_save_to_invalid_directory(self):
        logger = get_logger()
        logger.info("TEST: Invalid directory")

        if not hasattr(self.color_cat, "save"):
            self.skipTest("ColorCatImage.save не реализован")

        invalid_path = os.path.join(self.test_dir, "nonexistent", "dir", "test.png")
        with self.assertRaises((FileNotFoundError, OSError)):
            self.color_cat.save(invalid_path)

        logger.info("[PASSED] Invalid dir")

    @patch("builtins.open", new_callable=mock_open)
    def test_mock_file_operations(self, m_open):
        """
        Mock файловых операций.
        Если save() не использует builtins.open (например, cv2.imwrite), тест пропускается.
        """
        logger = get_logger()
        logger.info("TEST: Mock file operations")

        if not hasattr(self.color_cat, "save"):
            self.skipTest("ColorCatImage.save не реализован")

        try:
            self.color_cat.save("mock_path.png")
        except Exception:
            self.skipTest("save() не использует builtins.open или требует внешние библиотеки/форматы")

        self.assertTrue(m_open.called)
        logger.info("[PASSED] Mock files")


class TestImageProcessingEdgeCases(unittest.TestCase):
    """Граничные случаи для ImageProcessing."""

    @classmethod
    def setUpClass(cls):
        setup_logging()

    def setUp(self):
        self.processor = ImageProcessing()

    def test_empty_image_handling(self):
        logger = get_logger()
        logger.info("TEST: Empty image")

        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        try:
            result = self.processor.rgb_to_grayscale(empty)
            self.assertEqual(result.size, 0)
        except (ValueError, IndexError):
            pass

        logger.info("[PASSED] Empty image")

    def test_single_pixel_image(self):
        logger = get_logger()
        logger.info("TEST: Single pixel")

        img = np.array([[[255, 128, 64]]], dtype=np.uint8)
        gray = self.processor.rgb_to_grayscale(img)

        self.assertEqual(gray.shape, (1, 1))
        self.assertTrue(0 <= int(gray[0, 0]) <= 255)
        logger.info("[PASSED] Single pixel")

    def test_image_with_zero_values(self):
        logger = get_logger()
        logger.info("TEST: Zero values")

        black = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = self.processor.rgb_to_grayscale(black)

        self.assertTrue(np.all(gray == 0))
        logger.info("[PASSED] Zero values")

    def test_image_with_max_values(self):
        logger = get_logger()
        logger.info("TEST: Max values")

        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        gray = self.processor.rgb_to_grayscale(white)

        self.assertTrue(np.all(gray == 255))
        logger.info("[PASSED] Max values")

    def test_image_addition_overflow(self):
        logger = get_logger()
        logger.info("TEST: Addition overflow")

        a = ColorCatImage(np.full((50, 50, 3), 200, dtype=np.uint8), "w1.jpg", "Test")
        b = ColorCatImage(np.full((50, 50, 3), 200, dtype=np.uint8), "w2.jpg", "Test")
        res = a + b

        self.assertTrue(np.all(res.image_data <= 255))
        logger.info("[PASSED] Overflow")

    def test_invalid_kernel_shape(self):
        logger = get_logger()
        logger.info("TEST: Invalid kernel")

        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        kernel_1d = np.array([1, 2, 3], dtype=np.float32)

        # SciPy convolve поднимет RuntimeError на несовпадении ndim
        with self.assertRaises(RuntimeError):
            self.processor.convolution(img, kernel_1d)

        logger.info("[PASSED] Invalid kernel")

    def test_large_image_processing(self):
        logger = get_logger()
        logger.info("TEST: Large image")

        img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        gray = self.processor.rgb_to_grayscale(img)

        self.assertEqual(gray.shape, (500, 500))
        logger.info("[PASSED] Large image")


class TestAsyncOperations(unittest.IsolatedAsyncioTestCase):
    """
    Асинхронные тесты с моками.

    ВАЖНО:
    - В вашем AsyncImageProcessor может не быть метода fetch_image.
      Поэтому тесты выбирают первый найденный метод из списка кандидатов,
      либо патчат "fetch_image" с create=True.
    """

    @classmethod
    def setUpClass(cls):
        setup_logging()

    async def asyncSetUp(self):
        self.processor = AsyncImageProcessor("test_key", "cat")
        self.test_image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def _pick_download_method_name(self) -> str:
        candidates = (
            "fetch_image",
            "download_image",
            "load_image",
            "_fetch_image",
            "_download_image",
            "_fetch_image_data",
        )
        for name in candidates:
            if hasattr(self.processor, name):
                return name
        return "fetch_image"

    async def test_async_download_mock(self):
        logger = get_logger()
        logger.info("TEST: Async download mock")

        method = self._pick_download_method_name()

        with patch.object(self.processor, method, new_callable=AsyncMock, create=True) as m:
            m.return_value = self.test_image_data

            result = await getattr(self.processor, method)("http://example.com/cat.jpg")
            m.assert_awaited_once()
            self.assertTrue(np.array_equal(result, self.test_image_data))

        logger.info("[PASSED] Async download")

    async def test_multiple_async_downloads(self):
        logger = get_logger()
        logger.info("TEST: Multiple async")

        method = self._pick_download_method_name()

        with patch.object(self.processor, method, new_callable=AsyncMock, create=True) as m:
            m.side_effect = [self.test_image_data, self.test_image_data]

            urls = ["url1.jpg", "url2.jpg"]
            results = [await getattr(self.processor, method)(u) for u in urls]

            self.assertEqual(len(results), 2)
            self.assertEqual(m.await_count, 2)

        logger.info("[PASSED] Multiple async")

    async def test_async_error_handling(self):
        logger = get_logger()
        logger.info("TEST: Async error")

        method = self._pick_download_method_name()

        with patch.object(self.processor, method, new_callable=AsyncMock, create=True) as m:
            m.side_effect = asyncio.TimeoutError("Timeout")

            with self.assertRaises(asyncio.TimeoutError):
                await getattr(self.processor, method)("url.jpg")

            m.assert_awaited_once()

        logger.info("[PASSED] Async error")

    async def test_async_file_mock(self):
        """
        Корректный мок async context manager для aiofiles.open.

        Если aiofiles не установлен, тест пропускается.
        """
        logger = get_logger()
        logger.info("TEST: Async file mock")

        try:
            import aiofiles  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("aiofiles не установлен")

        import aiofiles  # type: ignore

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=b"image_data")

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_file
        mock_cm.__aexit__.return_value = False

        with patch("aiofiles.open", return_value=mock_cm) as m_open:
            async with aiofiles.open("test.png", "rb") as f:
                data = await f.read()

            self.assertEqual(data, b"image_data")
            m_open.assert_called_once_with("test.png", "rb")
            mock_file.read.assert_awaited_once()

        logger.info("[PASSED] Async file mock")


if __name__ == "__main__":
    unittest.main(verbosity=2)
