"""
Пакет image_processor - полная система обработки изображений.

Содержит:
- abstract_image.py: абстрактные классы для работы с изображениями (CatImage, ColorCatImage, GrayscaleCatImage)
- image_processing.py: реализация методов обработки (ImageProcessing)
- i_image_processing.py: интерфейсы (IImageProcessing)
- async_image_processor.py: асинхронная загрузка изображений (AsyncImageProcessor)
- async_pipeline.py: асинхронный пайплайн обработки (AsyncImagePipeline)
- parallel_convolution.py: параллельная обработка сверток (ParallelConvolutionProcessor)

ЛОГИРОВАНИЕ:
- logging_config.py: настройка логирования (setup_logging, get_logger)
- Все компоненты используют единый логгер "image_processor"
- DEBUG логи в app.log, INFO логи в консоль

Использование:
    from image_processor import ColorCatImage, ImageProcessing, AsyncImageProcessor
    from image_processor.logging_config import get_logger
    
    logger = get_logger()
    logger.info("Application started")
"""

# Импорт основных классов из implementation
from .implementation.abstract_image import CatImage, ColorCatImage, GrayscaleCatImage
from .implementation.image_processing import ImageProcessing
from .implementation.async_image_processor import AsyncImageProcessor, ImageMetadata
from .implementation.async_pipeline import AsyncImagePipeline, ProcessedImageData
from .implementation.parallel_convolution import ParallelConvolutionProcessor

# Импорт логирования
from .logging_config import setup_logging, get_logger

# Public API
__all__ = [
    # Основные классы
    'CatImage',
    'ColorCatImage',
    'GrayscaleCatImage',
    'ImageProcessing',
    'AsyncImageProcessor',
    'ImageMetadata',
    'AsyncImagePipeline',
    'ProcessedImageData',
    'ParallelConvolutionProcessor',
    # Логирование
    'setup_logging',
    'get_logger',
]

__version__ = "1.0.0"
__author__ = "Image Processing Team"