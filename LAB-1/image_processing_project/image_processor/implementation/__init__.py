"""
Пакет реализации обработки изображений.

Содержит:
- abstract_image.py: абстрактные классы для работы с изображениями
- image_processing.py: реализация методов обработки
- i_image_processing.py: интерфейсы
- async_image_processor.py: асинхронная обработка
- async_pipeline.py: асинхронный пайплайн
- parallel_convolution.py: параллельная свертка
"""

from .abstract_image import CatImage, ColorCatImage, GrayscaleCatImage
from .image_processing import ImageProcessing
from .async_image_processor import AsyncImageProcessor, ImageMetadata
from .async_pipeline import AsyncImagePipeline, ProcessedImageData
from .parallel_convolution import ParallelConvolutionProcessor

__all__ = [
    'CatImage',
    'ColorCatImage',
    'GrayscaleCatImage',
    'ImageProcessing',
    'AsyncImageProcessor',
    'ImageMetadata',
    'AsyncImagePipeline',
    'ProcessedImageData',
    'ParallelConvolutionProcessor'
]