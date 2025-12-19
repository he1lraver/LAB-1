"""
Модуль параллельной обработки свёртки и арифметических операций с логированием.

Обеспечивает:
- Выполнение свёртки в отдельных процессах
- Параллельное выполнение арифметических операций
- Использование всех доступных ядер процессора
- Детальное логирование с PID процессов
"""

import os
import asyncio
import numpy as np
from multiprocessing import Pool, current_process
from typing import Tuple, List, Dict
from .image_processing import ImageProcessing
from image_processor.logging_config import get_logger


def apply_convolution_worker(args: Tuple[int, str, np.ndarray]) -> Tuple[int, str, np.ndarray]:
    """
    Рабочая функция для применения пользовательской свёртки (Sobel) в отдельном процессе.
    """
    index, breed, image_data = args
    pid = os.getpid()
    logger = get_logger()

    logger.debug(f"Convolution worker started for image {index} (PID {pid}, breed: {breed})")

    try:
        processor = ImageProcessing()
        edges = processor.edge_detection(image_data)
        logger.debug(f"Convolution worker finished for image {index} (PID {pid})")
        return (index, breed, edges)
    except Exception as e:
        logger.error(f"Convolution worker failed for image {index} (PID {pid}): {e}")
        return (index, breed, np.zeros_like(image_data))


def apply_library_edge_detection_worker(args: Tuple[int, str, np.ndarray]) -> Tuple[int, str, np.ndarray]:
    """
    Рабочая функция для применения библиотечного edge detection (Canny) в отдельном процессе.
    """
    index, breed, image_data = args
    pid = os.getpid()
    logger = get_logger()

    logger.debug(f"Library edge detection worker started for image {index} (PID {pid}, breed: {breed})")

    try:
        processor = ImageProcessing()
        edges = processor.library_edge_detection(image_data)
        logger.debug(f"Library edge detection worker finished for image {index} (PID {pid})")
        return (index, breed, edges)
    except Exception as e:
        logger.error(f"Library edge detection worker failed for image {index} (PID {pid}): {e}")
        return (index, breed, np.zeros_like(image_data))


def apply_arithmetic_operations_worker(
    args: Tuple[int, str, object, np.ndarray, np.ndarray]
) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Рабочая функция для применения арифметических операций в отдельном процессе.
    """
    index, breed, image_obj, custom_edges, library_edges = args
    pid = os.getpid()
    logger = get_logger()

    logger.debug(f"Arithmetic operations worker started for image {index} (PID {pid}, breed: {breed})")

    try:
        result_plus_custom = image_obj + custom_edges
        original_plus_custom = result_plus_custom.image_data

        result_minus_custom = image_obj - custom_edges
        original_minus_custom = result_minus_custom.image_data

        result_plus_library = image_obj + library_edges
        original_plus_library = result_plus_library.image_data

        result_minus_library = image_obj - library_edges
        original_minus_library = result_minus_library.image_data

        logger.debug(f"Arithmetic operations worker finished for image {index} (PID {pid})")

        return (index, {
            'plus_custom': original_plus_custom,
            'minus_custom': original_minus_custom,
            'plus_library': original_plus_library,
            'minus_library': original_minus_library
        })
    except Exception as e:
        logger.error(f"Arithmetic operations worker failed for image {index} (PID {pid}): {e}")
        dummy_shape = image_obj.image_data.shape
        return (index, {
            'plus_custom': np.zeros(dummy_shape, dtype=np.uint8),
            'minus_custom': np.zeros(dummy_shape, dtype=np.uint8),
            'plus_library': np.zeros(dummy_shape, dtype=np.uint8),
            'minus_library': np.zeros(dummy_shape, dtype=np.uint8)
        })


class ParallelConvolutionProcessor:
    """
    Процессор для параллельной обработки свёртки с логированием.
    """

    def __init__(self, num_processes: int = None):
        """
        Инициализирует процессор с логированием.
        """
        self.logger = get_logger()
        self.num_processes = num_processes

        if num_processes is None:
            import multiprocessing
            self.num_processes = multiprocessing.cpu_count()
            self.logger.info(f"ParallelConvolutionProcessor initialized with {self.num_processes} processes (all CPU cores)")
        else:
            self.logger.info(f"ParallelConvolutionProcessor initialized with {num_processes} processes")

    def process_batch_custom(self, image_batch: List[Tuple[int, object]]) -> List[Tuple[int, str, np.ndarray]]:
        """
        Параллельно применяет пользовательское обнаружение границ (Sobel).
        """
        self.logger.info(f"Starting parallel custom edge detection for {len(image_batch)} images")

        args_list = [
            (index, img.breed, img.image_data)
            for index, img in image_batch
        ]

        with Pool(processes=self.num_processes) as pool:
            results = pool.map(apply_convolution_worker, args_list)

        self.logger.info(f"Parallel custom edge detection completed for {len(image_batch)} images")
        return results

    def process_batch_library(self, image_batch: List[Tuple[int, object]]) -> List[Tuple[int, str, np.ndarray]]:
        """
        Параллельно применяет библиотечное обнаружение границ (Canny).
        """
        self.logger.info(f"Starting parallel library edge detection for {len(image_batch)} images")

        args_list = [
            (index, img.breed, img.image_data)
            for index, img in image_batch
        ]

        with Pool(processes=self.num_processes) as pool:
            results = pool.map(apply_library_edge_detection_worker, args_list)

        self.logger.info(f"Parallel library edge detection completed for {len(image_batch)} images")
        return results

    def process_batch_arithmetic(
        self,
        image_batch: List[Tuple[int, object]],
        custom_results: List[Tuple[int, str, np.ndarray]],
        library_results: List[Tuple[int, str, np.ndarray]]
    ) -> List[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Параллельно применяет арифметические операции.
        """
        self.logger.info(f"Starting parallel arithmetic operations for {len(image_batch)} images")

        custom_dict = {idx: edges for idx, breed, edges in custom_results}
        library_dict = {idx: edges for idx, breed, edges in library_results}

        args_list = []
        for index, img_obj in image_batch:
            custom_edges = custom_dict.get(index, None)
            library_edges = library_dict.get(index, None)
            if custom_edges is not None and library_edges is not None:
                args_list.append((
                    index,
                    img_obj.breed,
                    img_obj,
                    custom_edges,
                    library_edges
                ))
            else:
                self.logger.warning(f"Skipping arithmetic operations for image {index} (missing edge detection results)")

        if not args_list:
            self.logger.warning("No images to process for arithmetic operations")
            return []

        with Pool(processes=self.num_processes) as pool:
            results = pool.map(apply_arithmetic_operations_worker, args_list)

        self.logger.info(f"Parallel arithmetic operations completed for {len(image_batch)} images")
        return results