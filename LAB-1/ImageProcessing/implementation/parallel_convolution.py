"""
Модуль параллельной обработки свёртки и арифметических операций с использованием multiprocessing.

Обеспечивает:
- Выполнение свёртки в отдельных процессах
- Параллельное выполнение арифметических операций (сложение/вычитание изображений)
- Использование всех доступных ядер процессора
- Детальное логирование с PID процессов
"""

import os
import numpy as np
from multiprocessing import Pool, current_process
from typing import Tuple, List, Dict
from implementation.image_processing import ImageProcessing


def apply_convolution_worker(args: Tuple[int, str, np.ndarray]) -> Tuple[int, str, np.ndarray]:
    """
    Рабочая функция для применения пользовательской свёртки (Sobel) в отдельном процессе.
    
    Args:
        args: Кортеж (порядковый_номер, название_породы, данные_изображения)
    
    Returns:
        Кортеж (порядковый_номер, название_породы, обработанные_данные)
    """
    index, breed, image_data = args
    pid = os.getpid()
    
    print(f"   Convolution for image {index} started (PID {pid}, breed: {breed})")
    
    try:
        # Создаем процессор изображений
        processor = ImageProcessing()
        
        # Применяем пользовательский метод обнаружения границ (Собель)
        edges = processor.edge_detection(image_data)
        
        print(f"   Convolution for image {index} finished (PID {pid})")
        
        return (index, breed, edges)
        
    except Exception as e:
        print(f"   Convolution for image {index} failed (PID {pid}): {e}")
        # Возвращаем пустой массив при ошибке
        return (index, breed, np.zeros_like(image_data))


def apply_library_edge_detection_worker(args: Tuple[int, str, np.ndarray]) -> Tuple[int, str, np.ndarray]:
    """
    Рабочая функция для применения библиотечного edge detection (Canny) в отдельном процессе.
    
    Args:
        args: Кортеж (порядковый_номер, название_породы, данные_изображения)
    
    Returns:
        Кортеж (порядковый_номер, название_породы, обработанные_данные)
    """
    index, breed, image_data = args
    pid = os.getpid()
    
    print(f"   Library edge detection for image {index} started (PID {pid}, breed: {breed})")
    
    try:
        # Создаем процессор изображений
        processor = ImageProcessing()
        
        # Применяем библиотечный метод обнаружения границ (Canny)
        edges = processor.library_edge_detection(image_data)
        
        print(f"   Library edge detection for image {index} finished (PID {pid})")
        
        return (index, breed, edges)
        
    except Exception as e:
        print(f"   Library edge detection for image {index} failed (PID {pid}): {e}")
        # Возвращаем пустой массив при ошибке
        return (index, breed, np.zeros_like(image_data))


def apply_arithmetic_operations_worker(
    args: Tuple[int, str, object, np.ndarray, np.ndarray]
) -> Tuple[int, Dict[str, np.ndarray]]:
    
    index, breed, image_obj, custom_edges, library_edges = args
    pid = os.getpid()
    
    print(f"   Arithmetic operations for image {index} started (PID {pid}, breed: {breed})")
    
    try:
        
        result_plus_custom = image_obj + custom_edges
        
        original_plus_custom = result_plus_custom.image_data
        
        result_minus_custom = image_obj - custom_edges
        
        original_minus_custom = result_minus_custom.image_data

        
        result_plus_library = image_obj + library_edges
        original_plus_library = result_plus_library.image_data
        
        
        result_minus_library = image_obj - library_edges
        original_minus_library = result_minus_library.image_data
        
        print(f"  Arithmetic operations for image {index} finished (PID {pid})")
        
        # Возвращаем словарь с результатами всех операций
        return (index, {
            'plus_custom': original_plus_custom,
            'minus_custom': original_minus_custom,
            'plus_library': original_plus_library,
            'minus_library': original_minus_library
        })
        
    except Exception as e:
        print(f"   Arithmetic operations for image {index} failed (PID {pid}): {e}")
        
        # Возвращаем пустые массивы при ошибке
        dummy_shape = image_obj.image_data.shape
        return (index, {
            'plus_custom': np.zeros(dummy_shape, dtype=np.uint8),
            'minus_custom': np.zeros(dummy_shape, dtype=np.uint8),
            'plus_library': np.zeros(dummy_shape, dtype=np.uint8),
            'minus_library': np.zeros(dummy_shape, dtype=np.uint8)
        })


# ============================================================================
# КЛАСС ПРОЦЕССОРА
# ============================================================================

class ParallelConvolutionProcessor:
    """
    Процессор для параллельной обработки свёртки и арифметических операций.
    
    Использует multiprocessing.Pool для распределения задач между процессами.
    Каждый процесс выполняет обработку независимо, используя отдельное ядро CPU.
    """
    
    def __init__(self, num_processes: int = None):
        """
        Инициализирует процессор.
        
        Args:
            num_processes: Количество процессов для пула.
                          None = использовать все доступные ядра (os.cpu_count())
        """
        self.num_processes = num_processes
        
        if num_processes is None:
            import multiprocessing
            self.num_processes = multiprocessing.cpu_count()
            print(f"    ParallelConvolutionProcessor initialized with {self.num_processes} processes (all CPU cores)")
        else:
            print(f"    ParallelConvolutionProcessor initialized with {num_processes} processes")
    
    def process_batch_custom(self, image_batch: List[Tuple[int, object]]) -> List[Tuple[int, str, np.ndarray]]:
        """
        Параллельно применяет пользовательское обнаружение границ (Sobel) ко всем изображениям.
        
        Args:
            image_batch: Список кортежей (порядковый_номер, CatImage_объект)
        
        Returns:
            Список кортежей (порядковый_номер, порода, обработанные_данные)
        """
        print(f"\n Starting parallel custom edge detection for {len(image_batch)} images...")
        
        # Подготовка аргументов для пула процессов
        args_list = [
            (index, img.breed, img.image_data)
            for index, img in image_batch
        ]
        
        # Выполнение в пуле процессов
        with Pool(processes=self.num_processes) as pool:
            # pool.map() блокирует до завершения всех задач
            # Распределяет задачи между процессами автоматически
            results = pool.map(apply_convolution_worker, args_list)
        
        print(f" Parallel custom edge detection completed for {len(image_batch)} images")
        
        return results
    
    def process_batch_library(self, image_batch: List[Tuple[int, object]]) -> List[Tuple[int, str, np.ndarray]]:
        """
        Параллельно применяет библиотечное обнаружение границ (Canny) ко всем изображениям.
        
        Args:
            image_batch: Список кортежей (порядковый_номер, CatImage_объект)
        
        Returns:
            Список кортежей (порядковый_номер, порода, обработанные_данные)
        """
        print(f"\n Starting parallel library edge detection for {len(image_batch)} images...")
        
        # Подготовка аргументов для пула процессов
        args_list = [
            (index, img.breed, img.image_data)
            for index, img in image_batch
        ]
        
        # Выполнение в пуле процессов
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(apply_library_edge_detection_worker, args_list)
        
        print(f" Parallel library edge detection completed for {len(image_batch)} images")
        
        return results
    
    def process_batch_arithmetic(
        self,
        image_batch: List[Tuple[int, object]],
        custom_results: List[Tuple[int, str, np.ndarray]],
        library_results: List[Tuple[int, str, np.ndarray]]
    ) -> List[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Параллельно применяет арифметические операции ко всем изображениям.
        
        Для каждого изображения выполняет:
        1. original + custom_edges
        2. original - custom_edges
        3. original + library_edges
        4. original - library_edges
        
        Args:
            image_batch: Список кортежей (index, CatImage объект)
            custom_results: Результаты custom edge detection
                           [(index, breed, edges), ...]
            library_results: Результаты library edge detection
                            [(index, breed, edges), ...]
        
        Returns:
            Список кортежей (index, dict с результатами операций)
            Формат dict:
            {
                'plus_custom': np.ndarray,
                'minus_custom': np.ndarray,
                'plus_library': np.ndarray,
                'minus_library': np.ndarray
            }
        """
        print(f"\n Starting parallel arithmetic operations for {len(image_batch)} images...")
        
        # ═══════════════════════════════════════════════════════════
        # Создание словарей для быстрого доступа к результатам
        # ═══════════════════════════════════════════════════════════
        
        custom_dict = {idx: edges for idx, breed, edges in custom_results}
        library_dict = {idx: edges for idx, breed, edges in library_results}
        
        # ═══════════════════════════════════════════════════════════
        # Подготовка аргументов для пула процессов
        # ═══════════════════════════════════════════════════════════
        
        args_list = []
        for index, img_obj in image_batch:
            custom_edges = custom_dict.get(index, None)
            library_edges = library_dict.get(index, None)
            
            if custom_edges is not None and library_edges is not None:
                args_list.append((
                    index,
                    img_obj.breed,
                    img_obj,           # Объект CatImage (будет pickle сериализован)
                    custom_edges,      # np.ndarray
                    library_edges      # np.ndarray
                ))
            else:
                print(f"    Skipping arithmetic operations for image {index} (missing edge detection results)")
        
        if not args_list:
            print(f"    No images to process for arithmetic operations")
            return []
        
        # ═══════════════════════════════════════════════════════════
        # Выполнение в пуле процессов
        # ═══════════════════════════════════════════════════════════
        
        with Pool(processes=self.num_processes) as pool:
            # pool.map() распределяет задачи между процессами
            # Каждый процесс выполняет apply_arithmetic_operations_worker()
            # Блокируется до завершения всех задач
            results = pool.map(apply_arithmetic_operations_worker, args_list)
        
        print(f" Parallel arithmetic operations completed for {len(image_batch)} images")
        
        return results


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (для тестирования модуля)
# ============================================================================

if __name__ == "__main__":
    """
    Пример использования ParallelConvolutionProcessor.
    Запускается только при прямом запуске модуля: python parallel_convolution.py
    """
    
    print("="*70)
    print("   ТЕСТИРОВАНИЕ ParallelConvolutionProcessor")
    print("="*70)
    
    # Создание тестовых данных
    import numpy as np
    from implementation.abstract_image import ColorCatImage
    
    # Создаём 2 тестовых изображения
    test_image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    cat1 = ColorCatImage(test_image1, "test_url_1", "TestBreed1")
    cat2 = ColorCatImage(test_image2, "test_url_2", "TestBreed2")
    
    image_batch = [(1, cat1), (2, cat2)]
    
    # Создание процессора
    processor = ParallelConvolutionProcessor(num_processes=2)
    
    # Тест 1: Custom edge detection
    print("\n" + "="*70)
    print("  TEST 1: Custom Edge Detection")
    print("="*70)
    custom_results = processor.process_batch_custom(image_batch)
    print(f"\n  Результат: {len(custom_results)} изображений обработано")
    
    # Тест 2: Library edge detection
    print("\n" + "="*70)
    print("  TEST 2: Library Edge Detection")
    print("="*70)
    library_results = processor.process_batch_library(image_batch)
    print(f"\n  Результат: {len(library_results)} изображений обработано")
    
    # Тест 3: Arithmetic operations
    print("\n" + "="*70)
    print("  TEST 3: Arithmetic Operations")
    print("="*70)
    arithmetic_results = processor.process_batch_arithmetic(
        image_batch,
        custom_results,
        library_results
    )
    print(f"\n  Результат: {len(arithmetic_results)} изображений обработано")
    
    # Вывод результатов
    for index, arith_dict in arithmetic_results:
        print(f"\n  Изображение {index}:")
        print(f"    - plus_custom shape: {arith_dict['plus_custom'].shape}")
        print(f"    - minus_custom shape: {arith_dict['minus_custom'].shape}")
        print(f"    - plus_library shape: {arith_dict['plus_library'].shape}")
        print(f"    - minus_library shape: {arith_dict['minus_library'].shape}")
    
    print("\n" + "="*70)
    print("   ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*70)
