"""
Асинхронный генераторный пайплайн для обработки изображений с chunked загрузкой.

Реализует:
- Chunked обработка (можно задать размер чанка)
- Три этапа обработки: скачивание, свёртка, сохранение
- Каждый этап - отдельный асинхронный генератор
- Этапы работают параллельно и не блокируют друг друга
- Порядковые номера сохраняются на всех этапах
- Параллельная обработка custom и library edge detection
"""

import os
import asyncio
import aiofiles
import numpy as np
from typing import AsyncGenerator, Tuple, List
from PIL import Image

from implementation.async_image_processor import AsyncImageProcessor, ImageMetadata
from implementation.parallel_convolution import ParallelConvolutionProcessor


class ProcessedImageData:
    """Данные обработанного изображения со всеми результатами."""
    
    def __init__(self, index: int, breed: str):
        self.index = index
        self.breed = breed
        
        # Оригинальные данные
        self.original_data: np.ndarray = None
        
        # Edge detection результаты
        self.custom_edges: np.ndarray = None
        self.library_edges: np.ndarray = None
        
        # ✨ НОВОЕ: Арифметические операции
        self.original_plus_custom: np.ndarray = None    # original + custom_edges
        self.original_minus_custom: np.ndarray = None   # original - custom_edges
        self.original_plus_library: np.ndarray = None   # original + library_edges
        self.original_minus_library: np.ndarray = None  # original - library_edges


class AsyncImagePipeline:
    """
    Асинхронный генераторный пайплайн для обработки изображений с поддержкой chunked загрузки.
    
    Этапы пайплайна:
    1. Скачивание изображений (асинхронно, по чанкам)
    2. Применение свёртки (параллельно в процессах: custom + library одновременно)
    3. Сохранение результатов (асинхронно: 3 файла на изображение)
    """
    
    def __init__(self, api_key: str, api_type: str, output_dir: str, chunk_size: int = 5):
        """
        Args:
            api_key: API ключ
            api_type: Тип API ('cat' или 'dog')
            output_dir: Директория для сохранения результатов
            chunk_size: Размер чанка для параллельной обработки
        """
        self.api_key = api_key
        self.api_type = api_type
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.processor = AsyncImageProcessor(api_key, api_type)
        self.convolution_processor = ParallelConvolutionProcessor()
    
    async def stage_1_download(self, limit: int) -> AsyncGenerator[List[Tuple[int, object]], None]:
        """
        Этап 1: Асинхронное скачивание изображений по чанкам.
        
        Args:
            limit: Количество изображений для скачивания
            
        Yields:
            Списки кортежей (порядковый_номер, объект_изображения) размером chunk_size
        """
        print(f"\n Chunk size set to: {self.chunk_size} images per batch")
        
        # Получаем список URL с порядковыми номерами
        metadata_list = await self.processor.fetch_image_urls(limit)
        
        # Разбиваем на чанки
        total_chunks = (len(metadata_list) + self.chunk_size - 1) // self.chunk_size
        print(f" Total {len(metadata_list)} images will be processed in {total_chunks} chunk(s)\n")
        
        for chunk_idx in range(0, len(metadata_list), self.chunk_size):
            chunk_metadata = metadata_list[chunk_idx:chunk_idx + self.chunk_size]
            chunk_num = (chunk_idx // self.chunk_size) + 1
            
            print(f"\n{'='*70}")
            print(f"   CHUNK {chunk_num}/{total_chunks}: Processing images {chunk_idx + 1} to {min(chunk_idx + self.chunk_size, len(metadata_list))}")
            print(f"{'='*70}\n")
            
            # Асинхронно загружаем текущий чанк
            downloaded_metadata = await self.processor.download_images(chunk_metadata)
            
            # Создаем объекты изображений
            image_objects = self.processor.create_image_objects(downloaded_metadata)
            
            # Передаем чанк дальше по пайплайну
            if image_objects:
                yield image_objects
    
    async def stage_2_convolution(
        self,
        download_generator: AsyncGenerator[List[Tuple[int, object]], None]
    ) -> AsyncGenerator[List[ProcessedImageData], None]:
        """
        Этап 2: Параллельное применение свёртки и арифметических операций.
        
        Выполняет:
        1. Custom edge detection (параллельно)
        2. Library edge detection (параллельно)
        3. Арифметические операции: сложение и вычитание (параллельно)
        
        Всё выполняется одновременно в разных потоках/процессах.
        """
        print("\n" + "="*70)
        print("   STAGE 2: CONVOLUTION & ARITHMETIC OPERATIONS")
        print("="*70)
        
        async for image_batch in download_generator:
            if not image_batch:
                continue
            
            chunk_size = len(image_batch)
            print(f"\n Processing chunk of {chunk_size} images...")
            
            print("  Starting parallel edge detection...")
            
            # Запускаем custom и library edge detection параллельно
            custom_task = asyncio.to_thread(
                self.convolution_processor.process_batch_custom,
                image_batch
            )
            
            library_task = asyncio.to_thread(
                self.convolution_processor.process_batch_library,
                image_batch
            )
            
            # Ждём завершения обоих
            custom_results, library_results = await asyncio.gather(
                custom_task, 
                library_task
            )
            
            print("   Edge detection completed")
            
            
            print("   Starting parallel arithmetic operations...")
            
            # Запускаем все 4 операции параллельно
            arithmetic_task = asyncio.to_thread(
                self.convolution_processor.process_batch_arithmetic,
                image_batch,
                custom_results,
                library_results
            )
            
            # Ждём завершения арифметических операций
            arithmetic_results = await arithmetic_task
            
            print("   Arithmetic operations completed")
            
            
            # Создаем словари для быстрого доступа
            custom_dict = {idx: (breed, edges) for idx, breed, edges in custom_results}
            library_dict = {idx: (breed, edges) for idx, breed, edges in library_results}
            arithmetic_dict = {idx: results for idx, results in arithmetic_results}
            
            # Формируем финальные данные
            processed_batch = []
            
            for index, image_obj in image_batch:
                processed = ProcessedImageData(index, image_obj.breed)
                
                # Оригинальное изображение
                processed.original_data = image_obj.image_data
                
                # Edge detection результаты
                if index in custom_dict:
                    processed.custom_edges = custom_dict[index][1]
                
                if index in library_dict:
                    processed.library_edges = library_dict[index][1]
                
                # ✨ НОВОЕ: Арифметические операции
                if index in arithmetic_dict:
                    arith = arithmetic_dict[index]
                    processed.original_plus_custom = arith['plus_custom']
                    processed.original_minus_custom = arith['minus_custom']
                    processed.original_plus_library = arith['plus_library']
                    processed.original_minus_library = arith['minus_library']
                
                processed_batch.append(processed)
            
            print(f"   Chunk processed: {chunk_size} images with 7 versions each")
            
            yield processed_batch
    
    async def stage_3_save(
        self, 
        convolution_generator: AsyncGenerator[List[ProcessedImageData], None]
    ) -> None:
        """
        Этап 3: Асинхронное сохранение результатов (3 файла на изображение).
        
        Args:
            convolution_generator: Генератор из этапа свёртки (чанки)
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        async for processed_batch in convolution_generator:
            # Сохраняем все изображения из чанка параллельно
            save_tasks = [
                self._save_processed_image(processed) 
                for processed in processed_batch
            ]
            await asyncio.gather(*save_tasks)
    
    async def _save_processed_image(self, processed: ProcessedImageData) -> None:
        """
        Асинхронно сохраняет все версии обработанного изображения (3 файла).
        
        Args:
            processed: Данные обработанного изображения
        """
        index = processed.index
        breed = processed.breed
        breed_safe = "".join(c if c.isalnum() else "_" for c in breed)
        
        print(f" Saving image {index} started (breed: {breed})")
        
        # Пути к файлам (3 файла на каждое изображение)
        original_path = os.path.join(self.output_dir, f"{index:03d}_{breed_safe}_original.png")
        custom_path = os.path.join(self.output_dir, f"{index:03d}_{breed_safe}_edges_custom.png")
        library_path = os.path.join(self.output_dir, f"{index:03d}_{breed_safe}_edges_library.png")
        
        # Параллельное асинхронное сохранение всех трёх файлов
        save_tasks = []
        
        if processed.original_data is not None:
            save_tasks.append(self._save_image_async(processed.original_data, original_path))
        
        if processed.custom_edges is not None:
            save_tasks.append(self._save_image_async(processed.custom_edges, custom_path))
        
        if processed.library_edges is not None:
            save_tasks.append(self._save_image_async(processed.library_edges, library_path))
        
        # Ждём завершения сохранения всех трёх файлов
        await asyncio.gather(*save_tasks)
        
        print(f" Saving image {index} finished (3 files: original, custom edges, library edges)")
    
    async def _save_image_async(self, image_data: np.ndarray, filepath: str) -> None:
        """
        Асинхронно сохраняет изображение в файл.
        
        Args:
            image_data: Данные изображения
            filepath: Путь к файлу для сохранения
        """
        # Конвертируем numpy array в PIL Image
        image = Image.fromarray(image_data)
        
        # Сохраняем в BytesIO (синхронная операция, но быстрая)
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Асинхронно записываем в файл
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(buffer.read())
    
    async def run_pipeline(self, limit: int) -> None:
        """
        Запускает полный пайплайн обработки изображений с chunked загрузкой.
        
        Args:
            limit: Количество изображений для обработки
        """
        print(f"\n{'='*70}")
        print(f"   Starting async chunked pipeline")
        print(f"   Total images: {limit}")
        print(f"   Chunk size: {self.chunk_size}")
        print(f"{'='*70}\n")
        
        # Создаем цепочку генераторов
        download_gen = self.stage_1_download(limit)
        convolution_gen = self.stage_2_convolution(download_gen)
        
        # Запускаем финальный этап (сохранение)
        await self.stage_3_save(convolution_gen)
        
        print(f"\n{'='*70}")
        print(f"   Pipeline completed successfully")
        print(f"   Results saved to: {os.path.abspath(self.output_dir)}")
        print(f"   Files per image: 3 (original, custom edges, library edges)")
        print(f"{'='*70}\n")
