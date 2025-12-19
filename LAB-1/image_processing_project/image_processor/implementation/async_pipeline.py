"""
Асинхронный генераторный пайплайн для обработки изображений с логированием.

Реализует:
- Chunked обработка (можно задать размер чанка)
- Три этапа обработки: скачивание, свёртка, сохранение
- Каждый этап - отдельный асинхронный генератор
- Детальное логирование всех операций
"""

import os
import asyncio
import aiofiles
import numpy as np
from typing import AsyncGenerator, Tuple, List
from PIL import Image
from .async_image_processor import AsyncImageProcessor, ImageMetadata
from .parallel_convolution import ParallelConvolutionProcessor
from image_processor.logging_config import get_logger



class ProcessedImageData:
    """Данные обработанного изображения со всеми результатами."""

    def __init__(self, index: int, breed: str):
        self.index = index
        self.breed = breed
        self.original_data: np.ndarray = None
        self.custom_edges: np.ndarray = None
        self.library_edges: np.ndarray = None
        self.original_plus_custom: np.ndarray = None
        self.original_minus_custom: np.ndarray = None
        self.original_plus_library: np.ndarray = None
        self.original_minus_library: np.ndarray = None
        self.logger = get_logger()


class AsyncImagePipeline:
    """
    Асинхронный генераторный пайплайн для обработки изображений с логированием.
    """

    def __init__(self, api_key: str, api_type: str, output_dir: str, chunk_size: int = 5):
        """Инициализирует пайплайн."""
        self.logger = get_logger()
        self.api_key = api_key
        self.api_type = api_type
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.processor = AsyncImageProcessor(api_key, api_type)
        self.convolution_processor = ParallelConvolutionProcessor()
        self.logger.info(f"AsyncImagePipeline initialized for {api_type} API with chunk_size={chunk_size}")

    async def stage_1_download(self, limit: int) -> AsyncGenerator[List[Tuple[int, object]], None]:
        """
        Этап 1: Асинхронное скачивание изображений по чанкам с логированием.
        """
        self.logger.info(f"Stage 1 started: Downloading {limit} images with chunk_size={self.chunk_size}")

        metadata_list = await self.processor.fetch_image_urls(limit)

        total_chunks = (len(metadata_list) + self.chunk_size - 1) // self.chunk_size
        self.logger.info(f"Total {len(metadata_list)} images will be processed in {total_chunks} chunk(s)")

        for chunk_idx in range(0, len(metadata_list), self.chunk_size):
            chunk_metadata = metadata_list[chunk_idx:chunk_idx + self.chunk_size]
            chunk_num = (chunk_idx // self.chunk_size) + 1

            self.logger.info(f"Processing chunk {chunk_num}/{total_chunks}: images {chunk_idx + 1} to {min(chunk_idx + self.chunk_size, len(metadata_list))}")

            downloaded_metadata = await self.processor.download_images(chunk_metadata)
            image_objects = self.processor.create_image_objects(downloaded_metadata)

            if image_objects:
                self.logger.debug(f"Chunk {chunk_num}: {len(image_objects)} images ready for processing")
                yield image_objects

        self.logger.info("Stage 1 completed: All chunks downloaded")

    async def stage_2_convolution(
        self,
        download_generator: AsyncGenerator[List[Tuple[int, object]], None]
    ) -> AsyncGenerator[List[ProcessedImageData], None]:
        """
        Этап 2: Параллельное применение свёртки и арифметических операций с логированием.
        """
        self.logger.info("Stage 2 started: Convolution & Arithmetic Operations")

        async for image_batch in download_generator:
            if not image_batch:
                continue

            chunk_size = len(image_batch)
            self.logger.debug(f"Processing chunk of {chunk_size} images")

            # Запускаем custom и library edge detection параллельно
            custom_task = asyncio.to_thread(
                self.convolution_processor.process_batch_custom,
                image_batch
            )

            library_task = asyncio.to_thread(
                self.convolution_processor.process_batch_library,
                image_batch
            )

            custom_results, library_results = await asyncio.gather(
                custom_task,
                library_task
            )

            self.logger.debug("Edge detection completed")

            # Запускаем арифметические операции
            arithmetic_task = asyncio.to_thread(
                self.convolution_processor.process_batch_arithmetic,
                image_batch,
                custom_results,
                library_results
            )

            arithmetic_results = await arithmetic_task
            self.logger.debug("Arithmetic operations completed")

            # Создаем словари для быстрого доступа
            custom_dict = {idx: (breed, edges) for idx, breed, edges in custom_results}
            library_dict = {idx: (breed, edges) for idx, breed, edges in library_results}
            arithmetic_dict = {idx: results for idx, results in arithmetic_results}

            # Формируем финальные данные
            processed_batch = []
            for index, image_obj in image_batch:
                processed = ProcessedImageData(index, image_obj.breed)

                processed.original_data = image_obj.image_data

                if index in custom_dict:
                    processed.custom_edges = custom_dict[index][1]
                if index in library_dict:
                    processed.library_edges = library_dict[index][1]

                if index in arithmetic_dict:
                    arith = arithmetic_dict[index]
                    processed.original_plus_custom = arith['plus_custom']
                    processed.original_minus_custom = arith['minus_custom']
                    processed.original_plus_library = arith['plus_library']
                    processed.original_minus_library = arith['minus_library']

                processed_batch.append(processed)

            self.logger.debug(f"Chunk processed: {chunk_size} images with all operations completed")
            yield processed_batch

        self.logger.info("Stage 2 completed: All convolutions and operations done")

    async def stage_3_save(
        self,
        convolution_generator: AsyncGenerator[List[ProcessedImageData], None]
    ) -> None:
        """
        Этап 3: Асинхронное сохранение результатов с логированием.
        """
        self.logger.info("Stage 3 started: Saving results")
        os.makedirs(self.output_dir, exist_ok=True)

        total_saved = 0
        async for processed_batch in convolution_generator:
            save_tasks = [
                self._save_processed_image(processed)
                for processed in processed_batch
            ]

            await asyncio.gather(*save_tasks)
            total_saved += len(processed_batch)

        self.logger.info(f"Stage 3 completed: {total_saved} images saved")

    async def _save_processed_image(self, processed: ProcessedImageData) -> None:
        """
        Асинхронно сохраняет все версии обработанного изображения (3 файла).
        """
        index = processed.index
        breed = processed.breed
        breed_safe = "".join(c if c.isalnum() else "_" for c in breed)

        self.logger.debug(f"Saving image {index} ({breed})")

        original_path = os.path.join(self.output_dir, f"{index:03d}_{breed_safe}_original.png")
        custom_path = os.path.join(self.output_dir, f"{index:03d}_{breed_safe}_edges_custom.png")
        library_path = os.path.join(self.output_dir, f"{index:03d}_{breed_safe}_edges_library.png")

        save_tasks = []

        if processed.original_data is not None:
            save_tasks.append(self._save_image_async(processed.original_data, original_path))
        if processed.custom_edges is not None:
            save_tasks.append(self._save_image_async(processed.custom_edges, custom_path))
        if processed.library_edges is not None:
            save_tasks.append(self._save_image_async(processed.library_edges, library_path))

        await asyncio.gather(*save_tasks)
        self.logger.debug(f"Image {index} saved (3 files)")

    async def _save_image_async(self, image_data: np.ndarray, filepath: str) -> None:
        """
        Асинхронно сохраняет изображение в файл.
        """
        image = Image.fromarray(image_data)

        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(buffer.read())

    async def run_pipeline(self, limit: int) -> None:
        """
        Запускает полный пайплайн обработки изображений с логированием.
        """
        self.logger.info(f"Pipeline execution started. Total images: {limit}, Chunk size: {self.chunk_size}")

        try:
            download_gen = self.stage_1_download(limit)
            convolution_gen = self.stage_2_convolution(download_gen)
            await self.stage_3_save(convolution_gen)

            self.logger.info(f"Pipeline execution completed successfully. Results saved to: {os.path.abspath(self.output_dir)}")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise