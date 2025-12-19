"""
Асинхронный процессор изображений с логированием.

Обеспечивает:
- Асинхронную загрузку изображений из API
- Неблокирующие HTTP-запросы
- Параллельную загрузку множества изображений
- Детальное логирование каждого этапа
"""

import os
import asyncio
import aiohttp
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from io import BytesIO
from .abstract_image import ColorCatImage, GrayscaleCatImage
from image_processor.logging_config import get_logger


class ImageMetadata:
    """Метаданные изображения с фиксированным порядковым номером."""

    def __init__(self, index: int, url: str, breed: str):
        self.index = index
        self.url = url
        self.breed = breed
        self.image_data: Optional[np.ndarray] = None
        self.logger = get_logger()


class AsyncImageProcessor:
    """
    Асинхронный процессор для загрузки изображений из API с логированием.
    """

    def __init__(self, api_key: str, api_type: str = "cat"):
        """Инициализация процессора."""
        self.logger = get_logger()
        self._api_key = api_key
        self._api_type = api_type.lower()
        self._base_url = self._get_base_url()
        self.logger.debug(f"AsyncImageProcessor initialized for {api_type} API")

    def _get_base_url(self) -> str:
        """Возвращает базовый URL в зависимости от типа API."""
        if self._api_type == "cat":
            return "https://api.thecatapi.com/v1"
        elif self._api_type == "dog":
            return "https://api.thedogapi.com/v1"
        else:
            self.logger.error(f"Unknown API type: {self._api_type}")
            raise ValueError("Unknown API type. Use 'cat' or 'dog'")

    async def fetch_image_urls(self, limit: int) -> List[ImageMetadata]:
        """
        Асинхронно получает список URL изображений из API с логированием.
        """
        self.logger.info(f"Fetching {limit} image URLs from {self._api_type} API")

        headers = {"x-api-key": self._api_key}
        url = f"{self._base_url}/images/search"
        params = {
            "limit": limit,
            "has_breeds": 1,
            "size": "small"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    metadata_list = []
                    for index, item in enumerate(data, start=1):
                        image_url = item['url']
                        breed_info = item.get('breeds', [{}])[0] if item.get('breeds') else {}
                        breed_name = breed_info.get('name', 'unknown')
                        metadata = ImageMetadata(index, image_url, breed_name)
                        metadata_list.append(metadata)

                    self.logger.info(f"Successfully fetched {len(metadata_list)} image URLs")
                    return metadata_list
        except Exception as e:
            self.logger.error(f"Error fetching image URLs: {e}")
            return []

    async def download_single_image(
        self,
        session: aiohttp.ClientSession,
        metadata: ImageMetadata
    ) -> Optional[ImageMetadata]:
        """
        Асинхронно загружает одно изображение с логированием.
        """
        index = metadata.index
        url = metadata.url

        self.logger.debug(f"Downloading image {index} (breed: {metadata.breed})")

        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                content = await response.read()

                image = Image.open(BytesIO(content))
                image_data = np.array(image)

                if image_data.size == 0:
                    self.logger.warning(f"Image {index}: empty image received")
                    return None

                metadata.image_data = image_data
                self.logger.debug(f"Image {index} downloaded successfully (shape: {image_data.shape})")
                return metadata
        except asyncio.TimeoutError:
            self.logger.warning(f"Image {index}: timeout during download")
            return None
        except Exception as e:
            self.logger.error(f"Image {index}: download failed - {e}")
            return None

    async def download_images(self, metadata_list: List[ImageMetadata]) -> List[ImageMetadata]:
        """
        Асинхронно загружает все изображения параллельно с логированием.
        """
        self.logger.info(f"Starting parallel download of {len(metadata_list)} images")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.download_single_image(session, metadata)
                for metadata in metadata_list
            ]

            results = await asyncio.gather(*tasks)

            successful = [r for r in results if r is not None and r.image_data is not None]
            self.logger.info(f"Parallel download completed: {len(successful)}/{len(metadata_list)} successful")

            return successful

    def create_image_objects(self, metadata_list: List[ImageMetadata]) -> List[Tuple[int, object]]:
        """
        Создает объекты CatImage из загруженных данных с логированием.
        """
        self.logger.debug(f"Creating image objects from {len(metadata_list)} metadata entries")

        image_objects = []
        for metadata in metadata_list:
            if metadata.image_data is None:
                continue

            if len(metadata.image_data.shape) == 3 and metadata.image_data.shape[2] == 3:
                cat_image = ColorCatImage(
                    metadata.image_data,
                    metadata.url,
                    metadata.breed
                )
            else:
                cat_image = GrayscaleCatImage(
                    metadata.image_data,
                    metadata.url,
                    metadata.breed
                )

            image_objects.append((metadata.index, cat_image))
            self.logger.debug(f"Image object created for index {metadata.index} (breed: {metadata.breed})")

        self.logger.info(f"Created {len(image_objects)} image objects")
        return image_objects