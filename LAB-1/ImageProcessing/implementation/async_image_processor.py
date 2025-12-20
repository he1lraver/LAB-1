"""
Асинхронный процессор изображений с использованием aiohttp.

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

from implementation.abstract_image import ColorCatImage, GrayscaleCatImage


class ImageMetadata:
    """Метаданные изображения с фиксированным порядковым номером."""
    
    def __init__(self, index: int, url: str, breed: str):
        """
        Args:
            index: Порядковый номер (назначается при получении URL)
            url: URL изображения
            breed: Порода животного
        """
        self.index = index
        self.url = url
        self.breed = breed
        self.image_data: Optional[np.ndarray] = None


class AsyncImageProcessor:
    """
    Асинхронный процессор для загрузки изображений из API.
    
    Использует aiohttp для неблокирующих HTTP-запросов.
    """

    def __init__(self, api_key: str, api_type: str = "cat"):
        """
        Инициализация процессора.
        
        Args:
            api_key: API ключ для доступа к API
            api_type: Тип API ('cat' или 'dog')
        """
        self._api_key = api_key
        self._api_type = api_type.lower()
        self._base_url = self._get_base_url()

    def _get_base_url(self) -> str:
        """Возвращает базовый URL в зависимости от типа API."""
        if self._api_type == "cat":
            return "https://api.thecatapi.com/v1"
        elif self._api_type == "dog":
            return "https://api.thedogapi.com/v1"
        else:
            raise ValueError("Неизвестный тип API. Используйте 'cat' или 'dog'")

    async def fetch_image_urls(self, limit: int) -> List[ImageMetadata]:
        """
        Асинхронно получает список URL изображений из API.
        
        Порядковые номера назначаются здесь и не меняются в дальнейшем.
        
        Args:
            limit: Количество изображений для получения
            
        Returns:
            Список метаданных изображений с порядковыми номерами
        """
        print(f" Fetching {limit} image URLs from {self._api_type} API started")
        
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
                    
                    # Создаем метаданные с фиксированными порядковыми номерами
                    metadata_list = []
                    for index, item in enumerate(data, start=1):
                        image_url = item['url']
                        breed_info = item.get('breeds', [{}])[0] if item.get('breeds') else {}
                        breed_name = breed_info.get('name', 'unknown')
                        
                        metadata = ImageMetadata(index, image_url, breed_name)
                        metadata_list.append(metadata)
                        
                    print(f" Fetching image URLs finished: {len(metadata_list)} URLs obtained")
                    return metadata_list
                    
        except Exception as e:
            print(f" Error fetching image URLs: {e}")
            return []

    async def download_single_image(
        self, 
        session: aiohttp.ClientSession, 
        metadata: ImageMetadata
    ) -> Optional[ImageMetadata]:
        """
        Асинхронно загружает одно изображение.
        
        Args:
            session: Активная aiohttp сессия
            metadata: Метаданные изображения для загрузки
            
        Returns:
            Метаданные с заполненными данными изображения или None при ошибке
        """
        index = metadata.index
        url = metadata.url
        
        print(f"⬇  Downloading image {index} started (breed: {metadata.breed})")
        
        try:
            async with session.get(
                url, 
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                content = await response.read()
                
                # Конвертация в numpy array
                image = Image.open(BytesIO(content))
                image_data = np.array(image)
                
                if image_data.size == 0:
                    print(f"  Downloading image {index} failed: empty image")
                    return None
                
                # Сохраняем данные в метаданные
                metadata.image_data = image_data
                
                print(f" Downloading image {index} finished (shape: {image_data.shape})")
                return metadata
                
        except asyncio.TimeoutError:
            print(f" Downloading image {index} failed: timeout")
            return None
        except Exception as e:
            print(f" Downloading image {index} failed: {e}")
            return None

    async def download_images(self, metadata_list: List[ImageMetadata]) -> List[ImageMetadata]:
        """
        Асинхронно загружает все изображения параллельно.
        
        Args:
            metadata_list: Список метаданных для загрузки
            
        Returns:
            Список успешно загруженных метаданных
        """
        print(f"\n Starting parallel download of {len(metadata_list)} images...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.download_single_image(session, metadata) 
                for metadata in metadata_list
            ]
            results = await asyncio.gather(*tasks)
        
        # Фильтруем успешные загрузки
        successful = [r for r in results if r is not None and r.image_data is not None]
        
        print(f" Parallel download completed: {len(successful)}/{len(metadata_list)} successful\n")
        return successful

    def create_image_objects(self, metadata_list: List[ImageMetadata]) -> List[Tuple[int, object]]:
        """
        Создает объекты CatImage из загруженных данных.
        
        Args:
            metadata_list: Список метаданных с загруженными изображениями
            
        Returns:
            Список кортежей (порядковый_номер, объект_изображения)
        """
        image_objects = []
        
        for metadata in metadata_list:
            if metadata.image_data is None:
                continue
                
            # Создание объекта CatImage
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
            
            # Сохраняем с порядковым номером
            image_objects.append((metadata.index, cat_image))
        
        return image_objects
