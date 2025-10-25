"""
Класс CatImageProcessor для работы с API и обработки изображений.
"""

import os
import time
import requests
import numpy as np
from typing import List, Dict, Any
from PIL import Image
import cv2

from implementation.abstract_image import CatImage, ColorCatImage, GrayscaleCatImage


def timing_decorator(func):
    """Декоратор для измерения времени выполнения методов."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"🕒 Начало выполнения {func.__name__}...")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"✅ {func.__name__} выполнен за {execution_time:.2f} секунд")
        
        return result
    return wrapper


class CatImageProcessor:
    """
    Класс для работы с API и обработки изображений животных.
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
    
    @timing_decorator
    def download_images(self, limit: int = 1) -> List[CatImage]:
        """Загружает изображения животных из API с повторными попытками."""
        print(f"📥 Загрузка {limit} изображений из {self._api_type} API...")
        
        headers = {"x-api-key": self._api_key}
        url = f"{self._base_url}/images/search"
        
        params = {
            "limit": limit,
            "has_breeds": 1,
            "size": "small"
        }
        
        try:
            # Запрос к API с увеличенным таймаутом
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            cat_images = []
            
            for item in data:
                try:
                    image_url = item['url']
                    breed_info = item.get('breeds', [{}])[0] if item.get('breeds') else {}
                    breed_name = breed_info.get('name', 'unknown')
                    
                    print(f"⬇️  Загрузка изображения: {breed_name}")
                    print(f"🔗 URL: {image_url}")
                    
                    # Загрузка изображения с повторными попытками
                    image_data = self._download_image_with_retry(image_url, max_retries=3)
                    if image_data is None:
                        continue
                    
                    # Создание объекта CatImage
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        cat_image = ColorCatImage(image_data, image_url, breed_name)
                    else:
                        cat_image = GrayscaleCatImage(image_data, image_url, breed_name)
                    
                    cat_images.append(cat_image)
                    print(f"✅ Успешно загружено: {breed_name}")
                    
                except Exception as e:
                    print(f"❌ Ошибка при загрузке изображения {image_url}: {e}")
                    continue
            
            return cat_images
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка при запросе к API: {e}")
            return []
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            return []

    def _download_image_with_retry(self, image_url: str, max_retries: int = 3) -> np.ndarray:
        """Загружает изображение с повторными попытками."""
        for attempt in range(max_retries):
            try:
                print(f"   Попытка {attempt + 1}/{max_retries}...")
                
                # Загрузка с увеличенным таймаутом
                response = requests.get(image_url, stream=True, timeout=60)
                response.raise_for_status()
                
                # Конвертация в numpy array
                image = Image.open(response.raw)
                image_data = np.array(image)
                
                if image_data.size == 0:
                    print(f"⚠️ Пустое изображение: {image_url}")
                    return None
                
                print(f"   Размер изображения: {image_data.shape}")
                return image_data
                
            except requests.exceptions.Timeout:
                print(f"   ⏰ Таймаут при загрузке (попытка {attempt + 1})")
                if attempt == max_retries - 1:
                    print(f"❌ Превышено количество попыток для {image_url}")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"   🌐 Ошибка сети: {e} (попытка {attempt + 1})")
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                print(f"   ⚠️ Ошибка при обработке изображения: {e}")
                return None
        
        return None
    
    @timing_decorator
    def process_images(self, images: List[CatImage], output_dir: str = "animal_images") -> None:
        """
        Обрабатывает и сохраняет изображения.
        
        Args:
            images: Список изображений для обработки
            output_dir: Директория для сохранения результатов
        """
        print(f"🔧 Обработка {len(images)} изображений...")
        
        # Создание директории для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        for i, cat_image in enumerate(images, 1):
            print(f"🔧 Обработка изображения {i}/{len(images)}: {cat_image.breed}")
            
            try:
                # Обработка изображения
                custom_edges = cat_image.custom_edge_detection()
                library_edges = cat_image.library_edge_detection()
                
                # Сохранение результатов
                breed_safe = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
                
                # Исходное изображение
                original_path = os.path.join(output_dir, f"{i}_{breed_safe}_original.png")
                if len(cat_image.image_data.shape) == 3:
                    Image.fromarray(cat_image.image_data).save(original_path)
                else:
                    Image.fromarray(cat_image.image_data).convert('RGB').save(original_path)
                
                # Пользовательский метод
                custom_path = os.path.join(output_dir, f"{i}_{breed_safe}_custom_edges.png")
                Image.fromarray(custom_edges).save(custom_path)
                
                # Библиотечный метод
                library_path = os.path.join(output_dir, f"{i}_{breed_safe}_library_edges.png")
                Image.fromarray(library_edges).save(library_path)
                
                print(f"💾 Сохранены файлы для {cat_image.breed}:")
                print(f"   - {original_path}")
                print(f"   - {custom_path}")
                print(f"   - {library_path}")
                
            except Exception as e:
                print(f"❌ Ошибка при обработке изображения {cat_image.breed}: {e}")
                continue
    
    @timing_decorator
    def run_processing_pipeline(self, limit: int = 3, output_dir: str = "animal_images") -> None:
        """
        Запускает полный pipeline обработки изображений.
        
        Args:
            limit: Количество изображений для обработки
            output_dir: Директория для сохранения результатов
        """
        print("🚀 Запуск pipeline обработки изображений...")
        print(f"🎯 Цель: {limit} изображений")
        print(f"📁 Выходная директория: {output_dir}")
        
        # Загрузка изображений
        images = self.download_images(limit)
        
        if not images:
            print("❌ Не удалось загрузить изображения")
            return
        
        # Обработка изображений
        self.process_images(images, output_dir)
        
        print("🎉 Pipeline обработки изображений завершен!")


class SimpleImageProcessor:
    """
    Упрощенный класс только для загрузки изображений (для обратной совместимости).
    """
    
    def __init__(self, api_key: str, api_type: str):
        self.api_key = api_key
        self.api_type = api_type
        self.base_url = f"https://api.the{api_type}api.com/v1/images/search"
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}

    def download_image(self, url: str, save_path: str, max_retries: int = 3, timeout: int = 30) -> bool:
        """Загрузка изображения с повторными попытками."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout, headers=self.headers)
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(response.content)
                return True
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Попытка {attempt + 1}/{max_retries}... Ошибка: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        return False

    def fetch_image_urls(self, limit: int) -> List[str]:
        """Получение списка URL изображений."""
        try:
            response = requests.get(
                f"{self.base_url}?limit={limit}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return [item["url"] for item in data]
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка при получении URL изображений: {e}")
            return []

    def run_processing_pipeline(self, limit: int, output_dir: str) -> None:
        """Основной метод обработки изображений."""
        print(f"🚀 Запуск pipeline обработки изображений {self.api_type}...")
        print(f"🎯 Цель: {limit} изображений")
        print(f"📁 Выходная директория: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        image_urls = self.fetch_image_urls(limit)
        if not image_urls:
            print("❌ Не удалось получить URL изображений")
            return

        success_count = 0
        for i, url in enumerate(image_urls, 1):
            filename = os.path.join(output_dir, f"{self.api_type}_image_{i}.jpg")
            print(f"⬇️ Загрузка изображения {i}/{limit}: {url.split('/')[-1]}")
            if self.download_image(url, filename):
                print(f"✅ Изображение сохранено: {filename}")
                success_count += 1
            else:
                print(f"❌ Не удалось загрузить изображение: {url}")

        elapsed_time = time.time() - start_time
        print(f"✅ Завершено за {elapsed_time:.2f} секунд")
        print(f"📊 Успешно загружено {success_count}/{limit} изображений")