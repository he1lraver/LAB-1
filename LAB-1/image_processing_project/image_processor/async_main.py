"""
async_main.py

Асинхронный модуль для обработки изображений животных через API.
Демонстрирует использование AsyncImageProcessor и AsyncImagePipeline.

Запуск:
    python async_main.py
    
или через пакет:
    python -m image_processor.async_main
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Импортируем из вашего пакета
from image_processor.implementation.async_image_processor import AsyncImageProcessor
from image_processor.implementation.async_pipeline import AsyncImagePipeline
from image_processor.logging_config import get_logger

# Инициализируем логгер
logger = get_logger()

# Загружаем переменные окружения
load_dotenv()


# ============================================================================
# УТИЛИТЫ ДЛЯ ИНТЕРФЕЙСА
# ============================================================================

def print_header(text: str) -> None:
    """Печатает красивый заголовок."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_menu(title: str, options: dict) -> None:
    """Печатает меню с опциями."""
    print(f"\n{title}:")
    for key, value in options.items():
        print(f"  {key}. {value}")


def get_user_choice(prompt: str, valid_choices: list) -> str:
    """Получает выбор пользователя с валидацией."""
    while True:
        choice = input(f"\n{prompt}: ").strip()
        if choice in valid_choices:
            return choice
        print(f"  ❌ Error: Please select from: {', '.join(valid_choices)}")


def get_positive_integer(prompt: str, min_val: int = 1, max_val: int = 50) -> int:
    """Получает положительное целое число с валидацией."""
    while True:
        try:
            value_str = input(f"\n{prompt} (recommended {min_val}-{max_val}): ").strip()
            if not value_str:
                return min_val
            value = int(value_str)
            if min_val <= value <= max_val:
                return value
            print(f"  ❌ Error: Enter number between {min_val} and {max_val}")
        except ValueError:
            print(f"  ❌ Error: Enter a valid number")


# ============================================================================
# ОСНОВНАЯ АСИНХРОННАЯ ФУНКЦИЯ
# ============================================================================

async def process_animals(api_keys: dict):
    """Основная функция обработки животных с интерактивным меню."""
    print_header("Async Image Processing Pipeline - Animal Selection")
    
    # ========== ВЫБОР ИСТОЧНИКА ДАННЫХ ==========
    api_options = {}
    if api_keys["cat"]:
        api_options["1"] = "Cats only (The Cat API)"
    if api_keys["dog"]:
        api_options["2"] = "Dogs only (The Dog API)"
    if api_keys["cat"] and api_keys["dog"]:
        api_options["3"] = "Cats and Dogs"
    
    print_menu("Select image source", api_options)
    api_choice = get_user_choice("Choose source", list(api_options.keys()))
    
    # Определяем какие животные обрабатывать
    animals = []
    if api_choice in ["1", "3"]:
        animals.append(("cat", api_keys["cat"]))
    if api_choice in ["2", "3"]:
        animals.append(("dog", api_keys["dog"]))
    
    logger.info(f"Selected animals: {', '.join([a[0].upper() for a in animals])}")
    
    # ========== КОЛИЧЕСТВО ИЗОБРАЖЕНИЙ ==========
    print_header("Image Count Configuration")
    print("Recommended: 1-5 images for testing, up to 50 for full processing")
    
    limit = get_positive_integer("Enter number of images per animal", min_val=1, max_val=50)
    logger.info(f"Image limit set to: {limit}")
    
    # ========== РАЗМЕР ЧАНКА ==========
    print_header("Chunk Size Configuration")
    print("Chunk size determines how many images are processed in parallel")
    print("  - Smaller chunk size = more sequential (safer for API)")
    print("  - Larger chunk size = more parallel (faster but needs more resources)")
    
    chunk_size = get_positive_integer("Enter chunk size", min_val=1, max_val=10)
    logger.info(f"Chunk size set to: {chunk_size}")
    
    # ========== ОБРАБОТКА КАЖДОГО ЖИВОТНОГО ==========
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    for animal_type, api_key in animals:
        if not api_key:
            logger.warning(f"Skipping {animal_type.upper()} - API key not available")
            continue
        
        logger.info("=" * 70)
        logger.info(f"Processing {animal_type.upper()} images")
        logger.info("=" * 70)
        
        try:
            # Инициализируем пайплайн для текущего животного
            logger.info(f"Initializing AsyncImagePipeline for {animal_type}")
            logger.debug(f"Parameters: api_type={animal_type}, limit={limit}, chunk_size={chunk_size}")
            
            pipeline = AsyncImagePipeline(
                api_key=api_key,
                api_type=animal_type,
                output_dir=os.path.join(output_dir, animal_type),
                chunk_size=chunk_size
            )
            
            logger.info(f"AsyncImagePipeline initialized successfully for {animal_type}")
            logger.debug(f"API type: {animal_type}")
            
            # Запускаем пайплайн
            logger.info(f"Running pipeline with {limit} {animal_type} images and chunk_size={chunk_size}...")
            await pipeline.run_pipeline(limit=limit)
            
            logger.info("=" * 70)
            logger.info(f"{animal_type.upper()} processing completed!")
            logger.info(f"Results saved to: {os.path.abspath(os.path.join(output_dir, animal_type))}")
            logger.info("=" * 70)
            
        except KeyboardInterrupt:
            logger.warning(f"Processing of {animal_type.upper()} interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error processing {animal_type.upper()}: {e}", exc_info=True)
            print(f"\n❌ Error processing {animal_type.upper()}: {e}")
            continue
    
    # ========== ЗАВЕРШЕНИЕ ==========
    print_header("Processing Completed")
    logger.info(f"All results saved to: {os.path.abspath(output_dir)}")
    print(f"✅ All results saved to: {os.path.abspath(output_dir)}\n")


async def main():
    """Основная асинхронная функция."""
    logger.info("=" * 70)
    logger.info("Async Image Processing Pipeline Started")
    logger.info("=" * 70)
    
    # Получаем API ключи
    cat_api_key = os.getenv("CAT_API_KEY")
    dog_api_key = os.getenv("DOG_API_KEY")
    
    api_keys = {
        "cat": cat_api_key,
        "dog": dog_api_key
    }
    
    # Проверяем наличие хотя бы одного ключа
    if not cat_api_key and not dog_api_key:
        logger.error("No API keys found in .env file")
        print("\n Error: API keys not found in .env file!")
        print("\n Setup instructions:")
        print("  1. Register on https://thecatapi.com/ or https://thedogapi.com/")
        print("  2. Get your API_KEY")
        print("  3. Create .env file in project root")
        print("  4. Add lines:")
        print("     CAT_API_KEY=your_cat_key")
        print("     DOG_API_KEY=your_dog_key")
        return
    
    logger.info("API keys loaded successfully")
    if cat_api_key:
        logger.debug("CAT_API_KEY is available")
    if dog_api_key:
        logger.debug("DOG_API_KEY is available")
    
    try:
        # Запускаем интерактивное меню обработки
        await process_animals(api_keys)
        
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Async main module started")
    
    # Запускаем асинхронную функцию
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Async main module finished")
