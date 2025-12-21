"""
main.py

Лабораторная работа: обработка изображений + асинхронный пайплайн.

Режимы:
1) Async Image Pipeline (реальная обработка: download -> edges -> save)
2) Demo (показ API/классов, без запуска пайплайна)
3) Run Tests

Запуск:
py -m image_processor
"""

import os
import sys
import asyncio

from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

from image_processor.implementation.image_processing import ImageProcessing
from image_processor.implementation.async_image_processor import AsyncImageProcessor, ImageMetadata
from image_processor.implementation.async_pipeline import AsyncImagePipeline
from image_processor.logging_config import get_logger

logger = get_logger()


# ============================================================================
# УТИЛИТЫ ДЛЯ ИНТЕРФЕЙСА
# ============================================================================

def print_header(text: str) -> None:
    """Печатает красивый заголовок."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


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
        print(f"  Error: Please select from: {', '.join(valid_choices)}")


def _select_api_source() -> tuple[str, str]:
    """
    Возвращает (api_key, api_type).
    api_type: 'cat' или 'dog'
    """
    cat_api_key = os.getenv("CAT_API_KEY")
    dog_api_key = os.getenv("DOG_API_KEY")

    if not cat_api_key and not dog_api_key:
        print("  Error: API keys not found in .env file!\n")
        print("  Setup instructions:")
        print("  1. Register on https://thecatapi.com/ or https://thedogapi.com/")
        print("  2. Get your API_KEY")
        print("  3. Create .env file in project root")
        print("  4. Add lines:")
        print("     CAT_API_KEY=your_cat_key")
        print("     DOG_API_KEY=your_dog_key")
        raise SystemExit(2)

    print("[OK] API keys loaded successfully.")

    api_options = {}
    if cat_api_key:
        api_options["1"] = "Cats (The Cat API)"
    if dog_api_key:
        api_options["2"] = "Dogs (The Dog API)"

    print_menu("Select image source", api_options)
    api_choice = get_user_choice("Choose source", list(api_options.keys()))

    if api_choice == "1":
        return cat_api_key, "cat"
    return dog_api_key, "dog"


# ============================================================================
# РЕЖИМ 1: РЕАЛЬНЫЙ ASYNC PIPELINE
# ============================================================================

def run_async_pipeline() -> None:
    """Запускает полный пайплайн: download -> edges -> save."""
    print_header("Mode 1: Async Image Pipeline")

    try:
        api_key, api_type = _select_api_source()

        limit_str = input("\nLimit (default 10): ").strip()
        limit = int(limit_str) if limit_str else 10

        chunk_str = input("Chunk size (default 5): ").strip()
        chunk_size = int(chunk_str) if chunk_str else 5

        out_dir = input("Output dir (default out): ").strip() or "out"

        pipeline = AsyncImagePipeline(
            api_key=api_key,
            api_type=api_type,
            output_dir=out_dir,
            chunk_size=chunk_size
        )

        asyncio.run(pipeline.run_pipeline(limit=limit))

        print("\n[OK] Pipeline finished.")
        print(f"     Results saved to: {os.path.abspath(out_dir)}")

    except KeyboardInterrupt:
        print("\n\n  Error: Program interrupted by user")

    except Exception as e:
        logger.error(f"Error in async pipeline: {e}")
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# РЕЖИМ 2: DEMO (как раньше, без реальной обработки)
# ============================================================================

def run_image_processing_demo() -> None:
    """Запускает демонстрацию API/классов (без запуска пайплайна)."""
    print_header("Mode 2: Image Processing Demo")

    try:
        api_key, api_type = _select_api_source()

        processor = AsyncImageProcessor(api_key, api_type)
        print(f"\n[OK] Initialized {api_type.upper()} API Processor")
        print(f"     Base URL: {processor.baseurl}")

        print("\n" + "=" * 70)
        print("  DEMO: Creating Image Metadata")
        print("=" * 70)

        metadata = ImageMetadata(index=1, url="http://example.com/image.jpg", breed="Sample Breed")
        print("\n[OK] Created metadata:")
        print(f"     Index: {metadata.index}")
        print(f"     URL: {metadata.url}")
        print(f"     Breed: {metadata.breed}")

        print("\n" + "=" * 70)
        print("  DEMO: ImageProcessing methods")
        print("=" * 70)

        processor_engine = ImageProcessing()
        print("\n[OK] Initialized ImageProcessing engine")
        print(f"     Optimization enabled: {processor_engine.optimization_enabled}")

        print("\n[INFO] Available methods:")
        print("     - RGB to Grayscale conversion")
        print("     - Edge detection (Canny-like pipeline)")
        print("     - Convolution operations")
        print("     - Gamma correction")

        print("\n" + "=" * 70)
        print("  [OK] Demo completed successfully!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n  Error: Program interrupted by user")

    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# РЕЖИМ 3: ТЕСТЫ
# ============================================================================

def run_tests() -> None:
    """Запускает встроенные тесты."""
    print_header("Mode 3: Running Tests")
    import unittest

    try:
        loader = unittest.TestLoader()
        suite = loader.discover("image_processor", pattern="tests*.py")

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        print("\n" + "=" * 70)
        if result.wasSuccessful():
            print("  [OK] All tests passed successfully!")
        else:
            print("  [ERROR] Tests failed:")
            print(f"  Failures: {len(result.failures)}")
            print(f"  Errors: {len(result.errors)}")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================================

def main() -> None:
    """Главная функция приложения с меню выбора режима."""
    print_header("IMAGE PROCESSING APPLICATION - Main Menu")

    print("  Choose operating mode:\n")
    print("  1. Run Async Image Pipeline")
    print("     (Download -> Process -> Save)")
    print("\n  2. Image Processing Demo")
    print("     (AsyncImageProcessor, ImageProcessing, API integration)")
    print("\n  3. Run Tests")
    print("     (unittest, package validation)")
    print("\n  0. Exit")

    choice = get_user_choice("Choose mode", ["0", "1", "2", "3"])

    if choice == "0":
        print("\n  Goodbye!\n")
        sys.exit(0)
    elif choice == "1":
        run_async_pipeline()
    elif choice == "2":
        run_image_processing_demo()
    elif choice == "3":
        run_tests()

    print("\n" + "=" * 70)
    again = input("Do you want to perform another operation? (y/n): ").strip().lower()
    if again == "y":
        main()
    else:
        print("\n  Goodbye!\n")


if __name__ == "__main__":
    logger.info("Application started")
    main()
    logger.info("Application finished")
