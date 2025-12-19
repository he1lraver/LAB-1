"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений.
Реализован консольный интерфейс для применения различных методов обработки к изображению.

Запуск:
    python main.py

Автор: Фомин Максим
"""

import os
import sys
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Импортируем из вашего пакета image_processor
from image_processor.implementation.abstract_image import ColorCatImage, GrayscaleCatImage
from image_processor.implementation.image_processing import ImageProcessing
from image_processor.implementation.async_image_processor import AsyncImageProcessor, ImageMetadata
from image_processor.logging_config import get_logger

# Инициализируем логгер
logger = get_logger()


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
        print(f"  Error: Please select from: {', '.join(valid_choices)}")


# ============================================================================
# РЕЖИМ 1: ОБРАБОТКА ИЗОБРАЖЕНИЙ (ДЕМОНСТРАЦИЯ API)
# ============================================================================

def run_image_processing_demo():
    """Запускает демонстрацию обработки изображений."""
    print_header("Mode 1: Image Processing Demo")
    
    # Получаем API ключи из переменных окружения
    cat_api_key = os.getenv("CAT_API_KEY")
    dog_api_key = os.getenv("DOG_API_KEY")
    
    if not cat_api_key and not dog_api_key:
        print(" Error: API keys not found in .env file!")
        print("\n Setup instructions:")
        print("  1. Register on https://thecatapi.com/ or https://thedogapi.com/")
        print("  2. Get your API_KEY")
        print("  3. Create .env file in project root")
        print("  4. Add lines:")
        print("     CAT_API_KEY=your_cat_key")
        print("     DOG_API_KEY=your_dog_key")
        return
    
    print("[OK] API keys loaded successfully.")
    
    try:
        # Выбор источника данных
        api_options = {}
        if cat_api_key:
            api_options["1"] = "Cats (The Cat API)"
        if dog_api_key:
            api_options["2"] = "Dogs (The Dog API)"
        
        print_menu("Select image source", api_options)
        api_choice = get_user_choice("Choose source", list(api_options.keys()))
        
        # Определяем какой API использовать
        if api_choice == "1":
            api_key = cat_api_key
            api_type = "cat"
            processor = AsyncImageProcessor(api_key, api_type)
        else:
            api_key = dog_api_key
            api_type = "dog"
            processor = AsyncImageProcessor(api_key, api_type)
        
        print(f"\n[OK] Initialized {api_type.upper()} API Processor")
        print(f"     Base URL: {processor.baseurl}")
        
        # Демонстрация создания метаданных
        print("\n" + "="*70)
        print(" DEMO: Creating Image Metadata")
        print("="*70)
        
        metadata = ImageMetadata(index=1, url="http://example.com/image.jpg", breed="Sample Breed")
        print(f"\n[OK] Created metadata:")
        print(f"     Index: {metadata.index}")
        print(f"     URL: {metadata.url}")
        print(f"     Breed: {metadata.breed}")
        
        # Демонстрация обработки изображений
        print("\n" + "="*70)
        print(" DEMO: Image Processing Pipeline")
        print("="*70)
        
        processor_engine = ImageProcessing()
        print(f"\n[OK] Initialized ImageProcessing engine")
        print(f"     Optimization enabled: {processor_engine.optimization_enabled}")
        
        print("\n[INFO] Available methods:")
        print("     - RGB to Grayscale conversion")
        print("     - Edge detection (Canny algorithm)")
        print("     - Convolution operations")
        print("     - Gamma correction")
        
        print("\n" + "="*70)
        print(" [OK] Demo completed successfully!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n  Error: Program interrupted by user")
    except Exception as e:
        logger.error(f"Error in image processing demo: {e}")
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# РЕЖИМ 2: ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОСТИ
# ============================================================================

def run_tests():
    """Запускает встроенные тесты."""
    print_header("Mode 2: Running Tests")
    
    import unittest
    
    try:
        # Загружаем тесты из модуля image_processor.tests
        loader = unittest.TestLoader()
        suite = loader.discover('image_processor', pattern='tests*.py')
        
        # Запускаем тесты
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Выводим результаты
        print("\n" + "="*70)
        if result.wasSuccessful():
            print(" [OK] All tests passed successfully!")
        else:
            print(f" [ERROR] Tests failed:")
            print(f"     Failures: {len(result.failures)}")
            print(f"     Errors: {len(result.errors)}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main() -> None:
    """Главная функция приложения с меню выбора режима."""
    
    print_header("IMAGE PROCESSING APPLICATION - Main Menu")
    
    print(" Choose operating mode:\n")
    print("  1. Image Processing Demo")
    print("     (AsyncImageProcessor, ImageProcessing, API integration)")
    print("\n  2. Run Tests")
    print("     (unittest, package validation)")
    print("\n  0. Exit")
    
    choice = get_user_choice("Choose mode", ["0", "1", "2"])
    
    if choice == "0":
        print("\n Goodbye!\n")
        sys.exit(0)
    elif choice == "1":
        run_image_processing_demo()
    elif choice == "2":
        run_tests()
    
    # Предложить повторный выбор
    print("\n" + "=" * 70)
    again = input("Do you want to perform another operation? (y/n): ").strip().lower()
    
    if again == 'y':
        main()
    else:
        print("\n Goodbye!\n")


if __name__ == "__main__":
    logger.info("Application started")
    main()
    logger.info("Application finished")
