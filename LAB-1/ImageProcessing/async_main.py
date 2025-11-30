"""
Асинхронная версия программы обработки изображений животных с поддержкой chunked загрузки.

Основные улучшения:
1. Асинхронное скачивание изображений (aiohttp)
2. Асинхронное сохранение файлов (aiofiles)
3. Параллельная обработка свёртки (multiprocessing)
4. Chunked обработка с настраиваемым размером чанка
5. Генераторный пайплайн для обработки
6. Детальное логирование всех этапов
7. Сохранение порядковых номеров изображений
8. Custom и Library edge detection выполняются параллельно
9. Сохранение 3 файлов: оригинал, custom edges, library edges
"""

import os
import sys
import asyncio
import time
from typing import List, Tuple, AsyncGenerator
from dotenv import load_dotenv
from implementation.async_image_processor import AsyncImageProcessor
from logging_config import setup_logging, get_logger

from implementation.async_pipeline import AsyncImagePipeline


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
        print(f"   Пожалуйста, выберите из: {', '.join(valid_choices)}")


def get_positive_integer(prompt: str, default: int, min_val: int = 1, max_val: int = 100) -> int:
    """Получает положительное целое число от пользователя."""
    while True:
        try:
            value_str = input(f"{prompt} (по умолчанию {default}): ").strip()
            value = int(value_str) if value_str else default
            if min_val <= value <= max_val:
                return value
            print(f"   Пожалуйста, введите число от {min_val} до {max_val}")
        except ValueError:
            print(f"   Пожалуйста, введите корректное число")


# ============================================================================
# ГЛАВНАЯ АСИНХРОННАЯ ФУНКЦИЯ ОБРАБОТКИ
# ============================================================================

async def run_async_image_processing():
    """Запускает асинхронный режим обработки изображений животных."""
    print_header(" Асинхронная Обработка Изображений Животных (Chunked Pipeline)")

    load_dotenv()
    cat_api_key = os.getenv("CAT_API_KEY")
    dog_api_key = os.getenv("DOG_API_KEY")

    # Проверка API ключей
    if not cat_api_key and not dog_api_key:
        print("   Ошибка: API ключи не найдены в файле .env!")
        print("\n  Инструкции по настройке:")
        print("    1. Зарегистрируйтесь на https://thecatapi.com/ или https://thedogapi.com/")
        print("    2. Получите API_KEY")
        print("    3. Создайте файл .env в корне проекта")
        print("    4. Добавьте строки:")
        print("       CAT_API_KEY=ваш_ключ_кошек")
        print("       DOG_API_KEY=ваш_ключ_собак")
        return

    print(" API ключи загружены успешно.")

    try:
        # Выбор источника данных
        api_options = {}
        if cat_api_key:
            api_options["1"] = "Кошки (The Cat API)"
        if dog_api_key:
            api_options["2"] = "Собаки (The Dog API)"
        if cat_api_key and dog_api_key:
            api_options["3"] = "Кошки и Собаки"

        print_menu(" Выбор источника изображений", api_options)
        api_choice = get_user_choice("Выберите источник", list(api_options.keys()))

        # Количество изображений
        print("\n  Настройка параметров обработки:")
        print("    Можно обрабатывать до 100 изображений параллельно!")
        
        limit = get_positive_integer(
            "  Введите количество изображений",
            default=10,
            min_val=1,
            max_val=100
        )
        
        # Размер чанка
        print("\n  Размер чанка для параллельной обработки:")
        print("    Чанк - это группа изображений, обрабатываемая одновременно")
        print("    Рекомендуется: 5-10 для большинства систем")
        
        chunk_size = get_positive_integer(
            "  Введите размер чанка",
            default=5,
            min_val=1,
            max_val=min(20, limit)
        )

        # Директория для сохранения
        output_dir = input("\n Директория для сохранения (по умолчанию './results_async'): ").strip()
        output_dir = output_dir if output_dir else "./results_async"
        os.makedirs(output_dir, exist_ok=True)
        print(f" Директория: {os.path.abspath(output_dir)}")

        # =====================================================================
        # ЗАПУСК АСИНХРОННОГО ПАЙПЛАЙНА
        # =====================================================================
        print_header(f" Запуск асинхронного chunked пайплайна")
        print(f"   Всего изображений: {limit}")
        print(f"   Размер чанка: {chunk_size}")
        print(f"   Файлов на изображение: 3 (оригинал + 2 edge detection)")
        
        total_start_time = time.time()

        # Обработка кошек
        if api_choice in ["1", "3"] and cat_api_key:
            print("\n Обработка изображений кошек...")
            cat_output_dir = os.path.join(output_dir, "cats") if api_choice == "3" else output_dir
            
            pipeline = AsyncImagePipeline(cat_api_key, "cat", cat_output_dir, chunk_size)
            await pipeline.run_pipeline(limit)

        # Обработка собак
        if api_choice in ["2", "3"] and dog_api_key:
            print("\n Обработка изображений собак...")
            dog_output_dir = os.path.join(output_dir, "dogs") if api_choice == "3" else output_dir
            
            pipeline = AsyncImagePipeline(dog_api_key, "dog", dog_output_dir, chunk_size)
            await pipeline.run_pipeline(limit)

        total_elapsed = time.time() - total_start_time
        total_files = limit * 3  # 3 файла на изображение

        print_header(" Асинхронная обработка изображений завершена")
        print(f"  Общее время выполнения: {total_elapsed:.2f} секунд")
        print(f" Обработано изображений: {limit}")
        print(f" Всего сохранено файлов: {total_files}")
        print(f" Средняя скорость: {limit / total_elapsed:.2f} изображений/сек")
        print(f" Результаты сохранены в: {os.path.abspath(output_dir)}\n")

    except KeyboardInterrupt:
        print("\n\n  Программа прервана пользователем")
    except Exception as e:
        print(f"\n   Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main() -> None:
    """Главная функция приложения."""
    print_header(" АСИНХРОННАЯ CHUNKED ОБРАБОТКА ИЗОБРАЖЕНИЙ")
    print("   Возможности:")
    print("    • Асинхронное скачивание (aiohttp)")
    print("    • Асинхронное сохранение (aiofiles)")
    print("    • Параллельная обработка (multiprocessing)")
    print("    • Chunked загрузка (настраиваемый размер)")
    print("    • Генераторный пайплайн")
    print("    • Custom + Library edge detection параллельно")
    print("    • 3 файла на изображение (оригинал, custom, library)")
    print("    • Детальное логирование этапов")
    print("    • Сохранение порядковых номеров")
    print("    • Масштабируемость до 100+ изображений")

    # Запуск асинхронной функции
    asyncio.run(run_async_image_processing())

    # Предложить повторный запуск
    print("\n" + "="*70)
    again = input("Хотите выполнить еще одну обработку? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("\n До свидания!\n")


if __name__ == "__main__":
    main()
