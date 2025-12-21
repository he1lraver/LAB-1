"""
main.py

Лабораторная работа: обработка изображений + асинхронный пайплайн
+ анализ данных video_games (CSV/Parquet).

Запуск:
py -m image_processor
"""

import os
import sys
import asyncio
from pathlib import Path

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
    print(f" {text}")
    print(f"{'=' * 70}\n")


def print_menu(title: str, options: dict) -> None:
    """Печатает меню с опциями."""
    print(f"\n{title}:")
    for key, value in options.items():
        print(f" {key}. {value}")


def get_user_choice(prompt: str, valid_choices: list) -> str:
    """Получает выбор пользователя с валидацией."""
    while True:
        choice = input(f"\n{prompt}: ").strip()
        if choice in valid_choices:
            return choice
        print(f" Error: Please select from: {', '.join(valid_choices)}")


def _select_api_source() -> tuple[str, str]:
    """
    Возвращает (api_key, api_type).
    api_type: 'cat' или 'dog'
    """
    cat_api_key = os.getenv("CAT_API_KEY")
    dog_api_key = os.getenv("DOG_API_KEY")

    if not cat_api_key and not dog_api_key:
        print(" Error: API keys not found in .env file!\n")
        print(" Setup instructions:")
        print(" 1. Register on https://thecatapi.com/ or https://thedogapi.com/")
        print(" 2. Get your API_KEY")
        print(" 3. Create .env file in project root")
        print(" 4. Add lines:")
        print(" CAT_API_KEY=your_cat_key")
        print(" DOG_API_KEY=your_dog_key")
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


def _get_video_games_paths() -> tuple[Path, Path, Path]:
    """
    Возвращает:
    - video_games_dir (папка image_processor/video_games)
    - csv_path (video_games.csv)
    - parquet_path (video_games.parquet)
    """
    pkg_dir = Path(__file__).resolve().parent
    vg_dir = pkg_dir / "video_games"
    csv_path = vg_dir / "video_games.csv"
    parquet_path = vg_dir / "video_games.parquet"
    return vg_dir, csv_path, parquet_path


def _import_video_games_analyzer():
    """
    Импортирует анализатор из image_processor/video_games/video_games_analyzer.py.

    Требование для 'чистого' импорта:
      - чтобы папка image_processor/video_games была пакетом, добавь пустой __init__.py
    """
    try:
        from image_processor.video_games.video_games_analyzer import DataPipeline, ParquetManager, DataVisualizer
        return DataPipeline, ParquetManager, DataVisualizer
    except Exception:
        # fallback: если нет __init__.py, пробуем добавить папку в sys.path
        vg_dir, _, _ = _get_video_games_paths()
        sys.path.insert(0, str(vg_dir))
        from video_games.video_games_analyzer import DataPipeline, ParquetManager, DataVisualizer
        return DataPipeline, ParquetManager, DataVisualizer


# ============================================================================
# MODE 1: ASYNC IMAGE PIPELINE
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
        print(f" Results saved to: {os.path.abspath(out_dir)}")

    except KeyboardInterrupt:
        print("\n\n Error: Program interrupted by user")

    except Exception as e:
        logger.error(f"Error in async pipeline: {e}")
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MODE 2: DEMO (без реальной обработки)
# ============================================================================

def run_image_processing_demo() -> None:
    """Запускает демонстрацию API/классов (без запуска пайплайна)."""
    print_header("Mode 2: Image Processing Demo")

    try:
        api_key, api_type = _select_api_source()

        processor = AsyncImageProcessor(api_key, api_type)

        print(f"\n[OK] Initialized {api_type.upper()} API Processor")
        print(f" Base URL: {processor.baseurl}")

        print("\n" + "=" * 70)
        print(" DEMO: Creating Image Metadata")
        print("=" * 70)

        metadata = ImageMetadata(index=1, url="http://example.com/image.jpg", breed="Sample Breed")
        print("\n[OK] Created metadata:")
        print(f" Index: {metadata.index}")
        print(f" URL: {metadata.url}")
        print(f" Breed: {metadata.breed}")

        print("\n" + "=" * 70)
        print(" DEMO: ImageProcessing methods")
        print("=" * 70)

        processor_engine = ImageProcessing()
        print("\n[OK] Initialized ImageProcessing engine")
        print(f" Optimization enabled: {processor_engine.optimization_enabled}")

        print("\n[INFO] Available methods:")
        print(" - RGB to Grayscale conversion")
        print(" - Edge detection (Canny-like pipeline)")
        print(" - Convolution operations")
        print(" - Gamma correction")

        print("\n" + "=" * 70)
        print(" [OK] Demo completed successfully!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n Error: Program interrupted by user")

    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MODE 3: TESTS
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
            print(" [OK] All tests passed successfully!")
        else:
            print(" [ERROR] Tests failed:")
            print(f" Failures: {len(result.failures)}")
            print(f" Errors: {len(result.errors)}")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MODE 4: VIDEO GAMES (CSV, generators)
# ============================================================================

def run_video_games_csv() -> None:
    print_header("Mode 4: Video Games Analyzer (CSV)")

    try:
        DataPipeline, _, DataVisualizer = _import_video_games_analyzer()

        vg_dir, default_csv, _ = _get_video_games_paths()
        print(f" Default folder: {vg_dir}")

        csv_in = input(f"\nCSV path (default: {default_csv}): ").strip()
        csv_path = Path(csv_in) if csv_in else default_csv

        chunksize_str = input("CSV chunksize (default 1000): ").strip()
        chunksize = int(chunksize_str) if chunksize_str else 1000

        pipeline = DataPipeline(csv_path=str(csv_path), chunksize=chunksize)

        # Каждый анализ "съедает" генератор, поэтому создаём цепочку заново
        def make_gen():
            gen = pipeline.read_csv_generator()
            gen = pipeline.filter_valid_data(gen)
            return gen

        sales_df = pipeline.analyze_sales_by_year(make_gen())
        _, ci_df = pipeline.analyze_publisher_variance(make_gen())
        rating_df = pipeline.count_games_by_rating(make_gen())

        plot = input("\nPlot charts? (y/n): ").strip().lower()
        if plot == "y":
            DataVisualizer.plot_sales_by_year(sales_df)
            DataVisualizer.plot_publisher_variance(ci_df)
            DataVisualizer.plot_rating_trends(rating_df)

        print("\n[OK] Video games CSV analysis finished.")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MODE 5: PARQUET (convert + speed + correlation)
# ============================================================================

def run_video_games_parquet() -> None:
    print_header("Mode 5: Video Games Parquet Analysis")

    try:
        _, ParquetManager, DataVisualizer = _import_video_games_analyzer()

        vg_dir, default_csv, default_parquet = _get_video_games_paths()
        print(f" Default folder: {vg_dir}")

        csv_in = input(f"\nCSV path (default: {default_csv}): ").strip()
        csv_path = Path(csv_in) if csv_in else default_csv

        pq_in = input(f"Parquet path (default: {default_parquet}): ").strip()
        parquet_path = Path(pq_in) if pq_in else default_parquet

        pm = ParquetManager(csv_path=str(csv_path), parquet_path=str(parquet_path))
        pm.csv_to_parquet()

        speeds = pm.compare_read_speed()
        corr_df = pm.calculate_correlation()

        plot = input("\nPlot Parquet charts? (y/n): ").strip().lower()
        if plot == "y":
            DataVisualizer.plot_speed_comparison(speeds)
            DataVisualizer.plot_correlation_scatter(corr_df)

        print("\n[OK] Parquet analysis finished.")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN MENU
# ============================================================================

def main() -> None:
    """Главная функция приложения с меню выбора режима."""
    print_header("IMAGE PROCESSING APPLICATION - Main Menu")

    print(" Choose operating mode:\n")

    print(" 1. Run Async Image Pipeline")
    print(" (Download -> Process -> Save)")

    print("\n 2. Image Processing Demo")
    print(" (AsyncImageProcessor, ImageProcessing, API integration)")

    print("\n 3. Run Tests")
    print(" (unittest, package validation)")

    print("\n 4. Video Games Analyzer (CSV)")
    print(" (Generator pipeline, pandas)")

    print("\n 5. Parquet Analysis (Video Games)")
    print(" (CSV -> Parquet, speed, correlation)")

    print("\n 0. Exit")

    choice = get_user_choice("Choose mode", ["0", "1", "2", "3", "4", "5"])

    if choice == "0":
        print("\n Goodbye!\n")
        sys.exit(0)
    elif choice == "1":
        run_async_pipeline()
    elif choice == "2":
        run_image_processing_demo()
    elif choice == "3":
        run_tests()
    elif choice == "4":
        run_video_games_csv()
    elif choice == "5":
        run_video_games_parquet()

    print("\n" + "=" * 70)
    again = input("Do you want to perform another operation? (y/n): ").strip().lower()
    if again == "y":
        main()
    else:
        print("\n Goodbye!\n")


if __name__ == "__main__":
    logger.info("Application started")
    main()
    logger.info("Application finished")
