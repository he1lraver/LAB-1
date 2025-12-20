"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений.
Реализован консольный интерфейс для применения различных методов обработки к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- обнаружение окружностей (circles)

Запуск:
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]

Аргументы:
    метод: edges | corners | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата (по умолчанию: <имя_входного_файла>_result.png)

Пример:
    python main.py edges test_images/NotreDame.jpg
    python main.py corners test_images/chessboard.jpg -o corners_result.png

Автор: [Ваше имя]
"""

import argparse
import os
import sys
<<<<<<< Updated upstream

import cv2

from implementation.image_processing import ImageProcessing


def check_image_file(filepath: str) -> bool:
    """
    Проверяет, что файл существует и является корректным изображением.
=======
import asyncio
from dotenv import load_dotenv


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
# РЕЖИМ 1: ОБРАБОТКА ИЗОБРАЖЕНИЙ ЖИВОТНЫХ
# ============================================================================

def run_image_processing():
    """Запускает режим обработки изображений животных."""
    print_header("Mode 1: Processing Animal Images")
>>>>>>> Stashed changes
    
    Args:
        filepath: Путь к файлу изображения
        
    Returns:
        bool: True если файл корректен, False в противном случае
    """
    if not os.path.exists(filepath):
        print(f"❌ Ошибка: файл '{filepath}' не существует")
        print("Проверьте путь к файлу и его наличие в директории")
        return False
    
    if not os.path.isfile(filepath):
        print(f"❌ Ошибка: '{filepath}' не является файлом")
        return False
    
<<<<<<< Updated upstream
    # Проверяем расширение файла
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    if not filepath.lower().endswith(valid_extensions):
        print(f"❌ Ошибка: неподдерживаемый формат файла")
        print(f"Поддерживаемые форматы: {', '.join(valid_extensions)}")
        return False
    
    # Пробуем загрузить изображение
    image = cv2.imread(filepath)
    if image is None:
        print(f"❌ Ошибка: не удалось загрузить изображение '{filepath}'")
        print("Файл может быть поврежден или иметь неподдерживаемый формат")
        return False
    
    return True


def list_available_images(directory: str = ".") -> None:
    """
    Показывает доступные изображения в указанной директории.
    
    Args:
        directory: Путь к директории для поиска изображений
    """
    print(f"\n📁 Поиск изображений в: {os.path.abspath(directory)}")
    
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                full_path = os.path.join(root, file)
                image_files.append(full_path)
    
    if image_files:
        print("✅ Найдены изображения:")
        for img_file in image_files:
            # Пробуем получить размер изображения
            img = cv2.imread(img_file)
            if img is not None:
                size_info = f"{img.shape[1]}x{img.shape[0]}"
            else:
                size_info = "ошибка загрузки"
            
            file_size = os.path.getsize(img_file) / 1024  # в КБ
            print(f"  📷 {img_file} ({size_info}, {file_size:.1f} КБ)")
    else:
        print("  ❌ Изображения не найдены")
    
    # Показываем поддиректории
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if subdirs:
        print(f"\n📂 Поддиректории: {', '.join(subdirs)}")


def suggest_image_paths() -> None:
    """Предлагает возможные пути к изображениям."""
    possible_paths = [
        "test_images/NotreDame.jpg",
        "test_images/billiard.png", 
        "test_images/chessboard.jpg",
        "NotreDame.jpg",
        "test.jpg",
        "sample_image.jpg"
    ]
    
    print("\n💡 Возможные пути к изображениям:")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"  ✅ {path} - существует")
        else:
            print(f"  ❌ {path} - не найден")


def main() -> None:
    """Основная функция приложения."""
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing.",
    )
    parser.add_argument(
        "method",
        choices=["edges", "corners", "circles"],
        help="Метод обработки: edges (границы), corners (углы), circles (окружности)",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению (например: test_images/NotreDame.jpg)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="Показать доступные изображения в текущей директории и поддиректориях",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Протестировать все методы на всех доступных изображениях",
    )

    args = parser.parse_args()

    # Если запрошен список изображений
    if args.list_images:
        list_available_images()
        suggest_image_paths()
        return

    # Режим тестирования всех методов на всех изображениях
    if args.test_all:
        test_all_methods()
        return

    # Проверяем существование и корректность файла
    if not check_image_file(args.input):
        print(f"\n🔍 Поиск изображений...")
        list_available_images()
        suggest_image_paths()
        print(f"\n💡 Пример использования: python main.py edges test_images/NotreDame.jpg")
        sys.exit(1)

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ Критическая ошибка: не удалось загрузить изображение {args.input}")
        sys.exit(1)

    print(f"✅ Изображение загружено: {image.shape[1]}x{image.shape[0]} пикселей")

    processor = ImageProcessing()

    # Выбор метода
    try:
        print(f"🔧 Применяем метод: {args.method}")
        if args.method == "edges":
            print("Выполняется обнаружение границ...")
            result = processor.edge_detection(image)
        elif args.method == "corners":
            print("Выполняется обнаружение углов...")
            result = processor.corner_detection(image)
        elif args.method == "circles":
            print("Выполняется обнаружение окружностей...")
            result = processor.circle_detection(image)
        else:
            print("❌ Ошибка: неизвестный метод")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка при обработке изображения: {e}")
        sys.exit(1)

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_{args.method}_result.png"

    # Создаем директорию для результата если её нет
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", 
                exist_ok=True)

    # Сохранение результата
    try:
        success = cv2.imwrite(output_path, result)
        if success:
            print(f"✅ Результат сохранён в: {output_path}")
            
            # Показываем размер исходного и результирующего изображения
            original_size = os.path.getsize(args.input) / 1024  # в КБ
            result_size = os.path.getsize(output_path) / 1024  # в КБ
            print(f"📊 Размер исходного файла: {original_size:.1f} КБ")
            print(f"📊 Размер результирующего файла: {result_size:.1f} КБ")
        else:
            print(f"❌ Ошибка: не удалось сохранить результат в {output_path}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка при сохранении результата: {e}")
        sys.exit(1)


def test_all_methods():
    """Тестирует все методы на всех доступных изображениях."""
    print("🧪 Запуск комплексного тестирования всех методов...")
    
    test_images = [
        "test_images/NotreDame.jpg",
        "test_images/billiard.png", 
        "test_images/chessboard.jpg"
    ]
    
    methods = ["edges", "corners", "circles"]
    
    processor = ImageProcessing()
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ Пропускаем: {image_path} - не существует")
            continue
            
        print(f"\n{'='*50}")
        print(f"📷 Обработка: {image_path}")
        print(f"{'='*50}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Ошибка загрузки: {image_path}")
            continue
            
        print(f"Размер изображения: {image.shape[1]}x{image.shape[0]}")
        
        for method in methods:
            try:
                print(f"\n🔧 Метод: {method}...")
                
                if method == "edges":
                    result = processor.edge_detection(image)
                elif method == "corners":
                    result = processor.corner_detection(image)
                elif method == "circles":
                    result = processor.circle_detection(image)
                
                # Сохраняем результат
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_{method}_result.png"
                cv2.imwrite(output_path, result)
                print(f"✅ Результат сохранен: {output_path}")
                
            except Exception as e:
                print(f"❌ Ошибка при обработке {method}: {e}")
    
    print(f"\n🎉 Тестирование завершено!")
=======
    # Проверка API ключей
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
        if cat_api_key and dog_api_key:
            api_options["3"] = "Cats and Dogs"
        
        print_menu("Select image source", api_options)
        api_choice = get_user_choice("Choose source", list(api_options.keys()))
        
        # Количество изображений
        print("\n Image count for processing:")
        print("  (Recommended 1-3)")
        
        while True:
            try:
                limit_str = input("  Enter count (default 2): ").strip()
                limit = int(limit_str) if limit_str else 2
                if 1 <= limit <= 10:
                    break
                print("   Error: Enter number between 1 and 10")
            except ValueError:
                print("   Error: Enter a number")
        
        # Выбор операций
        print_header("Choose processing operations")
        print(" Available operations:")
        print("  - Addition: Original + Edges")
        print("  - Subtraction: Original - Edges")
        
        operations_menu = {
            "1": "Addition: Original + Edges (custom)",
            "2": "Addition: Original + Edges (library Canny)",
            "3": "Subtraction: Original - Edges (custom)",
            "4": "Subtraction: Original - Edges (library Canny)",
            "5": "All operations",
            "6": "Skip operations"
        }
        
        print_menu("Choose operations", operations_menu)
        
        selected_ops = []
        while True:
            choice = input("\nEnter operation number(s) (comma-separated) or Enter for all: ").strip()
            
            if not choice:
                selected_ops = ["1", "2", "3", "4"]
                break
            
            if choice == "5":
                selected_ops = ["1", "2", "3", "4"]
                break
            
            if choice == "6":
                selected_ops = []
                break
            
            choices = [c.strip() for c in choice.split(",")]
            if all(c in operations_menu for c in choices):
                selected_ops = choices
                break
            
            print("  Error: Invalid choice. Try again.")
        
        if selected_ops:
            print(f"\n[OK] Selected operations: {len(selected_ops)}")
            for op_id in selected_ops:
                print(f"  - {operations_menu.get(op_id, 'Unknown operation')}")
        else:
            print("\n[INFO] Operations not selected. Saving only original and edges.")
        
        # Директория для сохранения
        output_dir = input("\n Directory for saving (default './results'): ").strip()
        output_dir = output_dir if output_dir else "./results"
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"[OK] Directory: {os.path.abspath(output_dir)}")
        
        # Обработка изображений
        print_header(f"Loading and processing {limit} image(s)")
        
        if api_choice in ["1", "3"] and cat_api_key:
            print("Processing cat images...")
            cat_processor = CatImageProcessor(cat_api_key, "cat")
            cat_images = cat_processor.download_images(limit=limit)
            
            cat_output_dir = os.path.join(output_dir, "cats") if api_choice == "3" else output_dir
            
            for idx, cat_image in enumerate(cat_images, 1):
                _process_image_with_operations(
                    cat_image,
                    cat_output_dir,
                    idx,
                    selected_ops
                )
        
        if api_choice in ["2", "3"] and dog_api_key:
            print("\nProcessing dog images...")
            dog_processor = CatImageProcessor(dog_api_key, "dog")
            dog_images = dog_processor.download_images(limit=limit)
            
            dog_output_dir = os.path.join(output_dir, "dogs") if api_choice == "3" else output_dir
            
            for idx, dog_image in enumerate(dog_images, 1):
                _process_image_with_operations(
                    dog_image,
                    dog_output_dir,
                    idx,
                    selected_ops
                )
        
        print_header("[OK] Image processing completed")
        print(f" Results saved to: {os.path.abspath(output_dir)}\n")
        
    except KeyboardInterrupt:
        print("\n\n  Error: Program interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()



def _process_image_with_operations(cat_image, output_dir: str, breed_index: int, 
                                   operations: list) -> None:
    """Обрабатывает изображение и применяет операции."""
    from PIL import Image
    
    breed_safe = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
    
    print(f"\n{'-'*70}")
    print(f"Processing image #{breed_index}: {cat_image.breed}")
    print(f"{'-'*70}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохранение оригинального изображения
        print(f"  Saving original image...")
        original_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_original.png")
        Image.fromarray(cat_image.image_data).save(original_path)
        print(f"  [OK] Saved: {original_path}")
        
        # Edge detection - пользовательский метод
        print(f"  Applying custom edge detection...")
        edges_custom = cat_image.custom_edge_detection()
        edges_custom_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_edges_custom.png")
        Image.fromarray(edges_custom).save(edges_custom_path)
        print(f"  [OK] Saved: {edges_custom_path}")
        
        # Edge detection - библиотечный метод
        print(f"  Applying library edge detection (Canny)...")
        edges_library = cat_image.library_edge_detection()
        edges_library_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_edges_library.png")
        Image.fromarray(edges_library).save(edges_library_path)
        print(f"  [OK] Saved: {edges_library_path}")
        
        # Арифметические операции
        if operations:
            print(f"\n   Applying arithmetic operations...")
            
            if "1" in operations:
                print(f"     Original + Edges (custom)")
                sum_custom = cat_image + edges_custom
                sum_custom_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sum_edges_custom.png")
                Image.fromarray(sum_custom.image_data).save(sum_custom_path)
                print(f"     [OK] Saved: {sum_custom_path}")
            
            if "2" in operations:
                print(f"     Original + Edges (library Canny)")
                sum_library = cat_image + edges_library
                sum_library_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sum_edges_library.png")
                Image.fromarray(sum_library.image_data).save(sum_library_path)
                print(f"     [OK] Saved: {sum_library_path}")
            
            if "3" in operations:
                print(f"    Original - Edges (custom)")
                sub_custom = cat_image - edges_custom
                sub_custom_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sub_edges_custom.png")
                Image.fromarray(sub_custom.image_data).save(sub_custom_path)
                print(f"     [OK] Saved: {sub_custom_path}")
            
            if "4" in operations:
                print(f"    Original - Edges (library Canny)")
                sub_library = cat_image - edges_library
                sub_library_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sub_edges_library.png")
                Image.fromarray(sub_library.image_data).save(sub_library_path)
                print(f"    [OK] Saved: {sub_library_path}")
        else:
            print(f"\n    [INFO] Operations not selected. Saved only original and edges.")
        
        print(f"\n   [OK] Processing image #{breed_index} completed.")
        
    except Exception as e:
        print(f"  Error processing image #{breed_index}: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# РЕЖИМ 2: АНАЛИЗ ДАННЫХ ВИДЕОИГР
# ============================================================================

def run_video_games_analyzer():
    """Запускает режим анализа данных видеоигр."""
    print_header("Mode 2: Video Game Data Analysis")
    
    try:
        video_games_dir = os.path.join(os.path.dirname(__file__), 'video_games')
        if video_games_dir not in sys.path:
            sys.path.insert(0, video_games_dir)
        
        from video_games.video_games_analyzer import (
            DataPipeline, ParquetManager, DataVisualizer
        )
        
    except ImportError as e:
        print(f" Error: {e}")
        print("\n Folder structure:")
        print("    project/")
        print("    ├── main.py")
        print("    └── video_games/")
        print("        ├── __init__.py")
        print("        ├── video_games_analyzer.py")
        print("        └── video_games.csv")
        return
    
    video_games_dir = os.path.join(os.path.dirname(__file__), 'video_games')
    csv_path = os.path.join(video_games_dir, 'video_games.csv')
    
    if not os.path.exists(csv_path):
        print(f" Error: File '{csv_path}' not found!")
        print("    Please make sure file is in video_games/ folder")
        print(f"\n    Current path: {os.path.abspath(csv_path)}")
        return
    
    parquet_path = os.path.join(video_games_dir, 'video_games.parquet')
    output_dir = 'results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = DataPipeline(csv_path, parquet_path, chunksize=100)
    parquet_mgr = ParquetManager(csv_path, parquet_path)
    
    print(" [OK] All files found. Starting analysis...\n")
    
    try:
        # ========== ЭТАП 0: Создание Parquet ==========
        print("=" * 70)
        print(" STAGE 0: CREATING PARQUET FILE")
        print("=" * 70)
        parquet_mgr.csv_to_parquet()
        
        # ========== ЗАДАНИЕ 1: Лучший/худший годы ==========
        print("\n" + "=" * 70)
        print(" TASK 1: BEST AND WORST YEARS BY SALES")
        print("=" * 70)
        sales_df = pipeline.analyze_sales_by_year(
            pipeline.filter_valid_data(pipeline.read_csv_generator())
        )
        print(f"\n Results (first 10 years):\n{sales_df.head(10).to_string(index=False)}")
        DataVisualizer.plot_sales_by_year(sales_df, f'{output_dir}/01_sales_by_year.png')
        
        # ========== ЗАДАНИЕ 2: Издатели с разбросом ==========
        print("\n" + "=" * 70)
        print(" TASK 2: PUBLISHERS WITH MOST/LEAST SCORE VARIANCE")
        print("=" * 70)
        stats_df, ci_df = pipeline.analyze_publisher_variance(
            pipeline.filter_valid_data(pipeline.read_csv_generator())
        )
        print(f"\n Results (TOP-6 publishers):\n{ci_df.to_string(index=False)}")
        DataVisualizer.plot_publisher_variance(ci_df, f'{output_dir}/02_publisher_variance.png')
        
        # ========== ЗАДАНИЕ 3: Игры по рейтингу E/T/M ==========
        print("\n" + "=" * 70)
        print(" TASK 3: GAME COUNT BY RATING (E/T/M) PER YEAR")
        print("=" * 70)
        rating_df = pipeline.count_games_by_rating(pipeline.read_csv_generator())
        print(f"\n Results (first 10 years):\n{rating_df.head(10).to_string(index=False)}")
        DataVisualizer.plot_rating_trends(rating_df, f'{output_dir}/03_rating_trends.png')
        
        # ========== ДОП. ЗАДАНИЕ: Корреляция Score vs Sales ==========
        print("\n" + "=" * 70)
        print(" EXTRA TASK: SCORE vs SALES CORRELATION")
        print("=" * 70)
        corr_df = parquet_mgr.calculate_correlation()
        DataVisualizer.plot_correlation_scatter(corr_df, f'{output_dir}/04_correlation.png')
        
        # ========== Демонстрация: Сравнение скорости ==========
        print("\n" + "=" * 70)
        print(" DEMO: CSV vs PARQUET SPEED COMPARISON")
        print("=" * 70)
        speeds = parquet_mgr.compare_read_speed()
        DataVisualizer.plot_speed_comparison(speeds, f'{output_dir}/05_speed_comparison.png')
        
        # ========== ЗАВЕРШЕНИЕ ==========
        print("\n" + "=" * 70)
        print(" [OK] ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f" Results saved to: {os.path.abspath(output_dir)}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n  Error: Program interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()



# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main() -> None:
    """Главная функция приложения с меню выбора режима."""
    
    print_header("MAIN APPLICATION MENU")
    
    print(" Choose operating mode:\n")
    print("  1. Processing Animal Images")
    print("     (API loading, edge detection, addition/subtraction)")
    print("\n  2. Video Game Data Analysis")
    print("     (generators, pipelines, Parquet, graphs)")
    print("\n  3. Advanced Async Image Pipeline (Lab 4)")
    print("     (aiohttp, aiofiles, ProcessPoolExecutor, async generators)")
    print("     FEATURES: API-only, performance metrics, image entropy analysis")
    print("\n  0. Exit")
    
    choice = get_user_choice("Choose mode", ["0", "1", "2", "3"])
    
    if choice == "0":
        print("\n Goodbye!\n")
        sys.exit(0)
    elif choice == "1":
        run_image_processing()
    elif choice == "2":
        run_video_games_analyzer()
    
    # Предложить повторный выбор
    print("\n" + "=" * 70)
    again = input("Do you want to perform another operation? (y/n): ").strip().lower()
    
    if again == 'y':
        main()
    else:
        print("\n Goodbye!\n")
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()