import os
import sys
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
        print(f"  ❌ Пожалуйста, выберите из: {', '.join(valid_choices)}")


# ============================================================================
# РЕЖИМ 1: ОБРАБОТКА ИЗОБРАЖЕНИЙ ЖИВОТНЫХ
# ============================================================================

def run_image_processing():
    """Запускает режим обработки изображений животных."""
    print_header("Режим 1: Обработка Изображений Животных")
    
    from implementation.image_processor import CatImageProcessor
    from PIL import Image
    import time
    
    load_dotenv()
    cat_api_key = os.getenv("CAT_API_KEY")
    dog_api_key = os.getenv("DOG_API_KEY")
    
    # Проверка API ключей
    if not cat_api_key and not dog_api_key:
        print(" ❌ Ошибка: API ключи не найдены в файле .env!")
        print("\n Инструкции по настройке:")
        print("  1. Зарегистрируйтесь на https://thecatapi.com/ или https://thedogapi.com/")
        print("  2. Получите API_KEY")
        print("  3. Создайте файл .env в корне проекта")
        print("  4. Добавьте строки:")
        print("     CAT_API_KEY=ваш_ключ_кошек")
        print("     DOG_API_KEY=ваш_ключ_собак")
        return
    
    print("✅ API ключи загружены успешно.")
    
    try:
        # Выбор источника данных
        api_options = {}
        if cat_api_key:
            api_options["1"] = "Кошки (The Cat API)"
        if dog_api_key:
            api_options["2"] = "Собаки (The Dog API)"
        if cat_api_key and dog_api_key:
            api_options["3"] = "Кошки и Собаки"
        
        print_menu("📸 Выбор источника изображений", api_options)
        api_choice = get_user_choice("Выберите источник", list(api_options.keys()))
        
        # Количество изображений
        print("\n 📊 Количество изображений для обработки:")
        print("  (Рекомендуется 1-3)")
        
        while True:
            try:
                limit_str = input("  Введите количество (по умолчанию 2): ").strip()
                limit = int(limit_str) if limit_str else 2
                if 1 <= limit <= 10:
                    break
                print("   ❌ Пожалуйста, введите число от 1 до 10")
            except ValueError:
                print("   ❌ Пожалуйста, введите число")
        
        # Выбор операций
        print_header("Выбор операций обработки")
        print(" 📝 Доступные операции:")
        print("  • Сложение: Original + Edges")
        print("  • Вычитание: Original - Edges")
        
        operations_menu = {
            "1": "Сложение: Original + Edges (custom)",
            "2": "Сложение: Original + Edges (library Canny)",
            "3": "Вычитание: Original - Edges (custom)",
            "4": "Вычитание: Original - Edges (library Canny)",
            "5": "Все операции",
            "6": "Пропустить операции"
        }
        
        print_menu("Выберите операции", operations_menu)
        
        selected_ops = []
        while True:
            choice = input("\nВведите номер(а) операции (через запятую) или Enter для всех: ").strip()
            
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
            
            print("  ❌ Неверный выбор. Попробуйте снова.")
        
        if selected_ops:
            print(f"\n✅ Выбрано операций: {len(selected_ops)}")
            for op_id in selected_ops:
                print(f"  • {operations_menu.get(op_id, 'Неизвестная операция')}")
        else:
            print("\n📋 Операции не выбраны. Будут сохранены только оригинал и контуры.")
        
        # Директория для сохранения
        output_dir = input("\n📁 Директория для сохранения (по умолчанию './results'): ").strip()
        output_dir = output_dir if output_dir else "./results"
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Директория: {os.path.abspath(output_dir)}")
        
        # Обработка изображений
        print_header(f"Загрузка и обработка {limit} изображения(й)")
        
        if api_choice in ["1", "3"] and cat_api_key:
            print("🐱 Обработка изображений кошек...")
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
            print("\n🐶 Обработка изображений собак...")
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
        
        print_header(" ✅ Обработка изображений завершена")
        print(f" 📁 Результаты сохранены в: {os.path.abspath(output_dir)}\n")
        
    except KeyboardInterrupt:
        print("\n\n  ❌ Программа прервана пользователем")
    except Exception as e:
        print(f"\n ❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


def _process_image_with_operations(cat_image, output_dir: str, breed_index: int, 
                                   operations: list) -> None:
    """Обрабатывает изображение и применяет операции."""
    from PIL import Image
    
    breed_safe = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
    
    print(f"\n{'-'*70}")
    print(f"Обработка изображения #{breed_index}: {cat_image.breed}")
    print(f"{'-'*70}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохранение оригинального изображения
        print(f"  📌 Сохранение оригинального изображения...")
        original_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_original.png")
        Image.fromarray(cat_image.image_data).save(original_path)
        print(f"  ✅ Сохранено: {original_path}")
        
        # Edge detection - пользовательский метод
        print(f"  📌 Применение custom edge detection...")
        edges_custom = cat_image.custom_edge_detection()
        edges_custom_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_edges_custom.png")
        Image.fromarray(edges_custom).save(edges_custom_path)
        print(f"  ✅ Сохранено: {edges_custom_path}")
        
        # Edge detection - библиотечный метод
        print(f"  📌 Применение library edge detection (Canny)...")
        edges_library = cat_image.library_edge_detection()
        edges_library_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_edges_library.png")
        Image.fromarray(edges_library).save(edges_library_path)
        print(f"  ✅ Сохранено: {edges_library_path}")
        
        # Арифметические операции
        if operations:
            print(f"\n   Применение арифметических операций...")
            
            if "1" in operations:
                print(f"     Original + Edges (custom)")
                sum_custom = cat_image + edges_custom
                sum_custom_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sum_edges_custom.png")
                Image.fromarray(sum_custom.image_data).save(sum_custom_path)
                print(f"     ✅ Сохранено: {sum_custom_path}")
            
            if "2" in operations:
                print(f"     Original + Edges (library Canny)")
                sum_library = cat_image + edges_library
                sum_library_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sum_edges_library.png")
                Image.fromarray(sum_library.image_data).save(sum_library_path)
                print(f"     ✅ Сохранено: {sum_library_path}")
            
            if "3" in operations:
                print(f"    Original - Edges (custom)")
                sub_custom = cat_image - edges_custom
                sub_custom_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sub_edges_custom.png")
                Image.fromarray(sub_custom.image_data).save(sub_custom_path)
                print(f"     ✅ Сохранено: {sub_custom_path}")
            
            if "4" in operations:
                print(f"    Original - Edges (library Canny)")
                sub_library = cat_image - edges_library
                sub_library_path = os.path.join(output_dir, f"{breed_index}_{breed_safe}_sub_edges_library.png")
                Image.fromarray(sub_library.image_data).save(sub_library_path)
                print(f"    ✅ Сохранено: {sub_library_path}")
        else:
            print(f"\n    📋 Операции не выбраны. Сохранены только оригинал и контуры.")
        
        print(f"\n   ✅ Обработка изображения #{breed_index} завершена.")
        
    except Exception as e:
        print(f"  ❌ Ошибка при обработке изображения #{breed_index}: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# РЕЖИМ 2: АНАЛИЗ ДАННЫХ ВИДЕОИГР
# ============================================================================

def run_video_games_analyzer():
    """Запускает режим анализа данных видеоигр."""
    print_header("Режим 2: Анализ Данных о Видеоиграх")
    
    try:
        # Добавляем папку video_games в путь поиска модулей
        video_games_dir = os.path.join(os.path.dirname(__file__), 'video_games')
        if video_games_dir not in sys.path:
            sys.path.insert(0, video_games_dir)
        
        # Импортируем модуль анализатора
        from video_games_analyzer import (
            DataPipeline, ParquetManager, DataVisualizer
        )
        
    except ImportError as e:
        print(f" ❌ Ошибка импорта: {e}")
        print("\n 📁 Проверьте структуру папок:")
        print("    project/")
        print("    ├── main.py")
        print("    └── video_games/")
        print("        ├── __init__.py")
        print("        ├── video_games_analyzer.py")
        print("        └── video_games.csv")
        return
    
    # Пути к файлам
    video_games_dir = os.path.join(os.path.dirname(__file__), 'video_games')
    csv_path = os.path.join(video_games_dir, 'video_games.csv')
    
    # Проверка наличия CSV файла
    if not os.path.exists(csv_path):
        print(f" ❌ Ошибка: Файл '{csv_path}' не найден!")
        print("    Пожалуйста, убедитесь, что файл находится в папке video_games/")
        print(f"\n    Текущий путь: {os.path.abspath(csv_path)}")
        return
    
    parquet_path = os.path.join(video_games_dir, 'video_games.parquet')
    output_dir = 'results'
    
    # Создаем директорию для результатов (в корне проекта)
    os.makedirs(output_dir, exist_ok=True)
    
    # Инициализируем компоненты
    pipeline = DataPipeline(csv_path, parquet_path, chunksize=100)
    parquet_mgr = ParquetManager(csv_path, parquet_path)
    
    print(" ✅ Все файлы найдены. Запуск анализа...\n")
    
    try:
        # ========== ЭТАП 0: Создание Parquet ==========
        print("=" * 70)
        print(" ЭТАП 0: СОЗДАНИЕ PARQUET ФАЙЛА")
        print("=" * 70)
        parquet_mgr.csv_to_parquet()
        
        # ========== ЗАДАНИЕ 1: Лучший/худший годы ==========
        print("\n" + "=" * 70)
        print(" ЗАДАНИЕ 1: ЛУЧШИЙ И ХУДШИЙ ГОДЫ ПО ПРОДАЖАМ")
        print("=" * 70)
        sales_df = pipeline.analyze_sales_by_year(
            pipeline.filter_valid_data(pipeline.read_csv_generator())
        )
        print(f"\n 📊 Результат (первые 10 лет):\n{sales_df.head(10).to_string(index=False)}")
        DataVisualizer.plot_sales_by_year(sales_df, f'{output_dir}/01_sales_by_year.png')
        
        # ========== ЗАДАНИЕ 2: Издатели с разбросом ==========
        print("\n" + "=" * 70)
        print(" ЗАДАНИЕ 2: ИЗДАТЕЛИ С НАИБОЛЬШИМ И НАИМЕНЬШИМ РАЗБРОСОМ ОЦЕНОК")
        print("=" * 70)
        stats_df, ci_df = pipeline.analyze_publisher_variance(
            pipeline.filter_valid_data(pipeline.read_csv_generator())
        )
        print(f"\n 📊 Результат (ТОП-6 издателей с экстремальными дисперсиями):\n{ci_df.to_string(index=False)}")
        DataVisualizer.plot_publisher_variance(ci_df, f'{output_dir}/02_publisher_variance.png')
        
        # ========== ЗАДАНИЕ 3: Игры по рейтингу E/T/M ==========
        print("\n" + "=" * 70)
        print(" ЗАДАНИЕ 3: КОЛИЧЕСТВО ИГР ПО РЕЙТИНГУ (E/T/M) ЗА КАЖДЫЙ ГОД")
        print("=" * 70)
        rating_df = pipeline.count_games_by_rating(pipeline.read_csv_generator())
        print(f"\n 📊 Результат (первые 10 лет):\n{rating_df.head(10).to_string(index=False)}")
        DataVisualizer.plot_rating_trends(rating_df, f'{output_dir}/03_rating_trends.png')
        
        # ========== ДОП. ЗАДАНИЕ: Корреляция Score vs Sales ==========
        print("\n" + "=" * 70)
        print(" ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: КОРРЕЛЯЦИЯ SCORE VS SALES")
        print("=" * 70)
        corr_df = parquet_mgr.calculate_correlation()
        DataVisualizer.plot_correlation_scatter(corr_df, f'{output_dir}/04_correlation.png')
        
        # ========== Демонстрация: Сравнение скорости ==========
        print("\n" + "=" * 70)
        print(" ДЕМОНСТРАЦИЯ: СРАВНЕНИЕ СКОРОСТИ CSV VS PARQUET")
        print("=" * 70)
        speeds = parquet_mgr.compare_read_speed()
        DataVisualizer.plot_speed_comparison(speeds, f'{output_dir}/05_speed_comparison.png')
        
        # ========== ЗАВЕРШЕНИЕ ==========
        print("\n" + "=" * 70)
        print(" ✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print(f" 📁 Результаты сохранены в директорию: {os.path.abspath(output_dir)}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n  ❌ Программа прервана пользователем")
    except Exception as e:
        print(f"\n ❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main() -> None:
    """Главная функция приложения с меню выбора режима."""
    
    print_header(" ГЛАВНОЕ МЕНЮ ПРИЛОЖЕНИЯ")
    
    print(" 📋 Выберите режим работы:\n")
    print("  1️⃣  Обработка изображений животных")
    print("     (загрузка с API, edge detection, сложение/вычитание)")
    print("\n  2️⃣  Анализ данных видеоигр")
    print("     (генераторы, пайплайны, Parquet, графики)")
    print("\n  0️⃣  Выход")
    
    choice = get_user_choice("Выберите режим", ["0", "1", "2"])
    
    if choice == "0":
        print("\n👋 До свидания!\n")
        sys.exit(0)
    elif choice == "1":
        run_image_processing()
    elif choice == "2":
        run_video_games_analyzer()
    
    # Предложить повторный выбор
    print("\n" + "=" * 70)
    again = input("Хотите выполнить еще одну операцию? (y/n): ").strip().lower()
    
    if again == 'y':
        main()
    else:
        print("\n👋 До свидания!\n")


if __name__ == "__main__":
    main()