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

import cv2

from implementation.image_processing import ImageProcessing


def check_image_file(filepath: str) -> bool:
    """
    Проверяет, что файл существует и является корректным изображением.
    
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


if __name__ == "__main__":
    main()