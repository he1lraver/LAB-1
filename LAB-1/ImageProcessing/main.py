import os
import time
import requests
from dotenv import load_dotenv
from implementation.image_processor import CatImageProcessor, SimpleImageProcessor


def main():
    """Основная функция с обработкой ошибок."""
    try:
        load_dotenv()

        cat_api_key = os.getenv('CAT_API_KEY')
        dog_api_key = os.getenv('DOG_API_KEY')

        if not cat_api_key and not dog_api_key:
            print("❌ API ключи не найдены в .env файле")
            return

        print("🐱🐶 Программа обработки изображений животных")
        print("=" * 50)

        api_choice = input("Выберите API (1 - Cats, 2 - Dogs, 3 - Оба): ").strip()

        try:
            limit_input = input("Количество изображений для обработки (по умолчанию 3): ").strip()
            limit = int(limit_input) if limit_input else 3
        except ValueError:
            print("⚠️ Неверный ввод, используется значение по умолчанию: 3")
            limit = 3

        output_dir = input("Директория для сохранения (по умолчанию animal_images): ").strip()
        output_dir = output_dir if output_dir else "animal_images"

        # ОГРАНИЧЕНИЕ ЛИМИТА ДЛЯ ТЕСТИРОВАНИЯ
        if limit > 5:
            print("⚠️ Для тестирования установлен лимит 5 изображений")
            limit = 5

        # ВЫБОР РЕЖИМА РАБОТЫ
        use_full_processing = input("Использовать полную обработку с гранями? (y/n, по умолчанию y): ").strip().lower()
        use_full_processing = use_full_processing != 'n'

        if use_full_processing:
            # ПОЛНАЯ ОБРАБОТКА С ГРАНЯМИ
            if api_choice == "1" and cat_api_key:
                processor = CatImageProcessor(cat_api_key, "cat")
                processor.run_processing_pipeline(limit, output_dir)
            elif api_choice == "2" and dog_api_key:
                processor = CatImageProcessor(dog_api_key, "dog")
                processor.run_processing_pipeline(limit, output_dir)
            elif api_choice == "3" and cat_api_key and dog_api_key:
                print("\n🐱 Обработка изображений кошек...")
                cat_processor = CatImageProcessor(cat_api_key, "cat")
                cat_processor.run_processing_pipeline(limit, os.path.join(output_dir, "cats"))

                print("\n🐶 Обработка изображений собак...")
                dog_processor = CatImageProcessor(dog_api_key, "dog")
                dog_processor.run_processing_pipeline(limit, os.path.join(output_dir, "dogs"))
            else:
                print("❌ Неверный выбор API или отсутствуют API ключи")
        else:
            # УПРОЩЕННАЯ ЗАГРУЗКА БЕЗ ОБРАБОТКИ
            if api_choice == "1" and cat_api_key:
                processor = SimpleImageProcessor(cat_api_key, "cat")
                processor.run_processing_pipeline(limit, output_dir)
            elif api_choice == "2" and dog_api_key:
                processor = SimpleImageProcessor(dog_api_key, "dog")
                processor.run_processing_pipeline(limit, output_dir)
            elif api_choice == "3" and cat_api_key and dog_api_key:
                print("\n🐱 Загрузка изображений кошек...")
                cat_processor = SimpleImageProcessor(cat_api_key, "cat")
                cat_processor.run_processing_pipeline(limit, os.path.join(output_dir, "cats"))

                print("\n🐶 Загрузка изображений собак...")
                dog_processor = SimpleImageProcessor(dog_api_key, "dog")
                dog_processor.run_processing_pipeline(limit, os.path.join(output_dir, "dogs"))
            else:
                print("❌ Неверный выбор API или отсутствуют API ключи")

    except KeyboardInterrupt:
        print("\n⏹️ Программа прервана пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()