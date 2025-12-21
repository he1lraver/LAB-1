# Image Processing Project (+ Video Games Analyzer)

Проект `image_processor` — это консольное приложение с меню, которое объединяет:
- асинхронную загрузку изображений из TheCatAPI / TheDogAPI, их обработку и сохранение на диск;
- анализ датасета `video_games.csv` и работу с Parquet (конвертация, сравнение скорости чтения, корреляции, графики).

## Возможности

### 1) Async Image Pipeline
- Получение списка URL изображений и метаданных (порода) из API.
- Асинхронная загрузка изображений (`aiohttp`) и преобразование в `numpy` через `PIL`.
- CPU-параллельная обработка (несколько процессов) и сохранение результатов в PNG.

### 2) Video Games Analyzer (CSV + Parquet)
- Генераторное чтение CSV чанками (`pandas.read_csv(..., chunksize=...)`). 
- Аналитика: продажи по годам, дисперсия оценок издателей (95% CI), количество игр по рейтингам E/T/M.
- Parquet: CSV → Parquet, сравнение скорости чтения CSV vs Parquet, корреляция `Review Score` vs `Sales`.
- Визуализация результатов через `matplotlib`.

## Установка

### Вариант A (рекомендуется): установка как пакета
1. Создай виртуальное окружение и активируй его.
2. Установи проект в editable-режиме:
pip install -e .


3. Убедись, что в `setup.py` добавлены зависимости для анализатора видеоигр:
- `pandas`, `matplotlib`, `pyarrow`

## Настройка API ключей (для режима изображений)

Создай `.env` в корне проекта и добавь хотя бы один ключ:
CAT_API_KEY=your_key_here
DOG_API_KEY=your_key_here


Получить ключи можно на сайтах TheCatAPI / TheDogAPI.

## Запуск

### Запуск через модуль (как в проекте)
py -m image_processor

После запуска доступно меню:
1. **Run Async Image Pipeline** (download → process → save) 
2. **Image Processing Demo** (демо классов без полного пайплайна) 
3. **Run Tests** (unittest)
4. **Video Games Analyzer (CSV)**
5. **Parquet Analysis (Video Games)**

## Данные video_games

По умолчанию приложение ожидает файлы в папке:
`image_processor/video_games/` (например `video_games.csv` и создаваемый `video_games.parquet`). [file:222]

## Структура проекта (логическая)

image_processor/
main.py
logging_config.py
implementation/
async_image_processor.py
async_pipeline.py
image_processing.py
parallel_convolution.py
video_games/
video_games_analyzer.py
video_games.csv
video_games.parquet (создаётся автоматически)
setup.py
README.md
.env


## Требования

- Python 3.8+ (рекомендуется).
- Для video games анализа: `pandas`, `matplotlib`, `pyarrow`.
- Для image pipeline: `numpy`, `scipy`, `Pillow`, `aiohttp`, `aiofiles`, `python-dotenv` (и др. зависимости проекта).

