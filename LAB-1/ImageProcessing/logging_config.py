# -*- coding: utf-8 -*-
"""
Модуль логирования для приложения обработки изображений.

Настраивает:
1. Файловое логирование (DEBUG уровень, подробные логи с временем и файлом)
2. Консольное логирование (INFO уровень, краткие логи)

Использование:
    from logging_config import setup_logging
    logger = setup_logging()
    logger.debug("Детальное сообщение")
    logger.info("Информационное сообщение")
    logger.error("Сообщение об ошибке")
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging(log_file: str = "app.log", level_file: int = logging.DEBUG, 
                  level_console: int = logging.INFO) -> logging.Logger:
    """
    Настраивает и возвращает логгер приложения.
    
    Args:
        log_file: Путь к файлу логов (по умолчанию 'app.log')
        level_file: Уровень логирования в файл (по умолчанию DEBUG)
        level_console: Уровень логирования в консоль (по умолчанию INFO)
    
    Returns:
        logging.Logger: Настроенный объект логгера
    
    Пример:
        logger = setup_logging()
        logger.info("Приложение запущено")
        logger.debug("Детальная информация")
    """
    
    # Создаем логгер
    logger = logging.getLogger('app_logger')
    
    # Устанавливаем уровень логирования для логгера (самый низкий из всех хендлеров)
    logger.setLevel(logging.DEBUG)
    
    # Очищаем существующие хендлеры (на случай если функция вызывается несколько раз)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
        
    # ════════════════════════════════════════════════════════════════════
    # ФАЙЛОВЫЙ ХЕНДЛЕР (DEBUG - подробные логи с временем, файлом и строкой)
    # ════════════════════════════════════════════════════════════════════
    
    # Удаляем старый файл логов если существует (для чистого запуска)
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except OSError:
            pass  # Файл может быть заблокирован, не критично
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level_file)
    
    # Формат для файла: время | уровень | модуль:строка | функция | сообщение
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # ════════════════════════════════════════════════════════════════════
    # КОНСОЛЬНЫЙ ХЕНДЛЕР (INFO - краткие логи без лишних подробностей)
    # ════════════════════════════════════════════════════════════════════
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level_console)
    
    # Формат для консоли: [УРОВЕНЬ] сообщение
    console_formatter = logging.Formatter(
        fmt='[%(levelname)-8s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # ════════════════════════════════════════════════════════════════════
    # ДОБАВЛЯЕМ ХЕНДЛЕРЫ К ЛОГГЕРУ
    # ════════════════════════════════════════════════════════════════════
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Логируем информацию о начале логирования
    logger.debug(f"═" * 70)
    logger.debug(f"Логирование инициализировано в {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Файл логов: {os.path.abspath(log_file)}")
    logger.debug(f"Уровень файлового логирования: {logging.getLevelName(level_file)}")
    logger.debug(f"Уровень консольного логирования: {logging.getLevelName(level_console)}")
    logger.debug(f"═" * 70)
    
    return logger


def get_logger(name: str = 'app_logger') -> logging.Logger:
    """
    Получает логгер приложения.
    
    Используется в других модулях для получения уже настроенного логгера.
    
    Args:
        name: Имя логгера (обычно 'app_logger')
    
    Returns:
        logging.Logger: Логгер приложения
    
    Пример:
        from logging_config import get_logger
        logger = get_logger()
        logger.info("Какое-то сообщение")
    """
    return logging.getLogger(name)


# Поддержка для быстрого логирования
if __name__ == "__main__":
    # Пример использования модуля
    logger = setup_logging()
    
    logger.debug("Это DEBUG сообщение (видно только в файле app.log)")
    logger.debug("Содержит подробную информацию: файл, строка, функция, время")
    
    logger.info("Это INFO сообщение (видно в файле и консоли)")
    logger.info("Запуск приложения завершен")
    
    logger.warning("Это WARNING сообщение (предупреждение)")
    logger.error("Это ERROR сообщение (ошибка)")
    
    print("\n✓ Логирование настроено успешно!")
    print(f"✓ Файл логов: app.log")
    print(f"✓ Проверьте файл app.log для просмотра DEBUG логов")
