"""
logging_config.py

Модуль конфигурации логирования для приложения обработки изображений.

Функции:
    - setup_logging(log_file='app.log') - инициализирует логгер
    - get_logger() - возвращает логгер

Логирование осуществляется на двух уровнях:
    1. Файл app.log (DEBUG) - подробные логи с временем и местом вызова
    2. Консоль (INFO) - краткие сообщения о важных событиях
"""

import logging
import sys
from logging.handlers import RotatingFileHandler


_logger = None


def setup_logging(log_file: str = "app.log") -> logging.Logger:
    """
    Настраивает логгер приложения с двумя обработчиками:
    - RotatingFileHandler для файла app.log (DEBUG уровень, подробный формат)
    - StreamHandler для консоли (INFO уровень, краткий формат)

    Args:
        log_file (str): Имя файла логов. По умолчанию "app.log"

    Returns:
        logging.Logger: Настроенный логгер
    """
    global _logger

    # Создаём логгер
    _logger = logging.getLogger("image_processor")
    _logger.setLevel(logging.DEBUG)

    if _logger.handlers:
        for h in list(_logger.handlers):
            try:
                h.flush()
                h.close()
            finally:
                _logger.removeHandler(h)

    # ========== ОБРАБОТЧИК ФАЙЛА (app.log) ==========
    # Формат: время - имя логгера - уровень - [файл:строка] - сообщение
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,  # 2 МБ
        backupCount=5,  # Хранить 5 файлов
        encoding='utf-8'  # UTF-8 для поддержки Unicode в файле
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    _logger.addHandler(file_handler)

    # ========== ОБРАБОТЧИК КОНСОЛИ ==========
    # Формат: уровень - сообщение (краткий, без деталей)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt='%(levelname)-8s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    _logger.addHandler(console_handler)

    return _logger


def get_logger() -> logging.Logger:
    """
    Возвращает существующий логгер или инициализирует новый.

    Returns:
        logging.Logger: Логгер приложения
    """
    global _logger
    if _logger is None:
        setup_logging()
    return _logger
