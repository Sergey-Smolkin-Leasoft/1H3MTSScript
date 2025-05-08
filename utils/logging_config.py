# utils/logging_config.py

import logging
import sys # Для sys.stdout
from config import settings # Импортируем настройки для уровней логирования и имени файла

# Словарь для хранения уже настроенных логгеров, чтобы избежать дублирования обработчиков
_configured_loggers = {}

def setup_logger(name: str = "1H3M_TradingBot") -> logging.Logger:
    """
    Настраивает и возвращает экземпляр логгера.
    Предотвращает дублирование обработчиков для одного и того же имени логгера.

    Args:
        name (str, optional): Имя логгера. По умолчанию "1H3M_TradingBot".

    Returns:
        logging.Logger: Настроенный экземпляр логгера.
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    
    # Устанавливаем общий минимальный уровень для самого логгера.
    # Обработчики могут иметь более высокие уровни.
    # Если LOG_LEVEL_FILE это INFO, а LOG_LEVEL_CONSOLE это WARNING,
    # то логгер должен быть INFO, чтобы сообщения INFO дошли до файлового обработчика.
    overall_min_level = min(settings.LOG_LEVEL_FILE, settings.LOG_LEVEL_CONSOLE)
    logger.setLevel(overall_min_level)

    # Предотвращаем распространение сообщений на корневой логгер, если он уже настроен иначе
    # logger.propagate = False # Раскомментируйте, если есть проблемы с дублированием от корневого логгера

    # --- Форматтер ---
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- Обработчик для файла ---
    # Проверяем, есть ли уже файловый обработчик с таким же файлом
    # Это более сложная проверка, обычно достаточно проверки по имени логгера выше.
    # Но для чистоты, можно и так, если очень нужно избегать дублирования файла.
    
    # if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(settings.LOG_FILE_NAME) for h in logger.handlers):
    try:
        file_handler = logging.FileHandler(settings.LOG_FILE_NAME, encoding='utf-8', mode='a') # mode 'a' для добавления
        file_handler.setLevel(settings.LOG_LEVEL_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # В случае ошибки при создании файлового логгера, выводим в stderr
        # Это может произойти, например, из-за проблем с правами доступа к файлу
        sys.stderr.write(f"CRITICAL: Could not configure file logger '{settings.LOG_FILE_NAME}': {e}\n")


    # --- Обработчик для консоли ---
    # if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    try:
        stream_handler = logging.StreamHandler(sys.stdout) # Явно указываем sys.stdout
        stream_handler.setLevel(settings.LOG_LEVEL_CONSOLE)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    except Exception as e:
        sys.stderr.write(f"CRITICAL: Could not configure console logger: {e}\n")


    # Сохраняем настроенный логгер
    _configured_loggers[name] = logger
    
    if not logger.handlers:
        # Если по какой-то причине ни один обработчик не был добавлен (например, из-за ошибок выше)
        # добавляем базовый обработчик, чтобы логирование хоть как-то работало
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        logger.warning("Ни один из целевых обработчиков логирования не был успешно настроен. Используется базовая конфигурация.")
    else:
        logger.debug(f"Логгер '{name}' успешно настроен с {len(logger.handlers)} обработчиками.")
        
    return logger

# Пример использования (можно удалить, это для теста)
if __name__ == '__main__':
    # Чтобы этот тест работал, нужно создать временный settings.py в этой же папке или настроить пути
    # Для теста создадим "фейковые" настройки settings:
    class FakeSettings:
        LOG_FILE_NAME = "test_util_log.log"
        LOG_LEVEL_FILE = logging.DEBUG
        LOG_LEVEL_CONSOLE = logging.INFO
    
    settings = FakeSettings() # Переопределяем импортированные settings для теста

    logger = setup_logger("TestLogger")
    logger.debug("Это тестовое debug сообщение.")
    logger.info("Это тестовое info сообщение.")
    logger.warning("Это тестовое warning сообщение.")
    logger.error("Это тестовое error сообщение.")
    logger.critical("Это тестовое critical сообщение.")

    logger2 = setup_logger("TestLogger") # Повторный вызов должен вернуть тот же экземпляр
    assert logger is logger2
    logger2.info("Сообщение от того же логгера TestLogger (проверка на дублирование обработчиков).")

    another_logger = setup_logger("AnotherModuleLogger")
    another_logger.info("Сообщение от другого логгера.")