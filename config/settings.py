# config/settings.py

import os
import logging
from dotenv import load_dotenv

# --- Загрузка переменных окружения ---
# Загружает переменные из файла .env в корне проекта (если он есть)
load_dotenv()

# --- Настройки API ---
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY', 'ВАШ_КЛЮЧ_API_ЗДЕСЬ') # Замените на ваш ключ или оставьте получение из .env

# --- Общие настройки бота ---
DEFAULT_SYMBOL = 'EURUSD' # Символ по умолчанию при запуске

# Маппинг символов для провайдера данных (Twelve Data)
# Ключ - символ, используемый в боте; Значение - тикер у провайдера
SYMBOL_MAPPING_TWELVE_DATA = {
    'EURUSD': 'EUR/USD',
    'GBPUSD': 'GBP/USD',
    'XAUUSD': 'XAU/USD',
    'GER40': 'DEU40EUR' # Важно: Уточните точный тикер для GER40/DAX у Twelve Data (может быть DAX, GER40, DE30EUR, DEU40EUR и т.д.)
    # Добавьте другие символы при необходимости
}

# Настройки для конкретных инструментов
# Ключ - символ, используемый в боте
INSTRUMENT_SETTINGS = {
    'EURUSD': {
        'max_target_points': 250,  # Максимальная цель в пунктах
        'point_size': 0.0001      # Размер пункта
    },
    'GBPUSD': {
        'max_target_points': 200,
        'point_size': 0.0001
    },
    'XAUUSD': {
        'max_target_points': 200,
        'point_size': 0.1         # Для золота пункт обычно 0.1
    },
    'GER40': {
        'max_target_points': 400,
        'point_size': 1            # Для индексов пункт часто равен 1.0
    }
    # Добавьте настройки для других символов
}

# --- Настройки таймфреймов ---
TIMEFRAME_1H = '1h'      # Основной таймфрейм для контекста и фракталов
TIMEFRAME_3M = '5min'    # Младший таймфрейм для входа (используем 5min как доступный у Twelve Data)
                         # Если нужен именно 3min, потребуется агрегация или другой провайдер

# Таймфреймы для анализа SSL/BSL (ключ: имя, значение: кол-во часов)
ANALYSIS_TIMEFRAMES = {
    '24h': 24,
    '1w': 168,  # 7 * 24
    '1m': 720   # Приблизительно 30 * 24
}

# --- Настройки торговых сессий (время UTC) ---
SESSIONS = {
    'asia':      {'start': 1, 'end': 9},  # 01:00 - 08:59 UTC
    'frankfurt': {'start': 7, 'end': 15}, # 07:00 - 14:59 UTC
    'london':    {'start': 8, 'end': 16}, # 08:00 - 15:59 UTC
    'newyork':   {'start': 13, 'end': 21} # 13:00 - 20:59 UTC
}

# --- Настройки данных ---
DATABASE_NAME = 'trading_stats.db' # Имя файла базы данных SQLite
DATA_FETCH_LIMIT = 500             # Количество свечей для загрузки за раз
DATA_FETCH_INTERVAL_SECONDS = 60   # Интервал ожидания при ошибках загрузки или в цикле (пример)

# --- Настройки анализа ---
FRACTAL_WINDOW = 2              # Окно для идентификации фракталов (2 свечи слева/справа)
SMA_WINDOW = 48                 # Период SMA для определения контекста
HORIZONTAL_TREND_WINDOW = 48    # Окно для детекции горизонтального тренда (в барах)
MIN_RR_RATIO = 1.3              # Минимальное соотношение Risk/Reward для входа

# --- Настройки логирования ---
LOG_FILE_NAME = 'trading_bot.log'
LOG_LEVEL_FILE = logging.INFO      # Уровень логирования в файл
LOG_LEVEL_CONSOLE = logging.WARNING # Уровень логирования в консоль

# --- Настройки UI и Визуализации ---
CHART_SAVE_PATH = './charts/' # Папка для сохранения графиков (убедитесь, что она существует)

# --- Проверка наличия ключа API ---
if not TWELVE_DATA_API_KEY or TWELVE_DATA_API_KEY == 'ВАШ_КЛЮЧ_API_ЗДЕСЬ':
    print(f"{Fore.RED}ПРЕДУПРЕЖДЕНИЕ: Ключ Twelve Data API не установлен в .env или config/settings.py.")
    print(f"{Fore.YELLOW}Бот будет использовать генерацию тестовых данных или может не работать.")
    # Можно добавить sys.exit(), если ключ обязателен для работы