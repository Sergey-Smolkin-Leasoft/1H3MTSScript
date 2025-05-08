# data/data_handler.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import twelvedata as td # Импортируем сам модуль
from twelvedata import TDClient
from twelvedata.exceptions import TwelveDataError # Явный импорт исключения

# Импортируем настройки (предполагаем, что settings.py в config/)
from config import settings # Для доступа к DATA_FETCH_LIMIT, SYMBOL_MAPPING_TWELVE_DATA

class DataHandler:
    """
    Отвечает за загрузку и первичную обработку рыночных данных.
    """
    def __init__(self, api_key: str, symbol_mapping: dict):
        """
        Инициализация обработчика данных.

        Args:
            api_key (str): Ключ API для Twelve Data.
            symbol_mapping (dict): Словарь для маппинга символов бота на тикеры провайдера.
        """
        self.logger = logging.getLogger(f"{__name__}.DataHandler")
        self.api_key = api_key
        self.symbol_mapping_provider = symbol_mapping
        if not self.api_key or self.api_key == 'ВАШ_КЛЮЧ_API_ЗДЕСЬ': # Убедитесь, что эта строка соответствует вашему placeholder в settings.py
            self.logger.warning("Ключ API для Twelve Data не предоставлен или является плейсхолдером. Реальные данные не будут загружены.")
            self.td_client = None
        else:
            try:
                self.td_client = TDClient(apikey=self.api_key)
                self.logger.info("TDClient для Twelve Data успешно инициализирован.")
            except Exception as e: # Ловим более общее исключение при инициализации клиента
                self.logger.error(f"Ошибка инициализации TDClient: {e}", exc_info=True)
                self.td_client = None

    def _fetch_single_timeframe_data(self, symbol_bot: str, timeframe_str: str, limit: int) -> pd.DataFrame | None:
        """
        Загрузка исторических данных для одного символа и таймфрейма с Twelve Data.
        Внутренний метод.
        """
        if not self.td_client:
            self.logger.warning(f"Пропуск загрузки данных для {symbol_bot} ({timeframe_str}): TDClient не инициализирован (нет API ключа?).")
            return None # Возвращаем None, если нет клиента (вместо генерации тестовых данных)

        symbol_provider = self.symbol_mapping_provider.get(symbol_bot)
        if not symbol_provider:
            self.logger.error(f"Символ {symbol_bot} не найден в symbol_mapping. Данные не могут быть загружены.")
            return None

        # Маппинг таймфреймов для Twelve Data
        td_timeframe_map = {
            '1h': '1h',
            '5min': '5min', # Используем 5min для "3m" стратегии
            # Добавьте другие, если необходимо
        }
        td_tf = td_timeframe_map.get(timeframe_str)
        if not td_tf:
            self.logger.error(f"Неподдерживаемый таймфрейм {timeframe_str} для Twelve Data.")
            return None

        retries = 3
        for attempt in range(retries):
            try:
                self.logger.info(f"Запрос данных (Попытка {attempt + 1}/{retries}) для {symbol_provider}, интервал {td_tf}, лимит {limit}")
                ts = self.td_client.time_series(
                    symbol=symbol_provider,
                    interval=td_tf,
                    outputsize=limit,
                    timezone='UTC' # Важно для консистентности
                )
                data = ts.as_pandas()

                if data is None or data.empty:
                    self.logger.warning(f"Получены пустые данные от Twelve Data для {symbol_provider}, {td_tf}.")
                    # Не возвращаем None сразу, даем шанс другим попыткам, если они есть
                    if attempt == retries - 1:
                        return None
                    else:
                        time.sleep(2**(attempt+1)) # Ждем перед следующей попыткой
                        continue


                # Обработка данных
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)

                rename_map = {}
                for col_orig in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    for col_var in [col_orig, col_orig.lower()]: # Проверяем оба регистра
                        if col_var in data.columns:
                            rename_map[col_var] = col_orig.lower() # Приводим к нижнему регистру
                            break
                data.rename(columns=rename_map, inplace=True)

                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in data.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in data.columns]
                    self.logger.error(f"Отсутствуют OHLC колонки: {missing}. Доступные: {data.columns.tolist()}")
                    return None

                for col in required_cols + (['volume'] if 'volume' in data.columns else []):
                    data[col] = pd.to_numeric(data[col], errors='coerce')

                data.dropna(subset=required_cols, inplace=True)
                if data.empty:
                    self.logger.warning("DataFrame пуст после удаления NaN.")
                    return None

                data.sort_index(inplace=True) # Убеждаемся, что данные отсортированы по времени
                self.logger.info(f"Успешно загружены и обработаны данные для {symbol_provider} ({td_tf}): {len(data)} свечей.")
                time.sleep(1) # Небольшая задержка для API
                return data

            except TwelveDataError as e:
                self.logger.warning(f"Ошибка TwelveData API: {e}. Повтор через {2**(attempt+1)} сек.")
                if attempt < retries - 1:
                    time.sleep(2**(attempt+1))
                else:
                    self.logger.error(f"Превышено кол-во попыток для TwelveData API по {symbol_provider}.")
                    return None
            except Exception as e:
                self.logger.error(f"Непредвиденная ошибка при получении данных для {symbol_provider}: {e}", exc_info=True)
                return None # В случае других ошибок также выходим
        return None # Если все попытки не удались

    def fetch_data(self, symbol_bot: str, timeframe_1h_str: str, timeframe_3m_str: str, limit: int = None):
        """
        Загружает данные для обоих основных таймфреймов (1H и 3M/5M).

        Returns:
            tuple: (pd.DataFrame | None, pd.DataFrame | None) -> (data_1h, data_3m)
        """
        effective_limit = limit if limit is not None else settings.DATA_FETCH_LIMIT

        data_1h = self._fetch_single_timeframe_data(symbol_bot, timeframe_1h_str, effective_limit)
        data_3m = self._fetch_single_timeframe_data(symbol_bot, timeframe_3m_str, effective_limit)

        if data_1h is None or data_3m is None:
            self.logger.warning(f"Не удалось загрузить все необходимые данные для {symbol_bot}.")
            # Можно добавить логику, что если одни данные загрузились, а другие нет - это тоже проблема
            # Но пока просто возвращаем то, что есть.

        return data_1h, data_3m

    def update_pdl_pdh(self, data_1h: pd.DataFrame | None):
        """
        Обновление Previous Day Low (PDL) и Previous Day High (PDH).
        Ожидает на вход DataFrame с часовыми данными.
        """
        pdl = None
        pdh = None

        if data_1h is None or data_1h.empty:
            self.logger.warning("data_1h отсутствуют или пусты. Невозможно рассчитать PDL/PDH.")
            return pdl, pdh

        try:
            if not isinstance(data_1h.index, pd.DatetimeIndex):
                self.logger.error("Индекс data_1h не является DatetimeIndex. PDL/PDH не могут быть рассчитаны.")
                return pdl, pdh

            # Гарантируем, что индекс имеет информацию о таймзоне (UTC)
            if data_1h.index.tzinfo is None:
                data_1h.index = data_1h.index.tz_localize('UTC')
            elif data_1h.index.tzinfo != pd.Timestamp("now", tz="UTC").tzinfo: # Сравнение с UTC
                data_1h.index = data_1h.index.tz_convert('UTC')


            latest_data_date = data_1h.index[-1].date()
            unique_dates_in_data = sorted(list(set(data_1h.index.date)))

            if not unique_dates_in_data:
                self.logger.warning("В data_1h нет дат для расчета PDL/PDH.")
                return pdl, pdh

            try:
                latest_date_idx = unique_dates_in_data.index(latest_data_date)
            except ValueError:
                self.logger.warning(f"Дата последнего бара {latest_data_date} не найдена в списке дат. PDL/PDH не обновлены.")
                return pdl, pdh

            if latest_date_idx > 0:
                previous_trading_day_date = unique_dates_in_data[latest_date_idx - 1]
            else:
                self.logger.warning(f"Недостаточно предыдущих торговых дней в data_1h для {latest_data_date}.")
                return pdl, pdh

            previous_day_data = data_1h[data_1h.index.date == previous_trading_day_date]

            if previous_day_data.empty:
                self.logger.warning(f"Нет данных за {previous_trading_day_date} в data_1h для PDL/PDH.")
                return pdl, pdh

            if 'low' not in previous_day_data.columns or 'high' not in previous_day_data.columns:
                self.logger.error("Колонки 'low'/'high' отсутствуют в previous_day_data.")
                return pdl, pdh

            pdl = previous_day_data['low'].min()
            pdh = previous_day_data['high'].max()
            # Форматированный вывод цены с учетом point_size
            instr_settings = settings.INSTRUMENT_SETTINGS.get(settings.DEFAULT_SYMBOL, {}) # Возьмем для примера дефолтный
            point_size = instr_settings.get('point_size', 0.0001)
            decimals = 5 if point_size < 0.01 else 2 # Упрощенное определение кол-ва знаков

            self.logger.info(f"PDL ({previous_trading_day_date}) = {pdl:.{decimals}f}, PDH ({previous_trading_day_date}) = {pdh:.{decimals}f}")

        except Exception as e:
            self.logger.error(f"Ошибка при обновлении PDL/PDH: {e}", exc_info=True)
            pdl, pdh = None, None

        return pdl, pdh

