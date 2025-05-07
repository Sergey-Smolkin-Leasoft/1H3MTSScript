import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
from ta.trend import SMAIndicator
import logging
from twelvedata import TDClient
from dotenv import load_dotenv
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from database import TradingDatabase
import mplfinance as mpf

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("1H3M_TradingBot")

class TradingBot1H3M:
    """
    Торговый бот, реализующий стратегию 1H3M
    """
    def __init__(self, symbol, timeframe_1h='1h', timeframe_3m='3m', exchange=None):
        """
        Инициализация бота
        
        Parameters:
        symbol (str): Торговый символ (например, 'EURUSD', 'GER40')
        timeframe_1h (str): Временной интервал для часового графика
        timeframe_3m (str): Временной интервал для 3-минутного графика
        exchange: Биржевой объект для торговли (если None, используется тестовый режим)
        """
        self.symbol = symbol
        self.timeframe_1h = timeframe_1h
        self.timeframe_3m = timeframe_3m
        self.exchange = exchange
        
        # Инициализация базы данных
        self.db = TradingDatabase()
        
        # Настройка Twelve Data API
        load_dotenv()
        self.td_api_key = os.getenv('TWELVE_DATA_API_KEY', '9c614fea46d04e3d8c4f3f76b0541ab6')
        self.td = TDClient(apikey=self.td_api_key)
        
        # Маппинг символов для Twelve Data
        self.symbol_mapping = {
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'XAUUSD': 'XAU/USD',
            'GER40': 'DAX'
        }
        
        # Настройки инструментов
        if symbol == 'EURUSD':
            self.max_target_points = 250
            self.point_size = 0.0001
        elif symbol == 'GBPUSD':
            self.max_target_points = 200
            self.point_size = 0.0001
        elif symbol == 'XAUUSD':
            self.max_target_points = 200
            self.point_size = 0.1
        elif symbol == 'GER40':
            self.max_target_points = 400
            self.point_size = 1
        else:
            raise ValueError(f"Неподдерживаемый символ: {symbol}")
        
        # Данные
        self.data_1h = None
        self.data_3m = None
        
        # Информация о сессиях
        self.sessions = {
            'asia': {'start': 1, 'end': 9},          # 01:00-09:00 UTC
            'london': {'start': 8, 'end': 16},       # 08:00-16:00 UTC
            'newyork': {'start': 13, 'end': 21},     # 13:00-21:00 UTC
            'frankfurt': {'start': 7, 'end': 15}     # 07:00-15:00 UTC
        }
        
        # Текущие сигналы
        self.current_context = None  # 'long' или 'short'
        self.fractal_levels = []
        self.skip_conditions = []
        
        # Состояния цены
        self.price_context = None  # Текущее состояние цены
        self.price_context_states = {
            'CLEAR_TREND': 'Четкое направление с Order Flow',
            'INEFFICIENT_DELIVERY': 'Неэффективная доставка без Order Flow',
            'TARGET_APPROACH': 'Приход к таргету',
            'HORIZONTAL_TREND': 'Горизонтальный тренд',
            'CORRECTION': 'Коррекция'
        }
        
        # Состояние реверсала
        self.reversal_state = {
            'confirmed_swipes': 0,  # Количество подтвержденных свипов
            'last_swipe_session': None,  # Последняя сессия свипа
            'important_target_removed': False,  # Флаг снятия важного таргета
            'last_liquidity_pool': None  # Последний большой пул ликвидности
        }
        
        # Дневной лимит (DL)
        self.daily_limit = None
        
        # Параметры для детекции горизонтального тренда
        self.horizontal_trend = {
            'is_horizontal': False,
            'confidence': 0.0,
            'slope': 0.0,
            'r_squared': 0.0,
            'std_dev': 0.0,
            'last_update': None
        }
        
        # Текущая волатильность
        self.current_volatility = None
        
        # Текущие открытые позиции
        self.open_positions = []
        
        # Таймфреймы для анализа
        self.timeframes = {
            '24h': 24,  # Последние 24 часа
            '1w': 168,  # Последняя неделя (7 дней)
            '1m': 720   # Последний месяц (30 дней)
        }
        
        # SSL/BSL уровни
        self.ssl_bsl_levels = []
        
        # Зоны POI/FVG
        self.poi_zones = []
        
        # Флаги для реакции
        self.ssl_bsl_removed = False
        self.poi_fvg_reacted = False
        
        # Получаем текущую цену EURUSD
        try:
            current_price = self.data_3m['close'].iloc[-1]
            logger.info(f"Бот инициализирован для {symbol} с максимальной целью {self.max_target_points} пунктов")
            logger.info(f"Текущая цена EURUSD: {current_price:.5f}")
        except Exception as e:
            logger.info(f"Бот инициализирован для {symbol} с максимальной целью {self.max_target_points} пунктов")
            logger.warning(f"Не удалось получить текущую цену: {e}")

    def fetch_data(self):
        """
        Загрузка исторических данных
        """
        if self.exchange:
            # Реальные данные с биржи
            self.data_1h = self.fetch_historical_data(self.timeframe_1h)
            self.data_3m = self.fetch_historical_data(self.timeframe_3m)
        else:
            # Тестовые данные для разработки
            self.data_1h = self.generate_test_data(self.timeframe_1h)
            self.data_3m = self.generate_test_data(self.timeframe_3m)
        
        logger.info(f"Данные загружены: {len(self.data_1h)} часовых свечей, {len(self.data_3m)} 3-минутных свечей")

    def fetch_historical_data(self, timeframe, limit=500):
        """
        Загрузка исторических данных с Twelve Data
        """
        # Преобразуем временной интервал для Twelve Data
        td_timeframe = {
            '1h': '1H',
            '3m': '3M'
        }[timeframe]
        
        # Получаем данные
        try:
            ts = self.td.time_series(
                symbol=self.symbol_mapping[self.symbol],
                interval=td_timeframe,
                outputsize=limit,
                timezone='UTC'
            )
            data = ts.as_pandas()
            
            # Переименуем столбцы для совместимости
            data = data.rename(columns={
                'datetime': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при получении данных с Twelve Data: {e}")
            raise

    def generate_test_data(self, timeframe):
        """
        Генерация тестовых данных для разработки
        """
        now = datetime.now()
        
        if timeframe == self.timeframe_1h:
            start = now - timedelta(days=10)
            periods = 240  # 10 дней по часу
            freq = '1H'
        else:  # 3m
            start = now - timedelta(days=3)
            periods = 1440  # 3 дня по 3 минуты
            freq = '3min'
        
        dates = pd.date_range(start=start, periods=periods, freq=freq)
        
        # Генерация случайных данных с небольшим трендом
        np.random.seed(42)
        base_price = 1.1000 if self.symbol == 'EURUSD' else 15000  # Базовая цена
        trend = np.cumsum(np.random.normal(0, 0.0002 if self.symbol == 'EURUSD' else 5, len(dates)))
        noise = np.random.normal(0, 0.0005 if self.symbol == 'EURUSD' else 15, len(dates))
        
        # Создание циклической компоненты
        t = np.linspace(0, 4*np.pi, len(dates))
        cycle = 0.001 * np.sin(t) if self.symbol == 'EURUSD' else 30 * np.sin(t)
        
        close_prices = base_price + trend + noise + cycle
        
        # Создание OHLC данных
        high_prices = close_prices + np.abs(np.random.normal(0, 0.0003 if self.symbol == 'EURUSD' else 10, len(dates)))
        low_prices = close_prices - np.abs(np.random.normal(0, 0.0003 if self.symbol == 'EURUSD' else 10, len(dates)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base_price
        
        # Создание DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        return df

    def calculate_volatility(self, data, period=14):
        """
        Расчет волатильности (ATR) для инструмента
        
        Parameters:
        data: DataFrame с ценовыми данными
        period: Период расчета ATR (по умолчанию 14)
        
        Returns:
        float: Текущая волатильность
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]

    def check_horizontal_trend(self, data, window=48):
        """
        Проверка на горизонтальный тренд с использованием AI
        
        Parameters:
        data: DataFrame с ценовыми данными
        window: Окно анализа (по умолчанию 48 часов)
        
        Returns:
        dict: Информация о тренде
        """
        try:
            prices = data['close'].tail(window).values
            time_index = np.arange(len(prices)).reshape(-1, 1)
            
            # Стандартизация данных
            scaler = StandardScaler()
            time_scaled = scaler.fit_transform(time_index)
            prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
            
            # Линейная регрессия
            model = LinearRegression()
            model.fit(time_scaled, prices_scaled)
            
            # Статистический анализ
            slope, intercept, r_value, p_value, std_err = linregress(time_index.flatten(), prices)
            
            # Проверка на горизонтальность
            is_horizontal = abs(slope) < 0.001  # Небольшой порог для горизонтальности
            confidence = 1.0 - abs(slope)  # Чем ближе к 0, тем более горизонтальный тренд
            
            return {
                'is_horizontal': is_horizontal,
                'confidence': confidence,
                'slope': slope,
                'r_squared': r_value**2,
                'std_dev': std_err,
                'p_value': p_value
            }
        except Exception as e:
            logger.error(f"Ошибка при проверке горизонтального тренда: {e}")
            return {
                'is_horizontal': False,
                'confidence': 0.0,
                'slope': 0.0,
                'r_squared': 0.0,
                'std_dev': 0.0,
                'p_value': 1.0
            }
        
    
    def analyze_market_context(self, recent_data):
        """
        Анализ контекста рынка
        
        Parameters:
        recent_data: DataFrame с последними ценовыми данными
        """
        # Проверка горизонтального тренда
        trend_info = self.check_horizontal_trend(recent_data)
        
        # Логирование информации о тренде
        if trend_info['is_horizontal']:
            logger.info(f"Обнаружен горизонтальный тренд (уверенность: {trend_info['confidence']:.2f}, R²: {trend_info['r_squared']:.2f})")
        else:
            logger.info(f"Тренд не горизонтальный (наклон: {trend_info['slope']:.6f}, R²: {trend_info['r_squared']:.2f})")
        
        # Проверка снятия SSL/BSL на разных таймфреймах
        self.ssl_bsl_removed = False
        self.ssl_bsl_levels = []
        
        # Проверяем SSL/BSL на разных таймфреймах
        for timeframe_name, timeframe_hours in self.timeframes.items():
            timeframe_data = self.data_1h.tail(timeframe_hours)
            self.ssl_bsl_levels.extend(self.check_ssl_bsl(timeframe_data, timeframe_name))
        
        # Проверяем реакцию от зон POI/FVG
        self.check_poi_fvg(recent_data)
        
        # Обновляем контекст рынка
        self.current_context = self.update_context(recent_data)
        
        return self.current_context
    
    def check_ssl_bsl(self, timeframe_data, timeframe_name):
        """
        Проверка SSL/BSL уровней
        
        Parameters:
        timeframe_data: DataFrame с данными для проверки
        timeframe_name: Название таймфрейма
        """
        ssl_bsl_levels = []
        
        # Для шортов проверяем SSL
        if self.current_context == 'short':
            ssl_price = float(timeframe_data['high'].max())
            if any(float(row['close']) > ssl_price for _, row in self.data_3m.iterrows()):
                ssl_bsl_levels.append({
                    'type': 'SSL',
                    'price': ssl_price,
                    'timeframe': timeframe_name
                })
        
        # Для лонгов проверяем BSL
        if self.current_context == 'long':
            bsl_price = float(timeframe_data['low'].min())
            if any(float(row['close']) < bsl_price for _, row in self.data_3m.iterrows()):
                ssl_bsl_levels.append({
                    'type': 'BSL',
                    'price': bsl_price,
                    'timeframe': timeframe_name
                })
        
        # Если нашли хотя бы один SSL/BSL, считаем его снятым
        if ssl_bsl_levels:
            self.ssl_bsl_removed = True
            for level in ssl_bsl_levels:
                logger.info(f"{level['type']} снят на уровне {level['price']} (таймфрейм: {level['timeframe']})")
            
            # Определяем наиболее значимый SSL/BSL
            if self.current_context == 'short':
                most_significant = max(ssl_bsl_levels, key=lambda x: x['price'])
            else:
                most_significant = min(ssl_bsl_levels, key=lambda x: x['price'])
            
            logger.info(f"Наиболее значимый {most_significant['type']} на уровне {most_significant['price']} (таймфрейм: {most_significant['timeframe']})")
        
        return ssl_bsl_levels
    
    def analyze_market_context(self, recent_data):
        """
        Анализ контекста рынка
        
        Parameters:
        recent_data: DataFrame с последними ценовыми данными
        """
        # Проверка горизонтального тренда
        trend_info = self.check_horizontal_trend(recent_data)
        
        # Логирование информации о тренде
        if trend_info['is_horizontal']:
            logger.info(f"Обнаружен горизонтальный тренд (уверенность: {trend_info['confidence']:.2f}, R²: {trend_info['r_squared']:.2f})")
        else:
            logger.info(f"Тренд не горизонтальный (наклон: {trend_info['slope']:.6f}, R²: {trend_info['r_squared']:.2f})")
        
        # Проверка снятия SSL/BSL на разных таймфреймах
        self.ssl_bsl_removed = False
        self.ssl_bsl_levels = []
        
        # Проверяем SSL/BSL на разных таймфреймах
        for timeframe_name, timeframe_hours in self.timeframes.items():
            timeframe_data = self.data_1h.tail(timeframe_hours)
            self.ssl_bsl_levels.extend(self.check_ssl_bsl(timeframe_data, timeframe_name))
        
        # Проверяем реакцию от зон POI/FVG
        self.check_poi_fvg(recent_data)
        
        # Обновляем контекст рынка
        self.current_context = self.update_context(recent_data)
        
        return self.current_context
    
    def check_poi_fvg(self, recent_data):
        """
        Проверка реакции от зон POI/FVG
        
        Parameters:
        recent_data: DataFrame с последними ценовыми данными
        """
        # Находим зоны POI (последние 48 часов)
        poi_zones = []
        for i in range(2):  # Проверяем последние 2 дня
            day_data = recent_data.iloc[i*24:(i+1)*24]
            if not day_data.empty:
                day_high = float(day_data['high'].max())
                day_low = float(day_data['low'].min())
                poi_zones.append((day_high, day_low))
        
        # Проверяем реакцию от зон POI
        current_price = float(self.data_3m['close'].iloc[-1])
        for high, low in poi_zones:
            # Проверяем FVG (First Visible Gap)
            if (current_price > high or current_price < low) and \
                any(float(row['close']) > high or float(row['close']) < low 
                    for _, row in self.data_3m.iterrows()):
                self.poi_fvg_reacted = True
                logger.info(f"Реакция от зоны POI/FVG: {high}-{low}")

    def update_context(self, recent_data):
        """
        Обновление контекста рынка на основе цены и SMA
        
        Parameters:
        recent_data: DataFrame с последними ценовыми данными
        """
        # Проверяем реакцию от зон POI/FVG
        self.check_poi_fvg(recent_data)
        
        # Корректируем контекст на основе SSL/BSL и POI/FVG
        if self.ssl_bsl_removed or self.poi_fvg_reacted:
            if self.current_context == 'long':
                self.current_context = 'short'
                logger.info("Контекст изменен на шорт из-за снятия SSL/BSL или реакции POI/FVG")
            else:
                self.current_context = 'long'
                logger.info("Контекст изменен на лонг из-за снятия SSL/BSL или реакции POI/FVG")
        
        # Простая логика: если цена выше SMA 48, считаем контекст лонговым, иначе шортовым
        sma = SMAIndicator(close=recent_data['close'], window=48).sma_indicator().iloc[-1]
        current_price = recent_data['close'].iloc[-1]
        
        if current_price > sma:
            self.current_context = 'long'
        else:
            self.current_context = 'short'
        
        logger.info(f"Определен контекст рынка: {self.current_context}")
        return self.current_context

    def identify_fractals(self, data, window=2):
        """
        Идентификация фракталов Билла Вильямса
        
        Parameters:
        data: DataFrame с ценовыми данными
        window: Окно анализа (по умолчанию 2)
        
        Returns:
        tuple: (бычьи фракталы, медвежьи фракталы)
        """
        # Make sure we're accessing values properly
        highs = data['high'].values
        lows = data['low'].values
        timestamps = data.index
        
        bullish_fractals = []
        bearish_fractals = []
        
        # Поиск бычьих фракталов (дно)
        for i in range(window, len(data) - window):
            if all(float(lows[i]) < float(lows[i-j]) for j in range(1, window+1)) and \
            all(float(lows[i]) < float(lows[i+j]) for j in range(1, window+1)):
                bullish_fractals.append({
                    'timestamp': timestamps[i],
                    'price': float(lows[i]),
                    'type': 'bullish'
                })
        
        # Поиск медвежьих фракталов (вершина)
        for i in range(window, len(data) - window):
            if all(float(highs[i]) > float(highs[i-j]) for j in range(1, window+1)) and \
            all(float(highs[i]) > float(highs[i+j]) for j in range(1, window+1)):
                bearish_fractals.append({
                    'timestamp': timestamps[i],
                    'price': float(highs[i]),
                    'type': 'bearish'
                })
        
        return bullish_fractals, bearish_fractals

    def find_entry_signals(self):
        """
        Поиск сигналов входа на основе фракталов и сессий
        
        Returns:
        list: Список сигналов входа
        """
        # Получение часовых фракталов
        bullish_fractals_1h, bearish_fractals_1h = self.identify_fractals(self.data_1h)
        
        # Фильтрация фракталов по сессиям
        frankfurt_fractals = self.filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, 'frankfurt')
        london_fractals = self.filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, 'london')
        ny_fractals = self.filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, 'newyork')
        
        # Объединение интересующих нас фракталов
        target_fractals = frankfurt_fractals + london_fractals + ny_fractals
        
        # Текущие точки набора
        self.fractal_levels = target_fractals
        
        # Сбрасываем skip_conditions при каждом поиске
        self.skip_conditions = []
        
        # Список для хранения сигналов входа
        entry_signals = []
        
        # Проверка времени для GER40
        if self.symbol == 'GER40':
            current_time = datetime.now() # Убедитесь, что datetime импортирован: from datetime import datetime
            # Проверяем, что до начала Нью-Йоркской сессии осталось больше 5 минут
            if current_time.hour == 12 and current_time.minute >= 55:  # 12:55 - 5 минут до 13:00
                logger.info(f"Пропуск сигнала для GER40: менее 5 минут до начала Нью-Йоркской сессии") # Убедитесь, что logger определен
                return []
        
        # Проверка каждого фрактала на возможность входа
        for fractal in target_fractals:
            # Проверяем соответствие направления фрактала текущему контексту
            valid_fractal_type = (self.current_context == 'long' and fractal['type'] == 'bullish') or \
                                (self.current_context == 'short' and fractal['type'] == 'bearish')
            
            if not valid_fractal_type:
                continue
            
            # Проверяем слом фрактала на 3-минутном графике
            # (логика слома фрактала, если она есть, должна быть здесь или в check_fractal_breakout)

            # Исправление: Добавляем вызов check_skip_conditions перед использованием skip_reasons
            skip_reasons = self.check_skip_conditions(fractal)
            
            if skip_reasons:
                logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']} по причинам: {', '.join(skip_reasons)}")
                self.skip_conditions.append({
                    'fractal': fractal,
                    'reasons': skip_reasons,
                    'timestamp': datetime.now() # Убедитесь, что datetime импортирован
                })
                continue
            
            # Проверяем потенциальную цель
            target = self.calculate_target(fractal)
            if target is not None:
                # Убедимся, что data_3m не пустой и содержит столбец 'close'
                if self.data_3m is None or self.data_3m.empty or 'close' not in self.data_3m.columns:
                    logger.warning("data_3m пуст или не содержит столбец 'close'. Невозможно получить entry_price.")
                    continue
                
                entry_price = float(self.data_3m['close'].iloc[-1])
                
                # Рассчитываем потенциальную прибыль и убыток
                if self.current_context == 'long':
                    potential_profit = target - entry_price
                    potential_loss = entry_price - fractal['price']
                else: # self.current_context == 'short'
                    potential_profit = entry_price - target
                    potential_loss = fractal['price'] - entry_price
                
                # Проверяем соотношение риск/прибыль
                # Добавлена проверка potential_loss > 0 чтобы избежать деления на ноль
                if potential_loss > 0 and potential_profit / potential_loss >= 1.3:
                    entry_signals.append({
                        'fractal': fractal,
                        'entry_price': entry_price,
                        'stop_loss': self.calculate_stop_loss(entry_price, fractal['price']),
                        'take_profit': target,
                        'direction': 'long' if self.current_context == 'long' else 'short'
                    })
                else:
                    if potential_loss <= 0:
                        logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']}: потенциальный убыток равен нулю или отрицателен ({potential_loss=}).")
                    else:
                        logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']}: RR < 1.3 ({potential_profit / potential_loss:.2f})")
            
        return entry_signals


        """
        Поиск сигналов входа на основе фракталов и сессий
        """
        # Получение часовых фракталов
        bullish_fractals_1h, bearish_fractals_1h = self.identify_fractals(self.data_1h)
        
        # Фильтрация фракталов по сессиям
        frankfurt_fractals = self.filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, 'frankfurt')
        london_fractals = self.filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, 'london')
        ny_fractals = self.filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, 'newyork')
        
        # Объединение интересующих нас фракталов
        target_fractals = frankfurt_fractals + london_fractals + ny_fractals
        
        # Текущие точки набора
        self.fractal_levels = target_fractals
        
        # Сбрасываем skip_conditions при каждом поиске
        self.skip_conditions = []
        
        # Список для хранения сигналов входа
        entry_signals = []
        
        # Проверка каждого фрактала на возможность входа
        for fractal in target_fractals:
            if self.check_skip_conditions(fractal):
                continue
            
            if self.check_fractal_breakout(fractal):
                # Проверяем скип-ситуации
                skip_reasons = self.check_skip_conditions(fractal)
                
                if skip_reasons:
                    logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']} по причинам: {', '.join(skip_reasons)}")
                    self.skip_conditions.append({
                        'fractal': fractal,
                        'reasons': skip_reasons,
                        'timestamp': datetime.now()
                    })
                    continue
                
                # Проверяем потенциальную цель
                target = self.calculate_target(fractal)
                if target is not None:
                    entry_price = float(self.data_3m['close'].iloc[-1])
                    
                    # Рассчитываем потенциальную прибыль и убыток
                    if self.current_context == 'long':
                        potential_profit = target - entry_price
                        potential_loss = entry_price - fractal['price']
                    else:
                        potential_profit = entry_price - target
                        potential_loss = fractal['price'] - entry_price
                    
                    # Проверяем соотношение риск/прибыль
                    if potential_profit / potential_loss >= 1.3:
                        entry_signals.append({
                            'fractal': fractal,
                            'entry_price': entry_price,
                            'stop_loss': self.calculate_stop_loss(entry_price, fractal['price']),
                            'take_profit': target,
                            'direction': 'long' if self.current_context == 'long' else 'short'
                        })
                    else:
                        logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']}: RR < 1.3")
        
        return entry_signals

    
        """
        Поиск сигналов входа на основе фракталов и сессий
        
        Returns:
        list: Список сигналов входа
        """
        # ... (предыдущий код метода) ...
        
        # Проверка каждого фрактала на возможность входа
        for fractal in target_fractals:
            # Проверяем соответствие направления фрактала текущему контексту
            valid_fractal_type = (self.current_context == 'long' and fractal['type'] == 'bullish') or \
                                (self.current_context == 'short' and fractal['type'] == 'bearish')
            
            if not valid_fractal_type:
                continue
            
            # Проверяем слом фрактала на 3-минутном графике
            
            # ИЗМЕНЕНИЕ ЗДЕСЬ: Добавляем вызов check_skip_conditions
            skip_reasons = self.check_skip_conditions(fractal)
            
            if skip_reasons:
                logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']} по причинам: {', '.join(skip_reasons)}")
                self.skip_conditions.append({
                    'fractal': fractal,
                    'reasons': skip_reasons,
                    'timestamp': datetime.now()
                })
                continue
            
            # Проверяем потенциальную цель
            target = self.calculate_target(fractal)
            if target is not None:
                entry_price = float(self.data_3m['close'].iloc[-1])
                
                # Рассчитываем потенциальную прибыль и убыток
                if self.current_context == 'long':
                    potential_profit = target - entry_price
                    potential_loss = entry_price - fractal['price']
                else:
                    potential_profit = entry_price - target
                    potential_loss = fractal['price'] - entry_price
                
                # Проверяем соотношение риск/прибыль
                if potential_loss > 0 and potential_profit / potential_loss >= 1.3: # Добавлена проверка potential_loss > 0 чтобы избежать деления на ноль
                    entry_signals.append({
                        'fractal': fractal,
                        'entry_price': entry_price,
                        'stop_loss': self.calculate_stop_loss(entry_price, fractal['price']),
                        'take_profit': target,
                        'direction': 'long' if self.current_context == 'long' else 'short'
                    })
                else:
                    if potential_loss <= 0:
                        logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']}: потенциальный убыток равен нулю или отрицателен.")
                    else:
                        logger.info(f"Пропуск сигнала для фрактала {fractal['timestamp']}: RR < 1.3")
            
        return entry_signals
    
    def filter_fractals_by_session(self, fractals, session_name, days_ago=0):
        """
        Фильтрация фракталов по торговой сессии
        """
        filtered_fractals = []
        # Изменения здесь:
        session_info = self.sessions[session_name]
        session_start = session_info['start']
        session_end = session_info['end']
        
        # Make sure we're getting the right number of values here
        for fractal in fractals:
            timestamp = fractal['timestamp']
            
            # Учитываем смещение дней
            if days_ago > 0:
                target_date = (datetime.now() - timedelta(days=days_ago)).date()
                if timestamp.date() != target_date:
                    continue
            
            # Проверка попадания в часы сессии
            if session_start <= timestamp.hour < session_end:
                filtered_fractals.append(fractal)
        
        return filtered_fractals

        """
        Фильтрация фракталов по торговой сессии
        """
        filtered_fractals = []
        session_start, session_end = self.sessions[session_name]
        
        # Make sure we're getting the right number of values here
        for fractal in fractals:
            timestamp = fractal['timestamp']
            
            # Учитываем смещение дней
            if days_ago > 0:
                target_date = (datetime.now() - timedelta(days=days_ago)).date()
                if timestamp.date() != target_date:
                    continue
            
            # Проверка попадания в часы сессии
            if session_start <= timestamp.hour < session_end:
                filtered_fractals.append(fractal)
        
        return filtered_fractals


    def check_fractal_breakout(self, fractal):
        """
        Проверка пробоя фрактала на 3-минутном графике с учетом типа фрактала
        """
        # Получаем текущие цены
        current_high = self.data_3m['high'].iloc[-1]
        current_low = self.data_3m['low'].iloc[-1]
        
        # Проверяем пробой с учетом типа фрактала
        if fractal['type'] == 'bullish':
            return current_high > fractal['price']
            if fractal['type'] == 'bullish':
                # Для лонга проверяем, что последний максимум пробил фрактал
                return last_high > fractal_price and \
                       any(float(row['close']) > fractal_price for _, row in recent_3m.iterrows())
        else:
            if fractal['type'] == 'bearish':
                # Для шорта проверяем, что последний минимум пробил фрактал
                return last_low < fractal_price and \
                       any(float(row['close']) < fractal_price for _, row in recent_3m.iterrows())
        
        return False

    def check_liquidity_swipe(self, session_name):
        """
        Проверка свипа ликвидности в указанной сессии
        """
        recent_3m = self.data_3m.tail(20)
        session_start, session_end = self.sessions[session_name]
        
        # Проверяем свип ликвидности в текущей сессии
        session_candles = recent_3m[(recent_3m.index.hour >= session_start) & 
                                  (recent_3m.index.hour < session_end)]
        
        if len(session_candles) < 2:
            return False
            
        # Проверяем наличие свипа (резкое движение в одну сторону)
        price_range = float(session_candles['high'].max()) - float(session_candles['low'].min())
        avg_range = float(recent_3m['high'] - recent_3m['low']).mean()
        
        return price_range > 2 * avg_range

    def check_price_context(self, data_3m):
        """
        Определяет текущее состояние цены с учетом азиатской сессии
        """
        # Проверяем четкое направление с Order Flow
        if self.is_clear_trend(data_3m):
            self.price_context = self.price_context_states['CLEAR_TREND']
            return
            
        # Проверяем неэффективную доставку без Order Flow
        if self.is_inefficient_delivery(data_3m):
            self.price_context = self.price_context_states['INEFFICIENT_DELIVERY']
            return
            
        # Проверяем приход к таргету
        if self.is_target_approach(data_3m):
            self.price_context = self.price_context_states['TARGET_APPROACH']
            return
            
        # Проверяем горизонтальный тренд
        if self.is_horizontal_trend(data_3m):
            self.price_context = self.price_context_states['HORIZONTAL_TREND']
            return
            
        # Проверяем состояние азиатской сессии
        if self.is_asian_session_active():
            asian_context = self.analyze_asian_session(data_3m)
            if asian_context:
                self.price_context = asian_context
                return
            
        # По умолчанию считаем это коррекцией
        self.price_context = self.price_context_states['CORRECTION']

    def is_asian_session_active(self):
        """
        Проверяет наличие четкого направления в азиатской сессии
        """
        # Проверяем, что мы в азиатской сессии
        current_hour = datetime.utcnow().hour
        if current_hour not in self.sessions['asia'].values():
            return False
            
        # Получаем данные за азиатскую сессию
        asian_data = self.get_asian_session_data(self.data_3m)
        
        # Проверяем четкое направление
        return self.has_clear_asian_direction(asian_data)

    def analyze_asian_session(self, data_3m):
        """
        Анализирует поведение азиатской сессии
        """
        # Получаем данные за азиатскую сессию
        asian_data = self.get_asian_session_data(data_3m)
        
        # Проверяем четкое направление Азии
        if self.has_clear_asian_direction(asian_data):
            return self.price_context_states['CLEAR_TREND']
            
        # Проверяем закрепление АХ/АЛ
        if self.is_asian_hl_confirmation(asian_data):
            return self.price_context_states['CLEAR_TREND']
            
        # Проверяем свип против тренда
        if self.is_asian_counter_trend_swipe(asian_data):
            return self.price_context_states['INEFFICIENT_DELIVERY']
            
        return None

    def get_asian_session_data(self, data_3m):
        """
        Получает данные за азиатскую сессию
        """
        # Берем последние 2 часа данных (с запасом)
        return data_3m.tail(40)  # 40 свечей по 3 минуты = 2 часа

    def has_clear_asian_direction(self, asian_data):
        """
        Проверяет наличие четкого направления в азиатской сессии
        """
        # Проверяем угол наклона тренда
        price_changes = asian_data['close'].pct_change().dropna()
        angle = np.degrees(np.arctan(price_changes.mean()))
        
        # Проверяем объемы
        avg_volume = asian_data['volume'].mean()
        last_volume = asian_data['volume'].iloc[-1]
        
        # Четкое направление: угол > 10 градусов и объем выше среднего
        return abs(angle) > 10 and last_volume > avg_volume * 1.2

    def is_asian_hl_confirmation(self, asian_data):
        """
        Проверяет закрепление АХ/АЛ
        """
        last_candle = asian_data.iloc[-1]
        prev_candle = asian_data.iloc[-2]
        
        # Закрепление АХ: последняя свеча выше предыдущей
        is_hl_confirmation = (
            (last_candle['high'] > prev_candle['high'] and 
             last_candle['low'] > prev_candle['low']) or
            (last_candle['low'] < prev_candle['low'] and 
             last_candle['high'] < prev_candle['high'])
        )
        
        return is_hl_confirmation

    def is_asian_counter_trend_swipe(self, asian_data):
        """
        Проверяет наличие свипа против тренда в азиатской сессии
        """
        # Проверяем глобальный тренд
        global_trend = self.get_global_trend(data_3m)
        
        # Проверяем последнюю свечу на свип
        last_candle = asian_data.iloc[-1]
        prev_candle = asian_data.iloc[-2]
        
        # Свип против тренда: большая тень в противоположном направлении
        if global_trend == 'up':
            return last_candle['high'] > prev_candle['high'] * 1.01 and \
                   abs(last_candle['high'] - last_candle['close']) > abs(last_candle['open'] - last_candle['close']) * 2
        else:
            return last_candle['low'] < prev_candle['low'] * 0.99 and \
                   abs(last_candle['low'] - last_candle['close']) > abs(last_candle['open'] - last_candle['close']) * 2

    def get_global_trend(self, data_3m):
        """
        Определяет глобальный тренд на основе данных
        """
        # Берем последние 100 свечей для анализа тренда
        trend_data = data_3m.tail(100)
        
        # Проверяем угол наклона тренда
        price_changes = trend_data['close'].pct_change().dropna()
        angle = np.degrees(np.arctan(price_changes.mean()))
        
        # Определяем тренд
        if angle > 5:
            return 'up'
        elif angle < -5:
            return 'down'
        else:
            return 'sideways'

    def is_clear_trend(self, data_3m):
        """
        Проверяет наличие четкого направления с Order Flow
        """
        # Проверяем силу тренда
        last_candle = data_3m.iloc[-1]
        prev_candle = data_3m.iloc[-2]
        
        # Проверяем Order Flow
        volume_increase = last_candle['volume'] > prev_candle['volume'] * 1.5
        price_momentum = abs(last_candle['close'] - last_candle['open']) > abs(prev_candle['close'] - prev_candle['open']) * 1.2
        
        # Проверяем угол наклона тренда
        price_changes = data_3m['close'].pct_change().dropna()
        angle = np.degrees(np.arctan(price_changes.mean()))
        
        # Четкий тренд: сильный Order Flow и угол наклона более 15 градусов
        return volume_increase and price_momentum and abs(angle) > 15

    def is_inefficient_delivery(self, data_3m):
        """
        Проверяет наличие неэффективной доставки без Order Flow
        """
        # Проверяем объемы
        last_candle = data_3m.iloc[-1]
        prev_candle = data_3m.iloc[-2]
        
        # Проверяем паттерны ценового движения
        price_range = abs(last_candle['high'] - last_candle['low'])
        body_size = abs(last_candle['close'] - last_candle['open'])
        
        # Неэффективная доставка: маленький объем при большом ценовом диапазоне
        return (last_candle['volume'] < prev_candle['volume'] * 0.8 and 
                price_range > body_size * 2)

    def is_target_approach(self, data_3m):
        """
        Проверяет приближение к таргету
        """
        last_candle = data_3m.iloc[-1]
        
        # Проверяем структуру свечи
        body_size = abs(last_candle['close'] - last_candle['open'])
        wick_size = abs(last_candle['high'] - last_candle['low'])
        
        # Приход к таргету: большая тень при маленьком теле
        if wick_size > body_size * 2:
            # Проверяем направление
            if last_candle['close'] > last_candle['open']:
                # Бычья свеча с большой нижней тенью
                return True
            else:
                # Медвежья свеча с большой верхней тенью
                return True
        
        return False

    def is_horizontal_trend(self, data_3m):
        """
        Проверяет наличие горизонтального тренда
        """
        # Проверяем стандартное отклонение
        price_std = data_3m['close'].std()
        
        # Проверяем угол наклона
        price_changes = data_3m['close'].pct_change().dropna()
        angle = np.degrees(np.arctan(price_changes.mean()))
        
        # Горизонтальный тренд: маленькое стандартное отклонение и небольшой угол
        is_horizontal = (price_std < 0.001 and abs(angle) < 5)
        
        # Обновляем параметры горизонтального тренда
        self.horizontal_trend['is_horizontal'] = is_horizontal
        self.horizontal_trend['confidence'] = 1.0 - abs(angle) / 5
        self.horizontal_trend['slope'] = angle
        self.horizontal_trend['std_dev'] = price_std
        self.horizontal_trend['last_update'] = datetime.utcnow()
        
        return is_horizontal

    def check_skip_conditions(self, fractal):
        """
        Проверка скип-ситуаций, когда не следует входить в сделку
        """
        skip_reasons = []
        
        # Проверяем горизонтальный тренд
        if self.horizontal_trend['is_horizontal'] and self.horizontal_trend['confidence'] > 0.8:
            # Если обнаружен сильный горизонтальный тренд, пропускаем сигналы в направлении тренда
            if self.current_context == 'long' and self.horizontal_trend['slope'] > 0:
                skip_reasons.append("Пропуск сигнала: горизонтальный тренд с положительным наклоном")
            elif self.current_context == 'short' and self.horizontal_trend['slope'] < 0:
                skip_reasons.append("Пропуск сигнала: горизонтальный тренд с отрицательным наклоном")
        
        # Проверяем реверсальную ситуацию
        if self.reversal_state['important_target_removed']:
            # Проверяем свип ликвидности слева
            if self.reversal_state['confirmed_swipes'] < 2:
                # Проверяем свип в европейской сессии
                if self.check_liquidity_swipe('frankfurt') or self.check_liquidity_swipe('london'):
                    self.reversal_state['confirmed_swipes'] += 1
                    self.reversal_state['last_swipe_session'] = 'european'
                
                # Если Азия поработала с ликвидностью дважды, считаем это одним подтверждением
                if self.check_liquidity_swipe('asia'):
                    self.reversal_state['confirmed_swipes'] += 1 # Может быть += 0.5 если нужно два разных свипа
                    self.reversal_state['last_swipe_session'] = 'asia'
            
            # Если у нас два подтверждения, проверяем условия для реверса
            if self.reversal_state['confirmed_swipes'] >= 2:
                # Проверяем, нет ли целей выше/ниже
                current_price = float(self.data_3m['close'].iloc[-1])
                if self.current_context == 'long': # Если контекст лонг, ищем реверс в шорт
                    recent_lows = self.data_1h['low'].tail(48) # Ищем цели ниже для шорта
                    if not any(float(low) < current_price for low in recent_lows): # Если нет целей ниже
                        skip_reasons.append("Реверс: нет таргета ниже для шорта")
                else: # Если контекст шорт, ищем реверс в лонг
                    recent_highs = self.data_1h['high'].tail(48) # Ищем цели выше для лонга
                    if not any(float(high) > current_price for high in recent_highs): # Если нет целей выше
                        skip_reasons.append("Реверс: нет таргета выше для лонга")
                        
                # Проверяем противостоящую ОФ (Order Flow)
                # Пример: если контекст лонг, и Нью-Йорк делает свип вниз (шортовый ОФ), это может быть против реверса в лонг
                if self.current_context == 'long' and self.check_liquidity_swipe('newyork'): # Предполагаем, что check_liquidity_swipe может вернуть направление
                    pass # Логика для "Против ОФ" может быть сложнее
                
                # Проверяем PWH/PWL (Previous Week High/Low) - эта логика здесь не реализована, но может быть добавлена
                # if self.current_context == 'long':
                #     # recent_highs = self.data_1h['high'].tail(168) # Данные за неделю
                #     # if any(float(high) > current_price for high in recent_highs): # Упрощенно
                #     #     skip_reasons.append("Есть PWH")
                # else:
                #     # recent_lows = self.data_1h['low'].tail(168)
                #     # if any(float(low) < current_price for low in recent_lows):
                #     #     skip_reasons.append("Есть PWL")
        
        # 1. Проверка снятия DL без закрепления в шортовом контексте
        if self.current_context == 'short' and self.daily_limit is not None:
            recent_3m_dl_check = self.data_3m.tail(5)  # Последние 5 свечей
            if not recent_3m_dl_check.empty:
                dl_value = float(self.daily_limit)
                
                # Fix: Make sure we're accessing DataFrame values correctly
                dl_broken = any(float(row['low']) < dl_value for _, row in recent_3m_dl_check.iterrows())
                # Проверяем закрепление: последние 2 свечи закрылись ниже DL
                dl_confirmed = all(float(row['close']) < dl_value for _, row in recent_3m_dl_check.tail(2).iterrows()) if len(recent_3m_dl_check) >=2 else False
                
                if dl_broken and not dl_confirmed:
                    skip_reasons.append("Снятие DL без закрепления в шортовом контексте")
        
        # 2. Проверка, что фрактал не слишком старый (больше 2 дней)
        fractal_time = pd.to_datetime(fractal['timestamp']) # Убедитесь, что pandas импортирован: import pandas as pd
        if (datetime.now() - fractal_time).days > 2: # Убедитесь, что datetime импортирован: from datetime import datetime
            skip_reasons.append("Фрактал старше 2 дней")
        
        # 3. Проверка расстояния до цели
        try:
            target_distance = float(self.calculate_target_distance(fractal))
            max_points = float(self.max_target_points)
            if target_distance > max_points:
                skip_reasons.append(f"Цель превышает максимальное расстояние ({target_distance:.0f} > {max_points:.0f})")
        except (ValueError, TypeError) as e:
            logger.error(f"Ошибка при расчете расстояния до цели для фрактала {fractal['timestamp']}: {e}")
            skip_reasons.append("Ошибка расчета расстояния до цели")
        
        # 5. Проверка на Frankfurt manipulation setup
        # Этот блок должен выполняться только если текущий контекст соответствует типу фрактала
        # Например, если контекст 'long', то ищем бычий франкфуртский фрактал для манипуляции
        if (self.current_context == 'long' and fractal['type'] == 'bullish') or \
           (self.current_context == 'short' and fractal['type'] == 'bearish'):
            # Проверяем, что фрактал находится в Франкфуртской сессии
            # Используем self.filter_fractals_by_session для проверки принадлежности фрактала сессии
            if fractal in self.filter_fractals_by_session([fractal], 'frankfurt'): # Передаем список с одним фракталом
                recent_3m_frankfurt_check = self.data_3m.tail(120) # Берем данные за последние 6 часов (120 * 3 мин) для анализа сессий
                if not recent_3m_frankfurt_check.empty:
                    frankfurt_end_hour = self.sessions['frankfurt']['end'] # 15 UTC
                    london_start_hour = self.sessions['london']['start']   # 8 UTC

                    # Находим свечи Франкфуртской сессии в указанном временном интервале (например, последний час Франкфурта)
                    # Уточняем время: ищем свечи в последний час Франкфуртской сессии (14:00-14:59 UTC)
                    frankfurt_candles_filtered = recent_3m_frankfurt_check[
                        (recent_3m_frankfurt_check.index.hour >= frankfurt_end_hour - 1) & 
                        (recent_3m_frankfurt_check.index.hour < frankfurt_end_hour)
                    ]
                    
                    if not frankfurt_candles_filtered.empty:
                        # Берем минимум/максимум за этот период Франкфурта в зависимости от контекста
                        if self.current_context == 'long': # Ищем снятие франкфуртского лоу
                            frankfurt_extreme_price = float(frankfurt_candles_filtered['low'].min())
                        else: # Ищем снятие франкфуртского хая
                            frankfurt_extreme_price = float(frankfurt_candles_filtered['high'].max())

                        # Находим все свечи Лондонской сессии в первый час (08:00-08:59 UTC)
                        london_candles = recent_3m_frankfurt_check[
                            (recent_3m_frankfurt_check.index.hour >= london_start_hour) & 
                            (recent_3m_frankfurt_check.index.hour < london_start_hour + 1)
                        ]
                        
                        if not london_candles.empty:
                            london_manipulation_detected = False
                            if self.current_context == 'long': # Лондон снял франкфуртский лоу
                                london_manipulation_detected = any(float(candle['low']) < frankfurt_extreme_price for _, candle in london_candles.iterrows())
                            else: # Лондон снял франкфуртский хай
                                london_manipulation_detected = any(float(candle['high']) > frankfurt_extreme_price for _, candle in london_candles.iterrows())
                            
                            if london_manipulation_detected:
                                # Проверяем 3м слом в сторону контекста после манипуляции
                                if self.check_fractal_breakout(fractal): 
                                    skip_reasons.append("Frankfurt manipulation setup")
                    else:
                        logger.info("Не найдены свечи Франкфуртской сессии для проверки Frankfurt manipulation setup.")
                else:
                    logger.info("recent_3m пуст, невозможно проверить Frankfurt manipulation setup.")
                        
        # 4. Проверка на противоречие контексту (это условие уже проверено в find_entry_signals, но можно оставить для надежности)
        if (self.current_context == 'long' and fractal['type'] == 'bearish') or \
           (self.current_context == 'short' and fractal['type'] == 'bullish'):
            skip_reasons.append("Фрактал противоречит текущему контексту рынка")
        
        # 5. Проверка на уже открытую позицию в том же направлении
        if any(pos['direction'] == self.current_context for pos in self.open_positions):
            skip_reasons.append("Уже открыта позиция в данном направлении")
            
        return skip_reasons

        """
        Проверка скип-ситуаций, когда не следует входить в сделку
        """
        skip_reasons = []
        
        # Проверяем горизонтальный тренд
        if self.horizontal_trend['is_horizontal'] and self.horizontal_trend['confidence'] > 0.8:
            # Если обнаружен сильный горизонтальный тренд, пропускаем сигналы в направлении тренда
            if self.current_context == 'long' and self.horizontal_trend['slope'] > 0:
                skip_reasons.append("Пропуск сигнала: горизонтальный тренд с положительным наклоном")
            elif self.current_context == 'short' and self.horizontal_trend['slope'] < 0:
                skip_reasons.append("Пропуск сигнала: горизонтальный тренд с отрицательным наклоном")
        
        # Проверяем реверсальную ситуацию
        if self.reversal_state['important_target_removed']:
            # Проверяем свип ликвидности слева
            if self.reversal_state['confirmed_swipes'] < 2:
                # Проверяем свип в европейской сессии
                if self.check_liquidity_swipe('frankfurt') or self.check_liquidity_swipe('london'):
                    self.reversal_state['confirmed_swipes'] += 1
                    self.reversal_state['last_swipe_session'] = 'european'
                
                # Если Азия поработала с ликвидностью дважды, считаем это одним подтверждением
                if self.check_liquidity_swipe('asia'):
                    self.reversal_state['confirmed_swipes'] += 1
                    self.reversal_state['last_swipe_session'] = 'asia'
            
            # Если у нас два подтверждения, проверяем условия для реверса
            if self.reversal_state['confirmed_swipes'] >= 2:
                # Проверяем, нет ли целей выше/ниже
                current_price = float(self.data_3m['close'].iloc[-1])
                if self.current_context == 'long':
                    recent_lows = self.data_1h['low'].tail(48)
                    if any(float(low) < current_price for low in recent_lows):
                        skip_reasons.append("Есть таргет ниже")
                else:
                    recent_highs = self.data_1h['high'].tail(48)
                    if any(float(high) > current_price for high in recent_highs):
                        skip_reasons.append("Есть таргет выше")
                        
                # Проверяем противостоящую ОФ
                if self.current_context == 'long' and self.check_liquidity_swipe('newyork'):
                    skip_reasons.append("Против ОФ")
                
                # Проверяем PWH/PWL
                if self.current_context == 'long':
                    recent_highs = self.data_1h['high'].tail(48)
                    if any(float(high) > current_price for high in recent_highs):
                        skip_reasons.append("Есть PWH")
                else:
                    recent_lows = self.data_1h['low'].tail(48)
                    if any(float(low) < current_price for low in recent_lows):
                        skip_reasons.append("Есть PWL")
        
        # 1. Проверка снятия DL без закрепления в шортовом контексте
        if self.current_context == 'short' and self.daily_limit is not None:
            recent_3m = self.data_3m.tail(5)  # Последние 5 свечей
            dl_value = float(self.daily_limit)
            
            # Fix: Make sure we're accessing DataFrame values correctly
            dl_broken = any(float(row['low']) < dl_value for _, row in recent_3m.iterrows())
            dl_confirmed = all(float(row['close']) < dl_value for _, row in recent_3m.tail(2).iterrows())
            
            if dl_broken and not dl_confirmed:
                skip_reasons.append("Снятие DL без закрепления в шортовом контексте")
        
        # 2. Проверка, что фрактал не слишком старый (больше 2 дней)
        fractal_time = pd.to_datetime(fractal['timestamp'])
        if (datetime.now() - fractal_time).days > 2:
            skip_reasons.append("Фрактал старше 2 дней")
        
        # 3. Проверка расстояния до цели
        try:
            target_distance = float(self.calculate_target_distance(fractal))
            max_points = float(self.max_target_points)
            if target_distance > max_points:
                skip_reasons.append(f"Цель превышает максимальное расстояние ({target_distance:.0f} > {max_points:.0f})")
        except (ValueError, TypeError) as e:
            logger.error(f"Ошибка при расчете расстояния до цели: {e}")
            skip_reasons.append("Ошибка расчета расстояния до цели")
        
        # 5. Проверка на Frankfurt manipulation setup
        if self.current_context == 'long':
            # Проверяем, что фрактал находится в Франкфуртской сессии
            if fractal in self.filter_fractals_by_session([fractal], 'frankfurt'):
                # Проверяем, что Лондонская сессия сняла Франкфуртскую
                recent_3m = self.data_3m.tail(20)
                frankfurt_end = 15  # Конец Франкфуртской сессии
                london_start = 8   # Начало Лондонской сессии
                
                # Находим последнюю свечу Франкфуртской сессии
                frankfurt_candle = recent_3m[(recent_3m.index.hour == frankfurt_end) & 
                                          (recent_3m.index.minute < 30)].iloc[-1]
                
                # Находим все свечи Лондонской сессии
                london_candles = recent_3m[(recent_3m.index.hour >= london_start) & 
                                         (recent_3m.index.hour < london_start + 1)]
                
                # Проверяем, есть ли хотя бы одна свеча Лондонской сессии, которая сняла Франкфуртскую
                frankfurt_low = float(frankfurt_candle['low'])
                london_break = any(float(candle['low']) < frankfurt_low for _, candle in london_candles.iterrows())
                
                if london_break:
                    # Проверяем 3м слом
                    if self.check_fractal_breakout(fractal):
                        skip_reasons.append("Frankfurt manipulation setup")
                        
        # 4. Проверка на противоречие контексту
        if (self.current_context == 'long' and fractal['type'] == 'bearish') or \
        (self.current_context == 'short' and fractal['type'] == 'bullish'):
            skip_reasons.append("Фрактал противоречит текущему контексту рынка")
        
        # 5. Проверка на уже открытую позицию в том же направлении
        if any(pos['direction'] == self.current_context for pos in self.open_positions):
            skip_reasons.append("Уже открыта позиция в данном направлении")
            
        return skip_reasons


    def calculate_target_distance(self, fractal):
        """
        Расчет расстояния до потенциальной цели в пунктах
        """
        current_price = self.data_3m['close'].iloc[-1]
        
        if self.current_context == 'long':
            # Для лонга целевое расстояние вверх
            recent_highs = self.data_1h['high'].tail(48)  # Последние 2 дня
            potential_target = recent_highs.max()
            distance = (potential_target - current_price) / self.point_size
        else:
            # Для шорта целевое расстояние вниз
            recent_lows = self.data_1h['low'].tail(48)  # Последние 2 дня
            potential_target = recent_lows.min()
            distance = (current_price - potential_target) / self.point_size

        logger.debug(f"Calculated distance: {distance}, type: {type(distance)}")
        return float(abs(distance))

    def calculate_stop_loss(self, entry_price, fractal_price):
        """
        Расчет стоп-лосса с учетом волатильности инструмента
        """
        # Рассчитываем текущую волатильность
        volatility = self.calculate_volatility(self.data_3m)
        
        # Определяем расстояние для стопа в зависимости от инструмента
        if self.symbol == 'XAUUSD':
            stop_distance = volatility * 2  # 2 ATR для золота
        elif self.symbol in ['EURUSD', 'GBPUSD']:
            stop_distance = volatility * 1.5  # 1.5 ATR для валют
        else:
            stop_distance = volatility * 2  # 2 ATR для GER40
        
        if self.current_context == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_target(self, fractal):
        """
        Расчет целевого уровня для позиции
        """
        if fractal is None:
            logger.warning("Фрактал не предоставлен для calculate_target.")
            return None

        # Убедимся, что data_3m не пустой и содержит столбец 'close'
        if self.data_3m is None or self.data_3m.empty or 'close' not in self.data_3m.columns:
            logger.error("data_3m пуст или не содержит столбец 'close'. Невозможно рассчитать цель.")
            return None
        
        current_price = float(self.data_3m['close'].iloc[-1])

        # Определяем направление и ищем цель
        if self.current_context == 'long':
            # Для лонга ищем подходящий максимум в пределах max_target_points
            # Цель должна быть выше текущей цены
            max_potential_target_price = current_price + (self.max_target_points * self.point_size)
            
            # Рассматриваем недавние максимумы на 1H графике (например, за последние 48 часов)
            recent_highs = self.data_1h['high'].tail(48) 
            
            # Фильтруем максимумы: они должны быть выше текущей цены и не дальше max_potential_target_price
            valid_highs = recent_highs[(recent_highs > current_price) & (recent_highs <= max_potential_target_price)]
            
            if not valid_highs.empty:
                # Берем самый ближний к текущей цене валидный максимум (или самый высокий, в зависимости от стратегии)
                # В данном случае, давайте возьмем самый высокий из доступных в пределах лимита
                calculated_target = valid_highs.max()
                logger.info(f"Long Target: Найдена цель по историческим максимумам: {calculated_target}")
                return float(calculated_target)
            else:
                # Если не нашли подходящую цель в исторических данных,
                # используем максимальное допустимое расстояние от текущей цены
                calculated_target = max_potential_target_price
                logger.info(f"Long Target: Используется максимальное расстояние от текущей цены: {calculated_target}")
                return float(calculated_target)

        elif self.current_context == 'short':
            # Для шорта ищем подходящий минимум в пределах max_target_points
            # Цель должна быть ниже текущей цены
            min_potential_target_price = current_price - (self.max_target_points * self.point_size)
            
            # Рассматриваем недавние минимумы на 1H графике
            recent_lows = self.data_1h['low'].tail(48)
            
            # Фильтруем минимумы: они должны быть ниже текущей цены и не ближе min_potential_target_price
            valid_lows = recent_lows[(recent_lows < current_price) & (recent_lows >= min_potential_target_price)]
            
            if not valid_lows.empty:
                # Берем самый ближний к текущей цене валидный минимум (или самый низкий)
                # В данном случае, давайте возьмем самый низкий из доступных в пределах лимита
                calculated_target = valid_lows.min()
                logger.info(f"Short Target: Найдена цель по историческим минимумам: {calculated_target}")
                return float(calculated_target)
            else:
                # Если не нашли подходящую цель в исторических данных,
                # используем максимальное допустимое расстояние от текущей цены
                calculated_target = min_potential_target_price
                logger.info(f"Short Target: Используется максимальное расстояние от текущей цены: {calculated_target}")
                return float(calculated_target)
        else:
            logger.warning(f"Неизвестный контекст рынка: {self.current_context} в calculate_target.")
            return None

        """
        Расчет целевого уровня для позиции
        """
        if fractal is None:
            return None
            
        # Определяем направление
            # Находим локальный минимум в пределах допустимого расстояния
            min_target = current_price - (self.max_target_points * self.point_size)
            recent_lows = self.data_1h['low'].tail(48)  # Последние 2 дня
            
            # Фильтруем минитумы, которые находятся в пределах допустимого расстояния
            valid_lows = recent_lows[recent_lows >= min_target]
            
            if not valid_lows.empty:
                return valid_lows.min()
        
        # Если не нашли подходящую цель в исторических данных,
        # используем максимальное допустимое расстояние
        if self.current_context == 'long':
            return current_price + (self.max_target_points * self.point_size)
        else:
            return current_price - (self.max_target_points * self.point_size)

    def execute_trade(self, entry_signal):
        """
        Исполнение торгового сигнала
        
        Parameters:
        entry_signal (dict): Словарь с параметрами входа
            - direction: 'long' или 'short'
            - entry_price: Цена входа
            - stop_loss: Стоп-лосс
            - take_profit: Тейк-профит
            - fractal_type: Тип фрактала
        """
        # Получаем текущее время для записи в базу
        entry_time = datetime.now()
        
        # Вычисляем размер позиции
        position_size = self.calculate_position_size(
            entry_signal['entry_price'],
            entry_signal['stop_loss']
        )
        
        # Логируем информацию о сделке в базу
        trade_data = {
            'symbol': self.symbol,
            'entry_time': entry_time,
            'direction': entry_signal['direction'],
            'entry_price': entry_signal['entry_price'],
            'stop_loss': entry_signal['stop_loss'],
            'take_profit': entry_signal['take_profit'],
            'position_size': position_size,
            'price_context': self.price_context,
            'session': self.get_current_session(),
            'fractal_type': entry_signal['fractal_type'],
            'is_profitable': False,  # Будет обновлено при закрытии сделки
            'risk_reward_ratio': abs(entry_signal['take_profit'] - entry_signal['entry_price']) / 
                               abs(entry_signal['entry_price'] - entry_signal['stop_loss'])
        }
        
        trade_id = self.db.log_trade(trade_data)
        """
        Исполнение торгового сигнала
        """
        if self.exchange:
            # Реальное исполнение через биржу
            direction = entry_signal['direction']
            entry_price = entry_signal['entry_price']
            target_price = entry_signal['target']
            
            # Расчет стоп-лосса (например, на основе фрактала)
            stop_loss = entry_signal['fractal']['price']
            
            # Расчет объема позиции (примерно 1% риска)
            risk_amount = 0.01  # 1% от депозита
            position_size = self.calculate_position_size(entry_price, stop_loss, risk_amount)
            
            # Формирование и отправка ордера
            order_type = 'buy' if direction == 'long' else 'sell'
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=order_type,
                    amount=position_size
                )
                
                # Добавление в открытые позиции
                self.open_positions.append({
                    'id': order['id'],
                    'direction': direction,
                    'entry_price': entry_price,
                    'target': target_price,
                    'stop_loss': stop_loss,
                    'size': position_size,
                    'entry_time': datetime.now()
                })
                
                logger.info(f"Открыта позиция: {direction} по {entry_price}, цель: {target_price}, стоп: {stop_loss}")
                return True
            except Exception as e:
                logger.error(f"Ошибка открытия позиции: {e}")
                return False
        else:
            # Режим тестирования - симуляция открытия позиции
            direction = entry_signal['direction']
            entry_price = entry_signal['entry_price']
            target_price = entry_signal['target']
            
            # Расчет стоп-лосса
            stop_loss = entry_signal['fractal']['price']
            
            # Добавление в открытые позиции
            self.open_positions.append({
                'id': f"test-{len(self.open_positions) + 1}",
                'direction': direction,
                'entry_price': entry_price,
                'target': target_price,
                'stop_loss': stop_loss,
                'size': 1.0,  # Тестовый размер
                'entry_time': datetime.now()
            })
            
            logger.info(f"[ТЕСТ] Открыта позиция: {direction} по {entry_price}, цель: {target_price}, стоп: {stop_loss}")
            return True

    def calculate_position_size(self, entry_price, stop_loss, risk_amount=0.01):
        """
        Расчет размера позиции на основе риска с учетом инструмента
        """
        if self.exchange:
            # Получение баланса аккаунта
            balance = self.exchange.fetch_balance()
            total_balance = balance['total']['USD']
            
            # Сумма риска
            risk_in_currency = total_balance * risk_amount
            
            # Расчет размера позиции с учетом волатильности
            pip_risk = abs(entry_price - stop_loss) / self.point_size
            position_size = risk_in_currency / pip_risk
            
            # Ограничение максимального размера позиции
            if self.symbol == 'XAUUSD':
                position_size = min(position_size, 100)  # Максимум 100 контрактов для золота
            elif self.symbol in ['EURUSD', 'GBPUSD']:
                position_size = min(position_size, 1000)  # Максимум 1000 контрактов для валют
            else:
                position_size = min(position_size, 100)  # Максимум 100 контрактов для GER40
            
            return position_size
        else:
            # В тестовом режиме используем адекватный размер
            if self.symbol == 'XAUUSD':
                return 10.0
            elif self.symbol in ['EURUSD', 'GBPUSD']:
                return 100.0
            else:
                return 10.0

    def manage_open_positions(self):
        """
        Управление открытыми позициями
        """
        current_time = datetime.now()
        
        # Получаем все открытые позиции из базы
        open_trades = self.db.get_trade_stats(
            symbol=self.symbol,
            start_date=current_time - timedelta(days=1)
        )
        
        # Получаем текущую цену
        current_price = self.get_current_price()
        
        for trade in open_trades:
            if trade['exit_time'] is None:  # Позиция все еще открыта
                direction = trade['direction']
                entry_price = trade['entry_price']
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']
                
                # Проверяем условия закрытия
                if direction == 'long':
                    if current_price <= stop_loss or current_price >= take_profit:
                        exit_price = current_price
                        profit_loss = (exit_price - entry_price) * trade['position_size']
                        is_profitable = profit_loss > 0
                        
                        # Обновляем информацию о сделке в базе
                        self.db.conn.execute('''
                            UPDATE trades SET
                                exit_time = ?,
                                exit_price = ?,
                                profit_loss = ?,
                                is_profitable = ?
                            WHERE id = ?
                        ''', (current_time, exit_price, profit_loss, is_profitable, trade['id']))
                        self.db.conn.commit()
                else:  # short
                    if current_price >= stop_loss or current_price <= take_profit:
                        exit_price = current_price
                        profit_loss = (entry_price - exit_price) * trade['position_size']
                        is_profitable = profit_loss > 0
                        
                        # Обновляем информацию о сделке в базе
                        self.db.conn.execute('''
                            UPDATE trades SET
                                exit_time = ?,
                                exit_price = ?,
                                profit_loss = ?,
                                is_profitable = ?
                            WHERE id = ?
                        ''', (current_time, exit_price, profit_loss, is_profitable, trade['id']))
                        self.db.conn.commit()
        """
        Управление открытыми позициями
        """
        if not self.open_positions:
            return
        
        current_price = self.data_3m['close'].iloc[-1]
        positions_to_close = []
        
        for i, position in enumerate(self.open_positions):
            # Проверка достижения цели
            target_reached = (position['direction'] == 'long' and current_price >= position['target']) or \
                            (position['direction'] == 'short' and current_price <= position['target'])
            
            # Проверка стоп-лосса
            stop_loss_triggered = (position['direction'] == 'long' and current_price <= position['stop_loss']) or \
                                 (position['direction'] == 'short' and current_price >= position['stop_loss'])
            
            if target_reached or stop_loss_triggered:
                if self.exchange:
                    # Закрытие реальной позиции
                    try:
                        order_type = 'sell' if position['direction'] == 'long' else 'buy'
                        self.exchange.create_order(
                            symbol=self.symbol,
                            type='market',
                            side=order_type,
                            amount=position['size']
                        )
                        
                        reason = "цель достигнута" if target_reached else "сработал стоп-лосс"
                        pnl = (current_price - position['entry_price']) * position['size'] if position['direction'] == 'long' else \
                              (position['entry_price'] - current_price) * position['size']
                        
                        logger.info(f"Закрыта позиция {position['id']}: {reason}, PnL: {pnl}")
                    except Exception as e:
                        logger.error(f"Ошибка закрытия позиции: {e}")
                else:
                    # Тестовый режим
                    reason = "цель достигнута" if target_reached else "сработал стоп-лосс"
                    pnl = (current_price - position['entry_price']) / self.point_size if position['direction'] == 'long' else \
                          (position['entry_price'] - current_price) / self.point_size
                    
                    logger.info(f"[ТЕСТ] Закрыта позиция {position['id']}: {reason}, PnL: {pnl} пунктов")
                
                positions_to_close.append(i)
        
        # Удаление закрытых позиций
        for i in sorted(positions_to_close, reverse=True):
            del self.open_positions[i]

    def update_daily_limit(self):
        """
        Обновление дневного лимита (DL)
        """
        # Дневной лимит - последний локальный экстремум в противоположном направлении контекста
        today_data = self.data_1h[self.data_1h.index.date == datetime.now().date()]
        
        if not today_data.empty:
            if self.current_context == 'long':
                # В лонговом контексте DL - это минимум
                self.daily_limit = today_data['low'].min()
            else:
                # В шортовом контексте DL - это максимум
                self.daily_limit = today_data['high'].max()
            
            logger.info(f"Обновлен дневной лимит (DL): {self.daily_limit}")

    def run(self):
        """
        Запуск торгового бота
        """
        while True:
            try:
                # 1. Загрузка данных, только если еще не загружены
                if bot.data_1h is None or bot.data_3m is None:
                    logging.info('Загрузка исторических данных...')
                    bot.fetch_data()
                    bot.analyze_market_context(bot.data_3m)
                    bot.update_daily_limit()
                
                # Выводим информацию о контексте рынка
                logging.info(f'Контекст рынка: {bot.current_context}')
                print_market_context(bot.current_context)
                
                # Проверяем условия для входа
                conditions = check_entry_conditions(bot)
                logging.info(f'Условия для входа: {conditions}')
                
                print(f"\n{Fore.WHITE}{Style.BRIGHT}Условия для входа:")
                for name, is_met, description in conditions:
                    print_condition(name, is_met, description)
                
                # Выводим информацию о дневном лимите
                logging.info(f'Дневной лимит: {bot.daily_limit}')
                print_daily_limit(bot.daily_limit, bot.current_context)
                
                # Выводим информацию о фрактальных уровнях
                logging.info(f'Фрактальные уровни: {bot.fractal_levels}')
                print_fractal_levels(bot.fractal_levels)
                
                # Выводим информацию о текущей сессии
                logging.info('Статус торговых сессий')
                print_session_status()
                
                # Проверяем наличие сигналов
                entry_signals = bot.find_entry_signals()
                logging.info(f'Найденные сигналы: {entry_signals}')
                
                # Выводим информацию о сигналах
                if entry_signals:
                    for signal in entry_signals:
                        print_signal(signal)
                else:
                    logging.info('Нет активных сигналов для входа')
                    print(f"\n{Fore.YELLOW}Нет активных сигналов для входа")
                
                # Выводим информацию о пропущенных сигналах
                logging.info(f'Пропущенные сигналы: {bot.skip_conditions}')
                print_skip_conditions(bot.skip_conditions)
                
                # Выводим информацию об открытых позициях
                logging.info(f'Открытые позиции: {bot.open_positions}')
                print_open_positions(bot.open_positions)
                
                # Управление открытыми позициями
                if bot.open_positions:
                    bot.manage_open_positions()
                
                # Выводим меню
                print_menu()
                
                # Получаем ввод пользователя
                choice = input(f"{Fore.WHITE}Введите номер действия: ")
                
                if choice == '0':
                    logging.info('Выход из программы')
                    print(f"{Fore.GREEN}Выход из программы...")
                    break
                
                elif choice == '1':
                    logging.info('Обновление данных')
                    print(f"{Fore.YELLOW}Обновление данных...")
                    bot.fetch_data()
                    bot.analyze_market_context(bot.data_3m)
                    bot.update_daily_limit()
                    bot.find_entry_signals()
                
                elif choice == '2':
                    if entry_signals:
                        logging.info('Выполнение сделки по сигналу')
                        print(f"{Fore.YELLOW}Выполнение сделки по сигналу...")
                        for signal in entry_signals:
                            bot.execute_trade(signal)
                        time.sleep(2)
                    else:
                        logging.info('Нет активных сигналов для входа')
                        print(f"{Fore.RED}Нет активных сигналов для входа")
                        time.sleep(2)
                
                elif choice == '3':
                    logging.info('Создание графика стратегии')
                    print(f"{Fore.YELLOW}Создание графика стратегии...")
                    bot.visualize_strategy(save_path="strategy_chart.png")
                    print(f"{Fore.GREEN}График сохранен в файл strategy_chart.png")
                    time.sleep(2)
                
                elif choice == '4':
                    new_symbol = input(f"{Fore.WHITE}Введите символ (EURUSD или GER40): ").upper()
                    if new_symbol in ['EURUSD', 'GER40']:
                        bot = TradingBot1H3M(symbol=new_symbol)
                        logging.info(f'Символ изменен на {new_symbol}')
                        print(f"{Fore.GREEN}Символ изменен на {new_symbol}")
                        bot.fetch_data()
                        bot.analyze_market_context(bot.data_3m)
                        bot.update_daily_limit()
                    else:
                        logging.info('Неподдерживаемый символ')
                        print(f"{Fore.RED}Неподдерживаемый символ. Используйте EURUSD или GER40")
                    time.sleep(2)
                
                else:
                    logging.info('Неверный ввод')
                    print(f"{Fore.RED}Неверный ввод")
                    time.sleep(1)
                
            except Exception as e:
                logging.error(f'Произошла ошибка: {e}')
                print(f"{Fore.RED}Произошла ошибка: {e}")
                time.sleep(5)

    def visualize_strategy(self, save_path=None):
        """
        Визуализация текущего состояния стратегии с использованием свечных графиков.
        """
        # Проверка наличия данных перед использованием
        if self.data_1h is None or self.data_1h.empty or self.data_3m is None or self.data_3m.empty:
            logger.error("Данные data_1h или data_3m не загружены или пусты. Визуализация невозможна.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Данные для графика отсутствуют", ha='center', va='center')
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Пустой график сохранен в {save_path}")
            else:
                plt.show()
            return fig

        # Подготовка данных для mplfinance
        recent_1h_ohlc = self.data_1h.tail(48).copy()
        if not recent_1h_ohlc.empty:
            recent_1h_ohlc.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            if recent_1h_ohlc.index.tzinfo is None:
                recent_1h_ohlc.index = recent_1h_ohlc.index.tz_localize('UTC')
            else:
                recent_1h_ohlc.index = recent_1h_ohlc.index.tz_convert('UTC')
        else:
            logger.warning("recent_1h_ohlc пуст после tail(48).")

        recent_3m_ohlc = self.data_3m.tail(100).copy()
        if not recent_3m_ohlc.empty:
            recent_3m_ohlc.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            if recent_3m_ohlc.index.tzinfo is None:
                recent_3m_ohlc.index = recent_3m_ohlc.index.tz_localize('UTC')
            else:
                recent_3m_ohlc.index = recent_3m_ohlc.index.tz_convert('UTC')
        else:
            logger.warning("recent_3m_ohlc пуст после tail(100).")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
        mc_style = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc_style, gridstyle=':', y_on_right=False)

        # --- Часовой график (ax1) ---
        if not recent_1h_ohlc.empty:
            logger.info("--- Информация о recent_1h_ohlc перед построением графика ---")
            logger.info(f"recent_1h_ohlc.index.name: {recent_1h_ohlc.index.name}, recent_1h_ohlc.shape: {recent_1h_ohlc.shape}")
            # Логирование .info() требует StringIO для захвата вывода
            import io
            buffer = io.StringIO()
            recent_1h_ohlc.info(buf=buffer)
            logger.info(buffer.getvalue())
            logger.info(f"recent_1h_ohlc.head():\n{recent_1h_ohlc.head().to_string()}")
            
            ohlc_columns = ['Open', 'High', 'Low', 'Close']
            if any(col not in recent_1h_ohlc.columns for col in ohlc_columns):
                logger.error(f"Одна или несколько колонок OHLC отсутствуют в recent_1h_ohlc. Имеющиеся колонки: {recent_1h_ohlc.columns.tolist()}")
                ax1.text(0.5, 0.5, "Ошибка: Отсутствуют колонки OHLC для 1H графика", ha='center', va='center', transform=ax1.transAxes, color='red')
            elif recent_1h_ohlc[ohlc_columns].isnull().all().all():
                logger.error("Все OHLC данные в recent_1h_ohlc являются NaN. Свечи не могут быть построены.")
                ax1.text(0.5, 0.5, "Нет валидных OHLC данных для 1H графика", ha='center', va='center', transform=ax1.transAxes)
            elif recent_1h_ohlc[ohlc_columns].isnull().any().any():
                logger.warning("В OHLC данных для recent_1h_ohlc присутствуют NaN значения.")
                logger.warning(f"Количество NaN в recent_1h_ohlc:\n{recent_1h_ohlc[ohlc_columns].isnull().sum()}")
                mpf.plot(recent_1h_ohlc, type='candle', ax=ax1, style=s, ylabel="Цена (1H)") # Пытаемся построить несмотря на NaN
            else:
                logger.info("Данные recent_1h_ohlc выглядят нормально для построения свечного графика.")
                mpf.plot(recent_1h_ohlc, type='candle', ax=ax1, style=s, ylabel="Цена (1H)")
            
            ax1.set_title(f"Стратегия 1H3M для {self.symbol} (1H)", fontsize=14)

            # Отмечаем фракталы на 1H графике
            bullish_fractals_1h_display, bearish_fractals_1h_display = self.identify_fractals(self.data_1h.tail(48))
            for f_bullish in bullish_fractals_1h_display:
                f_timestamp_aware = pd.Timestamp(f_bullish['timestamp'])
                if f_timestamp_aware.tzinfo is None: f_timestamp_aware = f_timestamp_aware.tz_localize('UTC')
                else: f_timestamp_aware = f_timestamp_aware.tz_convert('UTC')
                if f_timestamp_aware in recent_1h_ohlc.index:
                     ax1.scatter(f_timestamp_aware, f_bullish['price'], color='lime', marker='^', s=100, edgecolors='black', zorder=5)
            for f_bearish in bearish_fractals_1h_display:
                f_timestamp_aware = pd.Timestamp(f_bearish['timestamp'])
                if f_timestamp_aware.tzinfo is None: f_timestamp_aware = f_timestamp_aware.tz_localize('UTC')
                else: f_timestamp_aware = f_timestamp_aware.tz_convert('UTC')
                if f_timestamp_aware in recent_1h_ohlc.index:
                    ax1.scatter(f_timestamp_aware, f_bearish['price'], color='red', marker='v', s=100, edgecolors='black', zorder=5)
            
            if self.daily_limit is not None:
                ax1.axhline(y=self.daily_limit, color='purple', linestyle='--', label=f'Дневной лимит (DL): {self.daily_limit:.5f}', zorder=3)
            
            if self.current_context:
                context_color = 'green' if self.current_context == 'long' else 'red'
                # Проверка на случай если recent_1h_ohlc все же пуст на этом этапе (хотя не должен быть)
                if not recent_1h_ohlc.empty:
                     y_pos_context = recent_1h_ohlc['Low'].min() - (recent_1h_ohlc['High'].max() - recent_1h_ohlc['Low'].min()) * 0.05 
                     ax1.text(recent_1h_ohlc.index[0], y_pos_context,
                              f"Контекст: {self.current_context.upper()}", 
                              color=context_color, fontsize=14, bbox=dict(facecolor='white', alpha=0.5, pad=2))
            
            for pos in self.open_positions:
                marker = '^' if pos['direction'] == 'long' else 'v'
                color = 'green' if pos['direction'] == 'long' else 'red'
                entry_time_original = pd.Timestamp(pos['entry_time'])
                if entry_time_original.tzinfo is None: entry_time_aware = entry_time_original.tz_localize('UTC')
                else: entry_time_aware = entry_time_original.tz_convert('UTC')
                if not recent_1h_ohlc.empty and recent_1h_ohlc.index[0] <= entry_time_aware <= recent_1h_ohlc.index[-1]:
                    closest_date_1h = recent_1h_ohlc.index.asof(entry_time_aware)
                    if closest_date_1h is not pd.NaT:
                        ax1.scatter(closest_date_1h, pos['entry_price'], color=color, marker=marker, s=200, edgecolors='black', zorder=5)
                        ax1.axhline(y=pos['target'], color=color, linestyle='-.', alpha=0.7, label=f"Цель: {pos['target']:.5f}", zorder=3)
                        ax1.axhline(y=pos['stop_loss'], color='gray', linestyle=':', alpha=0.7, label=f"Стоп: {pos['stop_loss']:.5f}", zorder=3)
            
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) 
            if by_label: ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
        else:
            ax1.text(0.5, 0.5, "Нет данных для 1H графика", ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f"Стратегия 1H3M для {self.symbol} (1H) - Нет данных", fontsize=14)

        # --- 3-минутный график (ax2) ---
        if not recent_3m_ohlc.empty:
            mpf.plot(recent_3m_ohlc, type='candle', ax=ax2, style=s, ylabel="Цена (3M)")
            ax2.set_title("3-минутный график", fontsize=12)
            for skip in self.skip_conditions:
                skip_timestamp_original = pd.Timestamp(skip['timestamp'])
                if skip_timestamp_original.tzinfo is None: skip_time_aware = skip_timestamp_original.tz_localize('UTC')
                else: skip_time_aware = skip_timestamp_original.tz_convert('UTC')
                if not recent_3m_ohlc.empty and recent_3m_ohlc.index[0] <= skip_time_aware <= recent_3m_ohlc.index[-1]:
                    closest_date_3m = recent_3m_ohlc.index.asof(skip_time_aware)
                    if closest_date_3m is not pd.NaT and closest_date_3m in recent_3m_ohlc.index:
                        price_at_skip = recent_3m_ohlc.loc[closest_date_3m, 'Close'] 
                        ax2.scatter(closest_date_3m, price_at_skip, color='orange', marker='x', s=150, zorder=5)
                        reasons_text = '\n'.join(skip['reasons'][:2]) 
                        ax2.annotate(reasons_text, xy=(closest_date_3m, price_at_skip), xytext=(10, -30), 
                                     textcoords='offset points', fontsize=8,
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="black"), zorder=6)
        else:
            ax2.text(0.5, 0.5, "Нет данных для 3M графика", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("3-минутный график - Нет данных", fontsize=12)
        
        fig.tight_layout() 
        if save_path: plt.savefig(save_path)
        else: plt.show() 
        return fig
