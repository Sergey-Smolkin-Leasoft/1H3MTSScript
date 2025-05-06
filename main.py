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
        
        # Настройка Twelve Data API
        load_dotenv()
        self.td_api_key = os.getenv('TWELVE_DATA_API_KEY', '9c614fea46d04e3d8c4f3f76b0541ab6')
        self.td = TDClient(apikey=self.td_api_key)
        
        # Маппинг символов для Twelve Data
        self.symbol_mapping = {
            'EURUSD': 'EUR/USD',
            'GER40': 'DAX'
        }
        
        # Настройки инструментов
        if symbol == 'EURUSD':
            self.max_target_points = 250
            self.point_size = 0.0001
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
        
        # Дневной лимит (DL)
        self.daily_limit = None
        
        # Текущие открытые позиции
        self.open_positions = []
        
        logger.info(f"Бот инициализирован для {symbol} с максимальной целью {self.max_target_points} пунктов")

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

    def determine_market_context(self):
        """
        Определение текущего контекста рынка (лонг или шорт) на основе последних 1-2 дней
        """
        # Используем данные за последние 2 дня
        recent_data = self.data_1h.tail(48)  # 48 часов = 2 дня
        
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

def find_entry_points(self):
    """
    Поиск точек входа на основе фракталов и 3-минутного слома
    """
    # Получение часовых фракталов
    bullish_fractals_1h, bearish_fractals_1h = self.identify_fractals(self.data_1h)
    
    # Фильтрация фракталов по сессиям
    asia_fractals = self.filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, 'asia')
    ny_fractals_yesterday = self.filter_fractals_by_session(
        bullish_fractals_1h + bearish_fractals_1h, 'newyork', days_ago=1
    )
    
    # Объединение интересующих нас фракталов
    target_fractals = asia_fractals + ny_fractals_yesterday
    
    # Текущие точки набора
    self.fractal_levels = target_fractals
    
    # Сбрасываем skip_conditions при каждом поиске
    self.skip_conditions = []
    
    # Проверка 3-минутного слома для определения входа
    entry_signals = []
    
    for fractal in target_fractals:
        # Проверяем соответствие направления фрактала текущему контексту
        valid_fractal_type = (self.current_context == 'long' and fractal['type'] == 'bullish') or \
                            (self.current_context == 'short' and fractal['type'] == 'bearish')
        
        if not valid_fractal_type:
            continue
        
        # Проверяем слом фрактала на 3-минутном графике
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
                entry_signals.append({
                    'fractal': fractal,
                    'target': float(target),
                    'entry_price': float(self.data_3m['close'].iloc[-1]),
                    'direction': self.current_context
                })
    
    return entry_signals
   

    def filter_fractals_by_session(self, fractals, session_name, days_ago=0):
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
    # Получаем последние 3-минутные свечи
        recent_3m = self.data_3m.tail(20)  # Берем несколько последних свечей
        fractal_price = float(fractal['price'])
    
        if self.current_context == 'long':
        # Для лонга ищем пробой бычьего фрактала вверх
            if fractal['type'] == 'bullish':
                # Fix: Make sure we're accessing DataFrame values correctly
                return any(float(row['close']) > fractal_price for _, row in recent_3m.iterrows())
        else:
            # Для шорта ищем пробой медвежьего фрактала вниз
            if fractal['type'] == 'bearish':
            # Fix: Make sure we're accessing DataFrame values correctly
                return any(float(row['close']) < fractal_price for _, row in recent_3m.iterrows())
    
        return False

    def check_skip_conditions(self, fractal):
        """
        Проверка скип-ситуаций, когда не следует входить в сделку
        """
        skip_reasons = []
        
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
        if (datetime.now() - fractal['timestamp']).days > 2:
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

    def calculate_target(self, fractal):
        """
        Расчет целевого уровня для позиции
        """
        current_price = self.data_3m['close'].iloc[-1]
        
        if self.current_context == 'long':
            # Находим локальный максимум в пределах допустимого расстояния
            max_target = current_price + (self.max_target_points * self.point_size)
            recent_highs = self.data_1h['high'].tail(48)  # Последние 2 дня
            
            # Фильтруем максимумы, которые находятся в пределах допустимого расстояния
            valid_highs = recent_highs[recent_highs.astype(float) <= max_target]
            
            if not valid_highs.empty:
                return valid_highs.max()
        else:
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

    def calculate_position_size(self, entry_price, stop_loss, risk_amount):
        """
        Расчет размера позиции на основе риска
        """
        if self.exchange:
            # Получение баланса аккаунта
            balance = self.exchange.fetch_balance()
            total_balance = balance['total']['USD']  # Предполагаем, что баланс в USD
            
            # Сумма риска
            risk_in_currency = total_balance * risk_amount
            
            # Расчет размера позиции
            pip_risk = abs(entry_price - stop_loss) / self.point_size
            position_size = risk_in_currency / pip_risk
            
            return position_size
        else:
            # В тестовом режиме возвращаем фиксированный размер
            return 1.0

    def manage_open_positions(self):
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
                    bot.determine_market_context()
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
                entry_signals = bot.find_entry_points()
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
                    bot.determine_market_context()
                    bot.update_daily_limit()
                    bot.find_entry_points()
                
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
                        bot.determine_market_context()
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
        Визуализация текущего состояния стратегии
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Часовой график
        recent_1h = self.data_1h.tail(48)  # Последние 2 дня
        ax1.plot(recent_1h.index, recent_1h['close'], label='Цена закрытия (1H)', color='blue')
        
        # Отмечаем фракталы
        bullish_fractals, bearish_fractals = self.identify_fractals(recent_1h)
        
        for f in bullish_fractals:
            ax1.scatter(f['timestamp'], f['price'], color='green', marker='^', s=100)
        
        for f in bearish_fractals:
            ax1.scatter(f['timestamp'], f['price'], color='red', marker='v', s=100)
        
        # Отмечаем DL
        if self.daily_limit:
            ax1.axhline(y=self.daily_limit, color='purple', linestyle='--', label='Дневной лимит (DL)')
        
        # Отмечаем текущий контекст
        if self.current_context:
            context_color = 'green' if self.current_context == 'long' else 'red'
            ax1.text(recent_1h.index[0], recent_1h['close'].min(), 
                     f"Контекст: {self.current_context.upper()}", 
                     color=context_color, fontsize=14)
        
        # Отмечаем открытые позиции
        for pos in self.open_positions:
            marker = '^' if pos['direction'] == 'long' else 'v'
            color = 'green' if pos['direction'] == 'long' else 'red'
            
            # Находим ближайшую дату к времени входа
            closest_date = min(recent_1h.index, key=lambda d: abs(d - pos['entry_time']))
            
            ax1.scatter(closest_date, pos['entry_price'], color=color, marker=marker, s=200, edgecolors='black')
            ax1.axhline(y=pos['target'], color=color, linestyle='-.', alpha=0.7)
            ax1.axhline(y=pos['stop_loss'], color='black', linestyle=':', alpha=0.7)
        
        # 3-минутный график
        recent_3m = self.data_3m.tail(100)  # Последние 100 3-минутных свечей
        ax2.plot(recent_3m.index, recent_3m['close'], label='Цена закрытия (3M)', color='orange')
        
        # Отмечаем скип-ситуации
        for skip in self.skip_conditions:
            closest_date = min(recent_3m.index, key=lambda d: abs(d - skip['timestamp']))
            ax2.scatter(closest_date, recent_3m.loc[closest_date, 'close'], 
                       color='gray', marker='x', s=150)
            
            # Добавляем текст с причинами
            reasons_text = '\n'.join(skip['reasons'][:2])  # Показываем максимум 2 причины
            ax2.annotate(reasons_text, 
                        xy=(closest_date, recent_3m.loc[closest_date, 'close']),
                        xytext=(10, -30),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # Форматирование осей
        ax1.set_title(f"Стратегия 1H3M для {self.symbol}", fontsize=16)
        ax1.set_ylabel("Цена", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        ax2.set_title("3-минутный график", fontsize=14)
        ax2.set_xlabel("Время", fontsize=12)
        ax2.set_ylabel("Цена", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Форматирование дат на оси X
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Поворот меток
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            logger.info(f"График сохранен в {save_path}")
        
        return fig