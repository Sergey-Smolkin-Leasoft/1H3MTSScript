# execution/trade_executor.py

import logging
from datetime import datetime
from data.database import TradingDatabase # Для логирования сделок
from config import settings # Для доступа к общим настройкам, если понадобятся

class TradeExecutor:
    """
    Отвечает за "исполнение" торговых сигналов (симуляция)
    и расчет размера позиции.
    """
    def __init__(self, symbol: str, point_size: float, db: TradingDatabase):
        """
        Инициализация исполнителя сделок.

        Args:
            symbol (str): Текущий торговый символ.
            point_size (float): Размер пункта для текущего символа.
            db (TradingDatabase): Экземпляр класса для работы с БД.
        """
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol
        self.point_size = point_size
        self.db = db
        # Параметры для расчета размера позиции (можно вынести в settings)
        self.account_balance_simulated = 10000  # Симулированный баланс счета
        self.risk_per_trade_percentage = 0.01 # Риск 1% на сделку

    def _calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float | None:
        """
        Рассчитывает симулированный размер позиции.

        Returns:
            float | None: Размер позиции или None при ошибке.
        """
        if stop_loss_price == entry_price:
            self.logger.warning("Цена стоп-лосса равна цене входа. Невозможно рассчитать размер позиции.")
            return None

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0: # Дополнительная проверка
             self.logger.warning("Нулевой риск на единицу. Невозможно рассчитать размер позиции.")
             return None

        # Сумма риска в валюте счета
        risk_amount_currency = self.account_balance_simulated * self.risk_per_trade_percentage

        # Стоимость одного пункта (для упрощения, примем ее равной 1 у.е. за стандартный лот,
        # а размер позиции будет в этих "лотах" или единицах базового актива).
        # В реальности это зависит от инструмента и размера контракта.
        # Для симуляции можно просто определить размер позиции так, чтобы риск был соблюден.
        # Размер позиции = Сумма риска в валюте / (Риск в пунктах * Стоимость пункта)
        # Если стоимость пункта = 1, то Размер позиции = Сумма риска в валюте / Риск в пунктах
        # Риск в пунктах = risk_per_unit / self.point_size
        
        # Упрощенный расчет для симуляции:
        # Предположим, что position_size - это количество единиц, где изменение цены на 1 пункт
        # для 1 единицы position_size приводит к изменению на self.point_size * value_per_point.
        # Для простоты симуляции, пусть value_per_point = 1 (т.е. 1 USD за пункт на 1 лот)
        # Тогда risk_in_currency_per_lot = risk_per_unit
        # position_size (в лотах) = risk_amount_currency / risk_in_currency_per_lot
        
        # Еще проще: размер позиции в единицах, где каждая единица соответствует point_size
        # position_size = risk_amount_currency / risk_per_unit 
        # Этот размер будет означать, что при движении на `risk_per_unit` убыток будет `risk_amount_currency`.
        
        try:
            position_size = risk_amount_currency / risk_per_unit
            # Округление размера позиции (например, до 2 знаков для стандартных лотов)
            # Для простой симуляции можно не округлять или округлять до целых, если это акции
            # position_size = round(position_size, 2) # Пример для Forex лотов
            self.logger.info(f"Расчет размера позиции: баланс={self.account_balance_simulated}, "
                             f"риск={self.risk_per_trade_percentage*100}%, "
                             f"сумма риска={risk_amount_currency:.2f}, "
                             f"риск на ед.={risk_per_unit:.{self._get_decimals()}f}, "
                             f"размер позиции={position_size:.4f}")
            return position_size if position_size > 0 else None
        except ZeroDivisionError:
            self.logger.error("Ошибка деления на ноль при расчете размера позиции (risk_per_unit равен нулю).")
            return None


    def execute_trade(self, signal: dict) -> dict | None:
        """
        "Исполняет" сделку на основе сигнала (симуляция).
        Записывает информацию о входе в БД.

        Args:
            signal (dict): Словарь с информацией о сигнале.
                           Ожидает ключи 'direction', 'entry_price', 'stop_loss', 'take_profit',
                           'symbol', 'reason_for_entry', 'rr_ratio'.

        Returns:
            dict | None: Словарь с деталями открытой позиции для внутреннего учета бота,
                         или None в случае ошибки.
        """
        if not signal or not all(k in signal for k in ['direction', 'entry_price', 'stop_loss', 'take_profit', 'symbol']):
            self.logger.error(f"Неполные данные в сигнале для исполнения: {signal}")
            return None

        entry_price = signal['entry_price']
        stop_loss_price = signal['stop_loss']
        take_profit_price = signal['take_profit']
        direction = signal['direction']
        symbol = signal['symbol'] # Используем символ из сигнала
        reason_for_entry = signal.get('reason_for_entry', '')
        rr_ratio = signal.get('rr_ratio')

        # Расчет размера позиции
        position_size = self._calculate_position_size(entry_price, stop_loss_price)
        if position_size is None or position_size <= 0:
            self.logger.error(f"Не удалось рассчитать корректный размер позиции для сигнала: {signal}")
            return None

        entry_time = datetime.utcnow()

        # Формируем запись для БД и для внутреннего списка открытых позиций бота
        trade_entry_details = {
            'symbol': symbol,
            'entry_time': entry_time,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'position_size': position_size,
            'status': 'open', # Статус при открытии
            'reason_for_entry': reason_for_entry,
            'risk_reward_ratio': rr_ratio,
            'price_context': signal.get('price_context_at_entry'), # Нужно добавить в сигнал, если есть
            'session': signal.get('session_at_entry') # Нужно добавить в сигнал, если есть
        }

        # Логирование в БД
        trade_id_db = self.db.log_trade_entry(trade_entry_details)

        if trade_id_db is not None:
            self.logger.info(f"Сделка ({direction} {symbol} @ {entry_price:.{self._get_decimals()}f}) симулирована и записана в БД с ID: {trade_id_db}.")
            # Возвращаем детали для добавления в список активных позиций бота
            # Добавляем ID из БД для связи
            internal_position_details = trade_entry_details.copy()
            internal_position_details['id_db'] = trade_id_db # ID из базы данных
            internal_position_details['id_internal'] = f"sim_{trade_id_db}_{int(entry_time.timestamp())}" # Уникальный внутренний ID
            return internal_position_details
        else:
            self.logger.error(f"Не удалось записать сделку в БД для сигнала: {signal}")
            return None

    def _get_decimals(self) -> int:
        """Вспомогательный метод для определения количества знаков после запятой."""
        # Можно использовать self.point_size, который был передан при инициализации
        if self.point_size == 0.0001: return 5 # Forex
        elif self.point_size == 0.1: return 2  # Gold
        elif self.point_size == 1: return 1    # Index
        return 5 # Default