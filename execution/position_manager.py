# execution/position_manager.py

import logging
from datetime import datetime
from data.database import TradingDatabase # Для обновления статуса сделок в БД
from config import settings # Может понадобиться для специфичных правил закрытия

class PositionManager:
    """
    Управляет открытыми позициями: проверяет SL/TP, обновляет статус.
    """
    def __init__(self, symbol: str, point_size: float, db: TradingDatabase):
        """
        Инициализация менеджера позиций.

        Args:
            symbol (str): Текущий торговый символ (может быть не нужен, если позиции содержат символ).
            point_size (float): Размер пункта.
            db (TradingDatabase): Экземпляр TradingDatabase.
        """
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol # Используется для логирования или если позиции не содержат символ
        self.point_size = point_size
        self.db = db

    def manage_simulated_positions(self, open_positions_list: list, current_price: float | None):
        """
        Проверяет список симулированных открытых позиций на достижение SL или TP.
        Модифицирует `open_positions_list` на месте, удаляя закрытые позиции.

        Args:
            open_positions_list (list): Список словарей, представляющих открытые позиции.
                                       Каждый словарь должен содержать 'id_db', 'direction',
                                       'entry_price', 'stop_loss', 'take_profit', 'position_size'.
            current_price (float | None): Текущая рыночная цена.
        """
        if current_price is None:
            self.logger.warning("Текущая цена не предоставлена, управление позициями невозможно.")
            return

        # Итерируемся по копии списка или в обратном порядке, если удаляем элементы
        for i in range(len(open_positions_list) - 1, -1, -1):
            position = open_positions_list[i]

            if position.get('status') != 'open': # Пропускаем уже не активные
                continue

            direction = position['direction']
            entry_price = position['entry_price']
            sl_price = position['stop_loss']
            tp_price = position['take_profit']
            position_size = position['position_size']
            trade_id_db = position.get('id_db') # ID из базы данных

            exit_reason = None
            exit_status = None
            actual_exit_price = current_price # По умолчанию, если не SL/TP

            # Проверка SL
            if direction == 'long' and current_price <= sl_price:
                exit_reason = f"Stop Loss hit at {sl_price:.{self._get_decimals()}f}"
                exit_status = "closed_sl"
                actual_exit_price = sl_price # Выход точно по SL
            elif direction == 'short' and current_price >= sl_price:
                exit_reason = f"Stop Loss hit at {sl_price:.{self._get_decimals()}f}"
                exit_status = "closed_sl"
                actual_exit_price = sl_price

            # Проверка TP (если SL еще не сработал)
            if not exit_reason:
                if direction == 'long' and current_price >= tp_price:
                    exit_reason = f"Take Profit hit at {tp_price:.{self._get_decimals()}f}"
                    exit_status = "closed_tp"
                    actual_exit_price = tp_price # Выход точно по TP
                elif direction == 'short' and current_price <= tp_price:
                    exit_reason = f"Take Profit hit at {tp_price:.{self._get_decimals()}f}"
                    exit_status = "closed_tp"
                    actual_exit_price = tp_price
            
            # Если позиция должна быть закрыта
            if exit_reason and trade_id_db is not None:
                exit_time = datetime.utcnow()
                
                # Расчет PnL
                pnl_points = 0
                if direction == 'long':
                    pnl_points = (actual_exit_price - entry_price) / self.point_size
                else: # short
                    pnl_points = (entry_price - actual_exit_price) / self.point_size
                
                # Симулированный PnL в валюте (упрощенно)
                # (actual_exit_price - entry_price) * position_size для long
                # (entry_price - actual_exit_price) * position_size для short
                # Знак position_size может уже учитывать направление, или нужно явно
                pnl_currency = (actual_exit_price - entry_price) * position_size if direction == 'long' else \
                               (entry_price - actual_exit_price) * position_size

                self.logger.info(f"Позиция ID_DB {trade_id_db} ({position.get('symbol')} {direction}) закрывается. Причина: {exit_reason}. "
                                 f"Цена выхода: {actual_exit_price:.{self._get_decimals()}f}. PnL: {pnl_points:.1f} пт, {pnl_currency:.2f} у.е.")

                # Обновление в БД
                updated_in_db = self.db.update_trade_exit(
                    trade_id=trade_id_db,
                    exit_time=exit_time,
                    exit_price=actual_exit_price,
                    status=exit_status,
                    pnl_currency=pnl_currency,
                    pnl_points=pnl_points,
                    reason_for_exit=exit_reason
                )

                if updated_in_db:
                    # Удаляем позицию из активного списка бота
                    open_positions_list.pop(i)
                    self.logger.debug(f"Позиция ID_DB {trade_id_db} удалена из списка активных симулируемых позиций.")
                else:
                    self.logger.error(f"Не удалось обновить статус закрытия для сделки ID_DB {trade_id_db} в БД.")
            # Здесь можно добавить логику для Trailing Stop, безубытка и т.д.

    def _get_decimals(self) -> int:
        """Вспомогательный метод для определения количества знаков после запятой."""
        if self.point_size == 0.0001: return 5
        elif self.point_size == 0.1: return 2
        elif self.point_size == 1: return 1
        return 5