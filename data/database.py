# data/database.py

import sqlite3
import logging
from datetime import datetime

class TradingDatabase:
    """
    Класс для взаимодействия с базой данных SQLite для логирования сделок.
    """
    def __init__(self, db_name="trading_stats.db"):
        """
        Инициализация соединения с базой данных.

        Args:
            db_name (str): Имя файла базы данных.
        """
        self.db_name = db_name
        self.logger = logging.getLogger(f"{__name__}.TradingDatabase") # Используем имя модуля для логгера
        try:
            self.conn = sqlite3.connect(self.db_name, check_same_thread=False) # check_same_thread для возможной многопоточности
            self.cursor = self.conn.cursor()
            self._create_tables()
            self.logger.info(f"Соединение с базой данных {self.db_name} установлено.")
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка при подключении к SQLite {self.db_name}: {e}", exc_info=True)
            self.conn = None
            self.cursor = None

    def _create_tables(self):
        """
        Создает таблицу для сделок, если она еще не существует.
        """
        if not self.cursor:
            self.logger.error("Невозможно создать таблицы: курсор базы данных отсутствует.")
            return
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    direction TEXT NOT NULL, -- 'long' или 'short'
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    position_size REAL,
                    profit_loss REAL, -- PnL в валюте
                    pnl_points REAL,  -- PnL в пунктах
                    status TEXT,      -- 'open', 'closed_sl', 'closed_tp', 'closed_manual'
                    reason_for_entry TEXT,
                    reason_for_exit TEXT,
                    risk_reward_ratio REAL,
                    price_context TEXT, -- Контекст рынка при входе
                    session TEXT        -- Торговая сессия при входе
                )
            ''')
            self.conn.commit()
            self.logger.debug("Таблица 'trades' проверена/создана.")
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка при создании таблицы 'trades': {e}", exc_info=True)

    def log_trade_entry(self, trade_data: dict):
        """
        Логирует открытие новой сделки.

        Args:
            trade_data (dict): Словарь с данными по сделке.
                Пример: {'symbol': 'EURUSD', 'entry_time': datetime.now(), ... 'status': 'open'}
        Returns:
            int or None: ID вставленной записи или None в случае ошибки.
        """
        if not self.conn or not self.cursor:
            self.logger.error("Невозможно логировать сделку: нет соединения с БД.")
            return None

        # Преобразуем datetime в строки ISO для SQLite
        if 'entry_time' in trade_data and isinstance(trade_data['entry_time'], datetime):
            trade_data['entry_time'] = trade_data['entry_time'].isoformat()

        columns = ', '.join(trade_data.keys())
        placeholders = ', '.join('?' * len(trade_data))
        sql = f'INSERT INTO trades ({columns}) VALUES ({placeholders})'
        values = tuple(trade_data.values())

        try:
            self.cursor.execute(sql, values)
            self.conn.commit()
            last_id = self.cursor.lastrowid
            self.logger.info(f"Сделка ID {last_id} ({trade_data.get('symbol')} {trade_data.get('direction')}) записана в БД со статусом 'open'.")
            return last_id
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка при логировании входа в сделку: {e}\nSQL: {sql}\nValues: {values}", exc_info=True)
            return None

    def update_trade_exit(self, trade_id: int, exit_time: datetime, exit_price: float,
                            status: str, pnl_currency: float, pnl_points: float, reason_for_exit: str = None):
        """
        Обновляет информацию о закрытии сделки.
        """
        if not self.conn or not self.cursor:
            self.logger.error(f"Невозможно обновить сделку ID {trade_id}: нет соединения с БД.")
            return False

        sql = """
            UPDATE trades
            SET exit_time = ?, exit_price = ?, status = ?, profit_loss = ?, pnl_points = ?, reason_for_exit = ?
            WHERE id = ? AND status = 'open'
        """
        try:
            self.cursor.execute(sql, (exit_time.isoformat(), exit_price, status,
                                     pnl_currency, pnl_points, reason_for_exit, trade_id))
            self.conn.commit()
            if self.cursor.rowcount > 0:
                self.logger.info(f"Сделка ID {trade_id} обновлена: выход по {exit_price}, статус {status}, PnL: {pnl_currency:.2f} (валюта), {pnl_points:.2f} (пункты).")
                return True
            else:
                self.logger.warning(f"Сделка ID {trade_id} не найдена или уже закрыта. Обновление не выполнено.")
                return False
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка при обновлении выхода из сделки ID {trade_id}: {e}", exc_info=True)
            return False

    def get_open_trades(self, symbol: str = None):
        """
        Получает все открытые сделки, опционально фильтруя по символу.
        """
        if not self.conn or not self.cursor:
            self.logger.error("Невозможно получить открытые сделки: нет соединения с БД.")
            return []
        try:
            if symbol:
                self.cursor.execute("SELECT * FROM trades WHERE status = 'open' AND symbol = ?", (symbol,))
            else:
                self.cursor.execute("SELECT * FROM trades WHERE status = 'open'")
            
            trades = []
            columns = [desc[0] for desc in self.cursor.description]
            for row in self.cursor.fetchall():
                trade = dict(zip(columns, row))
                # Преобразование строковых дат обратно в datetime объекты
                if trade.get('entry_time'):
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                if trade.get('exit_time') and isinstance(trade['exit_time'], str):
                     trade['exit_time'] = datetime.fromisoformat(trade['exit_time'])
                trades.append(trade)
            return trades
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка при получении открытых сделок: {e}", exc_info=True)
            return []

    def get_trade_stats(self, symbol: str = None, start_date: datetime = None, end_date: datetime = None):
        """
        Получает статистику по сделкам.
        (Эта функция может быть более сложной, пока что пример)
        """
        if not self.conn or not self.cursor:
            self.logger.error("Невозможно получить статистику: нет соединения с БД.")
            return {}

        query = "SELECT COUNT(*) as total_trades, SUM(profit_loss) as total_pnl FROM trades WHERE status != 'open'"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())
        try:
            self.cursor.execute(query, tuple(params))
            row = self.cursor.fetchone()
            if row:
                stats = {'total_trades': row[0], 'total_pnl': row[1] if row[1] is not None else 0}
                self.logger.info(f"Статистика по сделкам: {stats}")
                return stats
            return {'total_trades': 0, 'total_pnl': 0}
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка при получении статистики по сделкам: {e}", exc_info=True)
            return {}

    def close(self):
        """
        Закрывает соединение с базой данных.
        """
        if self.conn:
            try:
                self.conn.close()
                self.logger.info(f"Соединение с базой данных {self.db_name} закрыто.")
            except sqlite3.Error as e:
                self.logger.error(f"Ошибка при закрытии соединения с БД {self.db_name}: {e}", exc_info=True)