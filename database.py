import sqlite3
from datetime import datetime

class TradingDatabase:
    def __init__(self, db_name='trading_stats.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        # Таблица для хранения информации о сделках
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                direction TEXT NOT NULL,  -- 'long' или 'short'
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                profit_loss REAL,
                price_context TEXT,
                session TEXT,
                fractal_type TEXT,
                is_profitable BOOLEAN,
                risk_reward_ratio REAL,
                volatility_index REAL,
                market_condition TEXT
            )
        ''')

        # Таблица для статистики по сессиям
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS session_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                session TEXT NOT NULL,
                trades_count INTEGER,
                profitable_trades INTEGER,
                total_profit_loss REAL,
                average_profit_loss REAL,
                win_rate REAL,
                max_drawdown REAL,
                volatility_index REAL
            )
        ''')

        # Таблица для ежедневной статистики
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_trades INTEGER,
                profitable_trades INTEGER,
                total_profit_loss REAL,
                average_profit_loss REAL,
                win_rate REAL,
                max_drawdown REAL,
                volatility_index REAL,
                daily_limit REAL
            )
        ''')

        # Таблица для статистики по инструментам
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS instrument_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                trades_count INTEGER,
                profitable_trades INTEGER,
                total_profit_loss REAL,
                average_profit_loss REAL,
                win_rate REAL,
                volatility_index REAL
            )
        ''')

        self.conn.commit()

    def log_trade(self, trade_data):
        """
        Записывает информацию о сделке в базу данных
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (
                symbol, entry_time, exit_time, direction, entry_price, exit_price,
                stop_loss, take_profit, position_size, profit_loss, price_context,
                session, fractal_type, is_profitable, risk_reward_ratio,
                volatility_index, market_condition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['symbol'],
            trade_data['entry_time'],
            trade_data.get('exit_time', None),
            trade_data['direction'],
            trade_data['entry_price'],
            trade_data.get('exit_price', None),
            trade_data.get('stop_loss', None),
            trade_data.get('take_profit', None),
            trade_data['position_size'],
            trade_data.get('profit_loss', 0),
            trade_data.get('price_context', None),
            trade_data.get('session', None),
            trade_data.get('fractal_type', None),
            trade_data.get('is_profitable', False),
            trade_data.get('risk_reward_ratio', None),
            trade_data.get('volatility_index', None),
            trade_data.get('market_condition', None)
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_trade_stats(self, symbol=None, start_date=None, end_date=None):
        """
        Получает статистику по сделкам
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)

        cursor = self.conn.execute(query, params)
        return cursor.fetchall()

    def close(self):
        self.conn.close()
