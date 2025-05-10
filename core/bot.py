# core/bot.py

import logging
from datetime import datetime

# --- Импорты из других модулей ---
from config import settings
from utils.logging_config import setup_logger
from data.data_handler import DataHandler
from data.database import TradingDatabase
from analysis.market_analyzer import MarketAnalyzer
from analysis.signal_generator import SignalGenerator
from execution.trade_executor import TradeExecutor
from execution.position_manager import PositionManager
from visualization.plotter import Plotter

class TradingBot1H3M:
    """
    Основной класс торгового бота 1H3M.
    Координирует загрузку данных, анализ, генерацию сигналов, исполнение и управление позициями.
    """
    def __init__(self, symbol: str):
        """
        Инициализация бота.

        Args:
            symbol (str): Торговый символ (например, 'EURUSD').
        """
        self.logger = setup_logger() 
        self.logger.info(f"Инициализация экземпляра бота для символа: {symbol}...")

        if symbol not in settings.INSTRUMENT_SETTINGS:
            critical_error_msg = f"КРИТИЧЕСКАЯ ОШИБКА: Неподдерживаемый символ '{symbol}' не найден в config/settings.py"
            print(critical_error_msg) # Используем print, т.к. логгер может быть не до конца настроен
            self.logger.critical(critical_error_msg) 
            raise ValueError(critical_error_msg)

        self.symbol = symbol
        self._load_instrument_settings() 

        # Инициализация компонентов
        self.db = TradingDatabase(settings.DATABASE_NAME)
        self.data_handler = DataHandler(
            settings.TWELVE_DATA_API_KEY,
            settings.SYMBOL_MAPPING_TWELVE_DATA
        )
        self.analyzer = MarketAnalyzer( # Конструктор MarketAnalyzer теперь принимает только sessions
            sessions=settings.SESSIONS
        )
        self.signal_generator = SignalGenerator(
            fractal_window=settings.FRACTAL_WINDOW,
            min_rr=settings.MIN_RR_RATIO,
            sessions=settings.SESSIONS,
            instrument_settings=settings.INSTRUMENT_SETTINGS
        )
        self.executor = TradeExecutor(self.symbol, self.point_size, self.db)
        self.position_manager = PositionManager(self.symbol, self.point_size, self.db)
        self.plotter = Plotter(self.symbol, self.point_size)

        # Инициализация состояния
        self.data_1h = None
        self.data_3m = None
        
        self.pdl: float | None = None
        self.pdh: float | None = None
        
        self.analysis_results: dict = {
            'primary_context': 'neutral',
            'asian_session_analysis': {},
            'asia_hl_breakout_confirmation': {},
            'of_confirmation': {}, 
            'derived_info': {}
        }
        self.current_context: str = 'neutral'

        self.daily_limit: float | None = None # <--- ИНИЦИАЛИЗИРУЕМ daily_limit

        self.fractal_levels: list = []
        self.skip_conditions: list = []
        self.open_positions: list = []

        self.logger.info(f"Бот для {self.symbol} инициализирован. Max Target: {self.max_target_points} пт, Point Size: {self.point_size}")

    def _load_instrument_settings(self):
        """Загружает и устанавливает настройки для текущего символа."""
        symbol_specific_settings = settings.INSTRUMENT_SETTINGS.get(self.symbol)
        if symbol_specific_settings:
            self.point_size: float = symbol_specific_settings.get('point_size', 0.0001)
            self.max_target_points: int = symbol_specific_settings.get('max_target_points', 250)
            self.logger.debug(f"Загружены настройки для {self.symbol}: point_size={self.point_size}, max_target_points={self.max_target_points}")
        else:
            self.logger.error(f"Критическая ошибка: Настройки для символа {self.symbol} не найдены во время _load_instrument_settings.")
            self.point_size = 0.0001
            self.max_target_points = 250
    
    def fetch_data(self):
        """Загружает или обновляет рыночные данные и PDL/PDH."""
        self.logger.info(f"Запрос на получение данных для {self.symbol}...")
        try:
            self.data_1h, self.data_3m = self.data_handler.fetch_data(
                self.symbol,
                settings.TIMEFRAME_1H,
                settings.TIMEFRAME_3M, 
                settings.DATA_FETCH_LIMIT
            )
            if self.data_1h is not None and not self.data_1h.empty:
                self.pdl, self.pdh = self.data_handler.update_pdl_pdh(self.data_1h)
                self.logger.info(f"Данные 1H загружены ({len(self.data_1h)} свечей). PDL={self._format_price_display(self.pdl)}, PDH={self._format_price_display(self.pdh)}")
            else:
                 self.logger.warning("Данные 1H не загружены или пусты.")
                 self.pdl, self.pdh = None, None 

            if self.data_3m is not None and not self.data_3m.empty:
                 current_price_display = self.get_current_price()
                 self.logger.info(f"Данные {settings.TIMEFRAME_3M} загружены ({len(self.data_3m)} свечей). Текущая цена: {self._format_price_display(current_price_display)}")
            else:
                 self.logger.warning(f"Данные {settings.TIMEFRAME_3M} не загружены или пусты.")

        except Exception as e:
            self.logger.error(f"Ошибка при получении данных в TradingBot1H3M: {e}", exc_info=True)
            self.data_1h, self.data_3m, self.pdl, self.pdh = None, None, None, None

    def analyze_market_context(self):
        """Анализирует рыночный контекст, вызывая MarketAnalyzer."""
        self.logger.debug("Анализ рыночного контекста...")
        if self.data_1h is None or self.data_3m is None:
            self.logger.warning("Пропуск анализа контекста: данные не загружены.")
            self.analysis_results['primary_context'] = 'neutral' 
            self.current_context = 'neutral'
            return

        try:
            analysis_output = self.analyzer.analyze(
                self.data_1h,
                self.data_3m,
                self.pdl, 
                self.pdh
            )
            self.analysis_results = analysis_output 
            self.current_context = self.analysis_results.get('primary_context', 'neutral') 
            
            self.logger.info(f"Результат анализа: Основной Контекст={self.current_context}")
            self.logger.debug(f"Полные результаты анализа: {self.analysis_results}")

        except Exception as e:
            self.logger.error(f"Ошибка при анализе рыночного контекста: {e}", exc_info=True)
            self.analysis_results['primary_context'] = 'neutral'
            self.current_context = 'neutral'
            
    
    def find_entry_signals(self) -> list:
        """Ищет сигналы для входа, вызывая SignalGenerator."""
        self.logger.debug("Поиск сигналов входа...")
        if self.data_1h is None or self.data_3m is None or self.current_context is None or self.current_context == 'neutral':
            self.logger.warning(f"Пропуск поиска сигналов: данные не загружены или контекст '{self.current_context}'.")
            self.fractal_levels = []
            self.skip_conditions = []
            return []

        try:
            signals, skips, fractals = self.signal_generator.find_signals(
                data_1h=self.data_1h,
                data_3m=self.data_3m,
                analysis_results=self.analysis_results, 
                pdl=self.pdl,
                pdh=self.pdh,
                open_positions=self.open_positions,
                current_symbol=self.symbol
            )
            self.skip_conditions = skips
            self.fractal_levels = fractals
            self.logger.info(f"Поиск сигналов завершен. Найдено сигналов: {len(signals)}, пропущено: {len(skips)}")
            return signals
        except Exception as e:
            self.logger.error(f"Ошибка при поиске сигналов входа: {e}", exc_info=True)
            self.fractal_levels = []
            self.skip_conditions = []
            return []

    def execute_trade(self, signal: dict) -> bool:
        """Выполняет торговый сигнал, вызывая TradeExecutor."""
        self.logger.info(f"Попытка выполнить сделку по сигналу: {signal.get('direction')} {signal.get('symbol')} @ {self._format_price_display(signal.get('entry_price'))}")
        try:
            if 'price_context_at_entry' not in signal:
                 signal['price_context_at_entry'] = self.current_context
            
            opened_position_details = self.executor.execute_trade(signal)
            if opened_position_details:
                self.open_positions.append(opened_position_details)
                self.logger.info(f"Сделка успешно выполнена/симулирована: {opened_position_details.get('id_internal')}")
                return True
            else:
                self.logger.warning("Не удалось выполнить сделку (executor вернул None).")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении сделки: {e}", exc_info=True)
            return False

    def manage_open_positions(self):
        """Управляет открытыми позициями, вызывая PositionManager."""
        self.logger.debug(f"Управление открытыми позициями ({len(self.open_positions)})...")
        if not self.open_positions:
            return

        current_price = self.get_current_price()
        if current_price is None:
            self.logger.warning("Невозможно управлять позициями: текущая цена неизвестна.")
            return

        try:
            self.position_manager.manage_simulated_positions(self.open_positions, current_price)
            self.logger.debug(f"Управление завершено. Активных симулируемых позиций: {len(self.open_positions)}")
        except Exception as e:
            self.logger.error(f"Ошибка при управлении открытыми позициями: {e}", exc_info=True)

    def get_current_price(self) -> float | None:
        """Безопасно возвращает последнюю цену закрытия с младшего таймфрейма."""
        if self.data_3m is not None and not self.data_3m.empty and 'close' in self.data_3m.columns:
            try:
                return float(self.data_3m['close'].iloc[-1])
            except (IndexError, ValueError, TypeError) as e:
                self.logger.debug(f"Не удалось получить текущую цену из data_3m: {e}")
                return None
        return None

    def visualize_strategy(self, save_path: str | None = None):
        """Визуализирует стратегию, вызывая Plotter."""
        self.logger.info("Запрос на визуализацию стратегии...")
        if self.data_1h is None: # Проверяем хотя бы data_1h
            self.logger.warning("Нет данных 1H для визуализации.")
            return
        try:
            asia_high = self.analysis_results.get('asian_session_analysis', {}).get('asia_high')
            asia_low = self.analysis_results.get('asian_session_analysis', {}).get('asia_low')

            fig = self.plotter.plot_strategy(
                data_1h=self.data_1h,
                data_3m=self.data_3m,
                fractals=self.fractal_levels,
                daily_limit=self.daily_limit, # Передаем None, если он не рассчитывается
                market_context=self.current_context,
                open_positions=self.open_positions,
                skipped_signals=self.skip_conditions,
                pdl=self.pdl,
                pdh=self.pdh,
                asia_high=asia_high,
                asia_low=asia_low,
                save_path=save_path
            )
        except Exception as e:
            self.logger.error(f"Ошибка при вызове визуализации стратегии: {e}", exc_info=True)

    def cleanup(self):
        """Выполняет очистку ресурсов при завершении работы."""
        self.logger.info("Очистка ресурсов бота...")
        try:
            if self.db:
                self.db.close()
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии соединения с БД: {e}", exc_info=True)

    def _format_price_display(self, price: float | None) -> str:
        """Форматирует цену для вывода в логи/консоль, используя point_size экземпляра."""
        if price is None:
            return "N/A"
        
        # Определяем количество знаков после запятой на основе self.point_size
        # Это должно быть более надежно, чем передавать point_size в каждую print функцию
        if self.point_size == 0.0001: fmt_str = "{:.5f}" # EURUSD, GBPUSD
        elif self.point_size == 0.001: fmt_str = "{:.3f}" # JPY пары
        elif self.point_size == 0.01: fmt_str = "{:.2f}"  # Некоторые индексы или акции
        elif self.point_size == 0.1: fmt_str = "{:.2f}"   # XAUUSD (цена типа 2010.55 или 2010.5)
        elif self.point_size == 1.0: fmt_str = "{:.1f}"   # Индексы типа GER40
        else: fmt_str = "{:.5f}" # Общее значение по умолчанию для неизвестных
        return fmt_str.format(price)