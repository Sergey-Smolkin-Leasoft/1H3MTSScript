# analysis/market_analyzer.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, time, timedelta, timezone

from sklearn.linear_model import LinearRegression # <--- ИСПРАВЛЕННЫЙ ИМПОРТ
# from scipy.stats import linregress # Убедитесь, что этот импорт есть, если вы его используете для linregress

from config import settings # Для SESSIONS

class MarketAnalyzer:
    """
    Анализирует рыночный контекст на основе движения цены за последние 2 дня,
    анализа Азиатской сессии и ее взаимодействия с PDL/PDH.
    """
    def __init__(self, sessions: dict = settings.SESSIONS):
        self.logger = logging.getLogger(__name__)
        self.sessions = sessions
        self.asia_session_name = 'asia' # Ключ для азиатской сессии в settings.SESSIONS

    def _get_session_times_utc(self, session_name: str) -> tuple[time | None, time | None]:
        """Получает время начала и конца сессии в UTC."""
        session_info = self.sessions.get(session_name)
        if not session_info:
            self.logger.warning(f"Информация о сессии '{session_name}' не найдена в настройках.")
            return None, None
        # Убедимся, что start и end являются целыми числами перед созданием time
        start_hour = session_info.get('start')
        end_hour = session_info.get('end')

        if not isinstance(start_hour, int) or not isinstance(end_hour, int):
            self.logger.error(f"Некорректные значения start/end для сессии {session_name}: {start_hour}, {end_hour}")
            return None, None

        return time(start_hour, 0, 0, tzinfo=timezone.utc), \
               time(end_hour, 0, 0, tzinfo=timezone.utc)


    def _get_relevant_candles(self, data_df: pd.DataFrame, target_date: datetime.date, session_name: str) -> pd.DataFrame:
        """
        Извлекает свечи для указанной сессии и даты из DataFrame.
        DataFrame должен иметь DatetimeIndex в UTC.
        """
        if data_df is None or data_df.empty:
            return pd.DataFrame()
        
        # Убедимся, что индекс DataFrame является DatetimeIndex
        if not isinstance(data_df.index, pd.DatetimeIndex):
            self.logger.error("data_df.index не является DatetimeIndex в _get_relevant_candles.")
            return pd.DataFrame()

        start_time_utc, end_time_utc = self._get_session_times_utc(session_name)
        if not start_time_utc or not end_time_utc:
            return pd.DataFrame()

        # Убедимся, что индекс DataFrame осведомлен о таймзоне или является UTC
        df_index_utc = data_df.index
        if df_index_utc.tzinfo is None:
            df_index_utc = df_index_utc.tz_localize('UTC')
        elif df_index_utc.tzinfo != timezone.utc:
            df_index_utc = df_index_utc.tz_convert('UTC')
        
        # Создаем временный DataFrame с корректным UTC индексом для фильтрации
        # Это нужно, если оригинальный data_df.index изменяется где-то еще (что не должно быть)
        # или если мы хотим быть уверены в работе с UTC-индексом локально.
        # Если data_df.index уже правильный, можно использовать его напрямую.
        # temp_df_for_filtering = data_df.set_index(df_index_utc) # Это может быть избыточно если data_df.index уже UTC

        target_date_candles = data_df[df_index_utc.date == target_date]


        if target_date_candles.empty:
            return pd.DataFrame()

        # Фильтруем по времени сессии
        if start_time_utc < end_time_utc: # Сессия не пересекает полночь
            # Используем .loc с индексом target_date_candles для избежания SettingWithCopyWarning
            session_candles_idx = target_date_candles.index.indexer_between_time(start_time_utc, end_time_utc, include_end=False)
            session_candles = target_date_candles.iloc[session_candles_idx]
        else: # Сессия пересекает полночь
            self.logger.warning(f"Сессия {session_name} ({start_time_utc}-{end_time_utc}) пересекает полночь в UTC. Такая логика не полностью обработана для _get_relevant_candles.")
            # Для простоты, если Азия в UTC с 01 до 09, этот блок else не должен выполняться.
            # Если он выполняется, значит настройки сессий или их интерпретация могут быть неверны.
            return pd.DataFrame() # Возвращаем пустой DataFrame, если логика сложная/неопределенная

        return session_candles


    def _determine_primary_context(self, data_1h: pd.DataFrame) -> str:
        """
        Определяет основной контекст (long/short/neutral) на основе анализа High/Low
        за последние 2 торговых дня (примерно 48 часов).
        """
        if data_1h is None or len(data_1h) < 20: # Нужно достаточно данных для анализа тренда
            self.logger.warning("Недостаточно данных 1H для определения основного контекста.")
            return 'neutral'

        relevant_data = data_1h.tail(48)
        if len(relevant_data) < 20:
            self.logger.warning("Мало данных в relevant_data для основного контекста.")
            return 'neutral'

        highs = relevant_data['high']
        lows = relevant_data['low']
        
        if len(highs) > 1 and len(lows) > 1: # Убедимся, что есть хотя бы 2 точки для регрессии
            x = np.arange(len(highs)).reshape(-1, 1)
            try:
                # Регрессия по максимумам
                model_high = LinearRegression().fit(x, highs.values.reshape(-1, 1))
                slope_high = model_high.coef_[0][0]

                # Регрессия по минимумам
                model_low = LinearRegression().fit(x, lows.values.reshape(-1, 1)) # Используем ту же длину x, т.к. highs и lows из одного df
                slope_low = model_low.coef_[0][0]
                
                self.logger.debug(f"Первичный контекст: Наклон Highs={slope_high:.5f}, Наклон Lows={slope_low:.5f}")

                avg_price_in_window = relevant_data['close'].mean()
                threshold = avg_price_in_window * 0.00005 

                if slope_high > threshold and slope_low > threshold:
                    return 'long'
                elif slope_high < -threshold and slope_low < -threshold:
                    return 'short'
                else:
                    return 'neutral'
            except Exception as e:
                self.logger.error(f"Ошибка при расчете регрессии для основного контекста: {e}", exc_info=True)
                return 'neutral'
        
        return 'neutral'

    def _analyze_asian_session_details(self, data_3m: pd.DataFrame, current_date_utc: datetime.date,
                                     primary_context: str, pdl: float | None, pdh: float | None) -> dict:
        """
        Анализирует детали Азиатской сессии.
        """
        results = {
            'asia_trend': 'ranging',
            'is_sync_with_primary': False,
            'pdl_pdh_rejection_skip_signal': False,
            'asia_high': None,
            'asia_low': None,
            'asia_close': None,
            'message': ""
        }

        asia_candles = self._get_relevant_candles(data_3m, current_date_utc, self.asia_session_name)

        if asia_candles.empty:
            results['message'] = f"Нет данных по Азиатской сессии ({self.asia_session_name}) за {current_date_utc}."
            self.logger.info(results['message'])
            return results

        results['asia_high'] = float(asia_candles['high'].max())
        results['asia_low'] = float(asia_candles['low'].min())
        results['asia_close'] = float(asia_candles['close'].iloc[-1])
        
        self.logger.info(f"Азия ({current_date_utc}): High={results['asia_high']:.5f}, Low={results['asia_low']:.5f}, Close={results['asia_close']:.5f}")

        asia_open = float(asia_candles['open'].iloc[0])
        asia_body = abs(results['asia_close'] - asia_open)
        asia_range = results['asia_high'] - results['asia_low']

        if asia_range > 0.00001 and asia_body / asia_range > 0.3: # Добавлена проверка asia_range > 0
            if results['asia_close'] > asia_open:
                results['asia_trend'] = 'bullish'
            elif results['asia_close'] < asia_open:
                results['asia_trend'] = 'bearish'
        
        self.logger.info(f"Тренд Азии: {results['asia_trend']}")

        if (primary_context == 'long' and results['asia_trend'] == 'bullish') or \
           (primary_context == 'short' and results['asia_trend'] == 'bearish'):
            results['is_sync_with_primary'] = True
        
        self.logger.info(f"Азия синхронизирована с основным контекстом ('{primary_context}'): {results['is_sync_with_primary']}")

        skip_reason_pdl_pdh = ""
        for idx, row in asia_candles.iterrows(): # Используем iterrows для доступа к свечам
            candle_high = float(row['high'])
            candle_low = float(row['low'])
            candle_close = float(row['close'])
            candle_time = idx.time() if isinstance(idx, pd.Timestamp) else "N/A"


            if primary_context == 'long' and pdh is not None:
                if candle_high > pdh and candle_close < pdh:
                    results['pdl_pdh_rejection_skip_signal'] = True
                    skip_reason_pdl_pdh = f"Азия ({candle_time}) показала отбой от PDH ({pdh:.5f}) против long-контекста (H={candle_high:.5f}, C={candle_close:.5f})."
                    self.logger.info(skip_reason_pdl_pdh)
                    break 
            
            elif primary_context == 'short' and pdl is not None:
                if candle_low < pdl and candle_close > pdl:
                    results['pdl_pdh_rejection_skip_signal'] = True
                    skip_reason_pdl_pdh = f"Азия ({candle_time}) показала отбой от PDL ({pdl:.5f}) против short-контекста (L={candle_low:.5f}, C={candle_close:.5f})."
                    self.logger.info(skip_reason_pdl_pdh)
                    break
        results['message'] += skip_reason_pdl_pdh if results['message'] == "" else f"; {skip_reason_pdl_pdh}"


        return results

    def _check_asia_hl_breakout_confirmation(self, data_3m_post_asia: pd.DataFrame, 
                                           asia_high: float | None, asia_low: float | None,
                                           primary_context: str) -> dict:
        breakout_results = {
            'breakout_confirms_primary': False,
            'breakout_direction': None, 
            'breakout_price': None,
            'breakout_time': None,
            'message': ""
        }

        if data_3m_post_asia.empty or asia_high is None or asia_low is None:
            breakout_results['message'] = "Нет данных после Азии или не определены H/L Азии для проверки пробоя."
            # self.logger.debug(breakout_results['message']) # Можно логировать, если это ожидаемо
            return breakout_results

        for idx, row in data_3m_post_asia.iterrows():
            candle_close = float(row['close'])
            candle_time = idx.time() if isinstance(idx, pd.Timestamp) else "N/A"

            if primary_context == 'long':
                if candle_close > asia_high:
                    breakout_results['breakout_confirms_primary'] = True
                    breakout_results['breakout_direction'] = 'bullish'
                    breakout_results['breakout_price'] = candle_close
                    breakout_results['breakout_time'] = candle_time
                    breakout_results['message'] = f"Подтверждение Long: Закрытие ({candle_close:.5f} в {candle_time}) выше Asia High ({asia_high:.5f})."
                    self.logger.info(breakout_results['message'])
                    break
            elif primary_context == 'short':
                if candle_close < asia_low:
                    breakout_results['breakout_confirms_primary'] = True
                    breakout_results['breakout_direction'] = 'bearish'
                    breakout_results['breakout_price'] = candle_close
                    breakout_results['breakout_time'] = candle_time
                    breakout_results['message'] = f"Подтверждение Short: Закрытие ({candle_close:.5f} в {candle_time}) ниже Asia Low ({asia_low:.5f})."
                    self.logger.info(breakout_results['message'])
                    break
        
        return breakout_results
    
    def _analyze_order_flow_confirmation(self, data_1h: pd.DataFrame, data_3m: pd.DataFrame,
                                        primary_context: str, 
                                        pdl: float | None, pdh: float | None,
                                        ) -> dict:
        of_confirmation_results = {
            'of_target_hit_and_confirmed': False,
            'confirmation_direction': None,
            'message': "Логика OF confirmation не реализована полностью."
        }
        self.logger.debug(of_confirmation_results['message'])
        return of_confirmation_results


    def analyze(self, data_1h: pd.DataFrame | None, data_3m: pd.DataFrame | None, 
                pdl: float | None, pdh: float | None) -> dict:
        self.logger.info(f"Запуск анализа рынка. PDL={pdl}, PDH={pdh}")
        current_utc_datetime = datetime.now(timezone.utc)
        current_utc_date = current_utc_datetime.date()

        # Инициализация структуры результатов по умолчанию
        analysis_output = {
            'timestamp': current_utc_datetime.isoformat(),
            'primary_context': 'neutral',
            'asian_session_analysis': {
                'asia_trend': 'ranging', 'is_sync_with_primary': False,
                'pdl_pdh_rejection_skip_signal': False, 'asia_high': None,
                'asia_low': None, 'asia_close': None, 'message': "Анализ Азии не проводился (нет данных)."
            },
            'asia_hl_breakout_confirmation': {
                'breakout_confirms_primary': False, 'breakout_direction': None,
                'breakout_price': None, 'breakout_time': None, 'message': "Проверка пробоя Азии не проводилась."
            },
            'of_confirmation': self._analyze_order_flow_confirmation(data_1h, data_3m, 'neutral', pdl, pdh), # Заглушка
            'derived_info': {
                'skip_trade_due_to_asia_rejection': False,
                'asia_confirms_primary_context': False,
            }
        }
        
        if data_1h is None or data_1h.empty:
            self.logger.warning("Отсутствуют данные 1H для анализа.")
            return analysis_output # Возвращаем дефолтные значения

        primary_context = self._determine_primary_context(data_1h)
        analysis_output['primary_context'] = primary_context
        self.logger.info(f"Определен основной контекст: {primary_context}")

        if data_3m is None or data_3m.empty:
            self.logger.warning("Отсутствуют данные 3M для полного анализа Азии и пробоев.")
            # primary_context уже установлен, остальное останется по умолчанию
            return analysis_output


        asian_session_analysis_results = self._analyze_asian_session_details(
            data_3m, current_utc_date, primary_context, pdl, pdh
        )
        analysis_output['asian_session_analysis'] = asian_session_analysis_results

        asia_end_time_utc_obj, _ = self._get_session_times_utc(self.asia_session_name)
        post_asia_candles = pd.DataFrame()

        if asia_end_time_utc_obj: # Проверяем, что время конца Азии определено
            # Убедимся, что индекс DataFrame data_3m осведомлен о таймзоне или является UTC
            df_3m_index_utc = data_3m.index
            if df_3m_index_utc.tzinfo is None:
                df_3m_index_utc = df_3m_index_utc.tz_localize('UTC')
            elif df_3m_index_utc.tzinfo != timezone.utc:
                df_3m_index_utc = df_3m_index_utc.tz_convert('UTC')

            # Создаем datetime объект для конца Азии сегодня
            # Используем combine для корректной работы с датой и временем
            asia_end_datetime_utc = datetime.combine(current_utc_date, asia_end_time_utc_obj)
            if asia_end_datetime_utc.tzinfo is None : # Если combine сбросил tzinfo
                 asia_end_datetime_utc = asia_end_datetime_utc.replace(tzinfo=timezone.utc)


            # Фильтруем свечи, которые строго после конца Азии и в пределах следующих ~4 часов
            post_asia_start_filter = df_3m_index_utc > asia_end_datetime_utc
            post_asia_end_filter = df_3m_index_utc < (asia_end_datetime_utc + timedelta(hours=4))
            
            # Также убедимся, что это свечи текущего дня
            current_day_filter = df_3m_index_utc.date == current_utc_date

            post_asia_candles = data_3m[post_asia_start_filter & post_asia_end_filter & current_day_filter]


        asia_hl_breakout_results = self._check_asia_hl_breakout_confirmation(
            post_asia_candles,
            asian_session_analysis_results['asia_high'],
            asian_session_analysis_results['asia_low'],
            primary_context
        )
        analysis_output['asia_hl_breakout_confirmation'] = asia_hl_breakout_results
        
        analysis_output['of_confirmation'] = self._analyze_order_flow_confirmation(
            data_1h, data_3m, primary_context, pdl, pdh
        )

        analysis_output['derived_info']['skip_trade_due_to_asia_rejection'] = asian_session_analysis_results['pdl_pdh_rejection_skip_signal']
        analysis_output['derived_info']['asia_confirms_primary_context'] = asia_hl_breakout_results['breakout_confirms_primary']
        
        self.logger.info(f"Анализ рынка завершен. Итоговый основной контекст: {analysis_output['primary_context']}.")
        self.logger.debug(f"Полные результаты анализа: {analysis_output}")
        return analysis_output