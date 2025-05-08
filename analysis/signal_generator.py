# analysis/signal_generator.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone, time # <--- ИСПРАВЛЕННАЯ СТРОКА (добавлен time)

from .indicators import calculate_atr
from config import settings

class SignalGenerator:
    def __init__(self, fractal_window: int, min_rr: float, sessions: dict, instrument_settings: dict):
        self.logger = logging.getLogger(__name__)
        self.fractal_window = fractal_window
        self.min_rr = min_rr
        self.sessions = sessions
        self.instrument_settings = instrument_settings

    def _identify_fractals(self, data: pd.DataFrame) -> tuple[list, list]:
        # ... (код без изменений) ...
        bullish_fractals = []
        bearish_fractals = []
        window = self.fractal_window

        if data is None or data.empty or 'high' not in data.columns or 'low' not in data.columns or len(data) <= window * 2:
            self.logger.warning(f"Недостаточно данных ({len(data)} баров) для идентификации фракталов с окном {window}.")
            return bullish_fractals, bearish_fractals

        highs = data['high'].values
        lows = data['low'].values
        timestamps = data.index

        for i in range(window, len(data) - window):
            is_bullish = True
            for j in range(1, window + 1):
                if not (lows[i] < lows[i-j] and lows[i] < lows[i+j]):
                    is_bullish = False
                    break
            if is_bullish:
                bullish_fractals.append({
                    'timestamp': timestamps[i], 'price': float(lows[i]), 'type': 'bullish'
                })

            is_bearish = True
            for j in range(1, window + 1):
                if not (highs[i] > highs[i-j] and highs[i] > highs[i+j]):
                    is_bearish = False
                    break
            if is_bearish:
                bearish_fractals.append({
                    'timestamp': timestamps[i], 'price': float(highs[i]), 'type': 'bearish'
                })
        
        self.logger.debug(f"Найдено бычьих фракталов: {len(bullish_fractals)}, медвежьих: {len(bearish_fractals)}")
        return bullish_fractals, bearish_fractals


    def _filter_fractals_by_session(self, fractals: list, session_name: str, reference_date: datetime = None) -> list:
        filtered = []
        if session_name not in self.sessions:
            self.logger.warning(f"Сессия {session_name} не найдена в настройках.")
            return filtered

        session_info = self.sessions[session_name]
        start_hour_utc_val = session_info.get('start')
        end_hour_utc_val = session_info.get('end')

        if not isinstance(start_hour_utc_val, int) or not isinstance(end_hour_utc_val, int):
            self.logger.error(f"Некорректные значения start/end для сессии {session_name}")
            return filtered

        if reference_date is None:
            reference_date = datetime.now(timezone.utc)
        # Убедимся, что reference_date имеет таймзону UTC
        if reference_date.tzinfo is None or reference_date.tzinfo.utcoffset(reference_date) != timedelta(0):
            reference_date = reference_date.astimezone(timezone.utc)
        
        # Определяем дату, для которой ищем сессию
        target_date = reference_date.date()
        
        # Создаем offset-aware datetime объекты для начала и конца сессии в нужную дату
        try:
            session_start_dt = datetime.combine(target_date, time(start_hour_utc_val, 0), tzinfo=timezone.utc)
            # Конец сессии не включается, поэтому берем начало следующего часа или дня
            # Правильнее будет сравнивать время фрактала со временем начала и конца
            session_end_dt = datetime.combine(target_date, time(end_hour_utc_val, 0), tzinfo=timezone.utc)
            # Обработка сессий, пересекающих полночь (если end_hour < start_hour)
            if session_end_dt <= session_start_dt:
                session_end_dt += timedelta(days=1) # Конец сессии на следующий день
                # В этом случае фракталы могут быть либо >= start в target_date, либо < end в target_date + 1 день
                # Это усложняет фильтрацию, пока предполагаем, что сессия в пределах одного UTC дня (start < end)
                if start_hour_utc_val > end_hour_utc_val:
                     self.logger.debug(f"Сессия {session_name} пересекает полночь UTC.")

        except ValueError as ve:
            self.logger.error(f"Ошибка создания datetime для сессии {session_name}: {ve}")
            return filtered

        for fractal in fractals:
            fractal_ts_orig = fractal['timestamp']
            fractal_ts = None
            try:
                # Конвертируем время фрактала в offset-aware UTC Timestamp
                if isinstance(fractal_ts_orig, pd.Timestamp):
                    fractal_ts = fractal_ts_orig
                else:
                    fractal_ts = pd.to_datetime(fractal_ts_orig)

                if fractal_ts.tzinfo is None:
                    fractal_ts = fractal_ts.tz_localize('UTC')
                else:
                    fractal_ts = fractal_ts.tz_convert('UTC')

                # --- ИЗМЕНЕННОЕ УСЛОВИЕ СРАВНЕНИЯ ---
                # Сравниваем полные offset-aware datetime/timestamp объекты
                
                in_session = False
                if session_start_dt < session_end_dt: # Стандартный случай (сессия в пределах одного дня)
                    if session_start_dt <= fractal_ts < session_end_dt:
                        in_session = True
                else: # Случай пересечения полуночи (start_hour > end_hour)
                    # Фрактал должен быть либо после начала сегодня, либо до конца завтра (относительно начала сессии)
                     if fractal_ts >= session_start_dt or fractal_ts < session_end_dt: 
                        # Дополнительно убедимся, что дата фрактала = target_date (для части ДО полуночи)
                        # или target_date+1 (для части ПОСЛЕ полуночи) - это усложняет.
                        # Проще проверить, попадает ли время в диапазон, а дату проверить отдельно.
                        # Но сравнение datetime объектов уже учитывает дату.
                        # Если fractal_ts.date() == target_date и fractal_ts >= session_start_dt -> подходит
                        # Если fractal_ts.date() == target_date + 1 и fractal_ts < session_end_dt -> подходит
                        # Условие `fractal_ts >= session_start_dt or fractal_ts < session_end_dt` должно работать.
                        in_session = True # Упрощенная проверка для пересекающей полуночь сессии
                        
                # Дополнительная проверка даты (на всякий случай, если логика пересечения не идеальна)
                # Хотим фракталы только с reference_date (или reference_date + 1 для утренней части кросс-сессии)
                is_correct_date = False
                if session_start_dt.time() < session_end_dt.time(): # Не пересекает
                    if fractal_ts.date() == target_date: is_correct_date = True
                else: # Пересекает
                     if fractal_ts.date() == target_date or fractal_ts.date() == target_date + timedelta(days=1): is_correct_date = True

                if in_session and is_correct_date:
                    filtered.append(fractal)
                # --- КОНЕЦ ИЗМЕНЕННОГО УСЛОВИЯ ---

            except Exception as e:
                self.logger.warning(f"Не удалось обработать время фрактала {fractal_ts_orig} при фильтрации по сессии: {e}")
                continue
        
        self.logger.debug(f"Отфильтровано {len(filtered)} фракталов для сессии '{session_name}' на дату {target_date}.")
        return filtered

    def _calculate_target(self, data_1h: pd.DataFrame, current_price: float, context: str,
                         point_size: float, max_target_points: int) -> float | None:
        # ... (код без изменений, но убедитесь, что он корректен) ...
        if data_1h is None or data_1h.empty: return None
        
        if context == 'long':
            max_potential_price = current_price + (max_target_points * point_size)
            recent_highs = data_1h['high'].tail(48) 
            potential_targets = recent_highs[recent_highs > current_price]
            
            if not potential_targets.empty:
                best_historical_target = potential_targets.min() 
                target = min(best_historical_target, max_potential_price)
            else:
                target = max_potential_price
            return target

        elif context == 'short':
            min_potential_price = current_price - (max_target_points * point_size)
            recent_lows = data_1h['low'].tail(48)
            potential_targets = recent_lows[recent_lows < current_price]

            if not potential_targets.empty:
                best_historical_target = potential_targets.max()
                target = max(best_historical_target, min_potential_price)
            else:
                target = min_potential_price
            return target
        return None


    def _calculate_stop_loss(self, entry_price: float, fractal_price: float, atr_value: float | None,
                             context: str, point_size: float, symbol: str) -> float | None:
        # ... (код без изменений, но убедитесь, что он корректен) ...
        sl_price_fractal = fractal_price
        atr_multiplier = 1.5 
        if symbol == 'XAUUSD': atr_multiplier = 2.0
        elif symbol == 'GER40': atr_multiplier = 2.0

        atr_offset = 0
        if atr_value is not None and atr_value > 0:
            atr_offset = atr_value * atr_multiplier
        
        if context == 'long':
            sl_candidate_atr = entry_price - atr_offset if atr_offset > 0 else None
            if sl_candidate_atr is not None:
                final_sl = min(sl_price_fractal - (2 * point_size), sl_candidate_atr) 
            else:
                final_sl = sl_price_fractal - (2 * point_size) 
            return final_sl

        elif context == 'short':
            sl_candidate_atr = entry_price + atr_offset if atr_offset > 0 else None
            if sl_candidate_atr is not None:
                final_sl = max(sl_price_fractal + (2 * point_size), sl_candidate_atr)
            else:
                final_sl = sl_price_fractal + (2 * point_size)
            return final_sl
        return None

    def _check_skip_conditions(self, fractal: dict, data_1h: pd.DataFrame, data_3m: pd.DataFrame,
                               current_context: str, # Из analysis_results
                               analysis_results: dict, # Полный словарь результатов анализа
                               pdl: float | None, pdh: float | None,
                               open_positions: list,
                               point_size: float, max_target_points: int, atr_value_3m: float | None,
                               current_price_3m: float, symbol: str
                               ) -> list:
        skip_reasons = []
        derived_info = analysis_results.get('derived_info', {})
        asian_session_info = analysis_results.get('asian_session_analysis', {})
        # primary_context_info = analysis_results.get('primary_context_details', {}) # Если MarketAnalyzer будет возвращать больше деталей о первичном контексте

        # 1. Фрактал слишком старый
        fractal_ts_orig = fractal['timestamp']
        fractal_ts = pd.to_datetime(fractal_ts_orig).tz_convert('UTC') if isinstance(fractal_ts_orig, pd.Timestamp) else pd.to_datetime(fractal_ts_orig).tz_localize('UTC')
        current_time_utc = datetime.now(timezone.utc)
        if (current_time_utc.date() - fractal_ts.date()).days > 1:
            skip_reasons.append(f"Фрактал слишком старый ({fractal_ts.strftime('%Y-%m-%d')})")

        # 2. Используем флаг из MarketAnalyzer по поводу Азии и PDL/PDH
        if derived_info.get('skip_trade_due_to_asia_rejection', False):
            skip_reasons.append(f"Скип из-за отбоя Азии от PDL/PDH против тренда: {asian_session_info.get('message', '')}")

        # 3. Расстояние до цели (оценка)
        potential_target = self._calculate_target(data_1h, current_price_3m, current_context, point_size, max_target_points)
        if potential_target is not None:
            distance_to_target_points = abs(potential_target - current_price_3m) / point_size
            if distance_to_target_points > max_target_points:
                skip_reasons.append(f"Цель > {max_target_points} пт ({distance_to_target_points:.0f} пт)")
            
            min_dist_to_target = (atr_value_3m / point_size if atr_value_3m and atr_value_3m > 0 else 10.0) # Минимальная цель в пунктах
            if distance_to_target_points < min_dist_to_target :
                skip_reasons.append(f"Цель слишком близко ({distance_to_target_points:.0f} пт, мин: {min_dist_to_target:.0f})")
        else:
            skip_reasons.append("Не удалось рассчитать потенциальную цель для оценки.")

        # 4. Уже есть открытая позиция в том же направлении
        if any(pos.get('direction') == current_context for pos in open_positions if pos.get('status') == 'open'):
            skip_reasons.append("Уже есть открытая позиция в этом направлении")

        # 5. Проверка PDL/PDH - если цена уже за PDL/PDH и контекст против этого
        # (эта проверка может дублировать или уточнять 'skip_trade_due_to_asia_rejection')
        if current_context == 'long' and pdl is not None and current_price_3m < pdl:
            skip_reasons.append(f"Цена ({current_price_3m:.{self._get_decimals_for_symbol(symbol)}f}) ниже PDL ({pdl:.{self._get_decimals_for_symbol(symbol)}f}) при long контексте")
        if current_context == 'short' and pdh is not None and current_price_3m > pdh:
            skip_reasons.append(f"Цена ({current_price_3m:.{self._get_decimals_for_symbol(symbol)}f}) выше PDH ({pdh:.{self._get_decimals_for_symbol(symbol)}f}) при short контексте")

        self.logger.debug(f"Проверка скип-условий для фрактала {fractal.get('timestamp')}: {skip_reasons if skip_reasons else 'Нет причин для пропуска'}")
        return skip_reasons

    # ИЗМЕНЕНА СИГНАТУРА МЕТОДА find_signals
    def find_signals(self, data_1h: pd.DataFrame, data_3m: pd.DataFrame,
                     analysis_results: dict, # <--- ПРИНИМАЕМ ВЕСЬ СЛОВАРЬ
                     pdl: float | None, pdh: float | None, open_positions: list,
                     current_symbol: str) -> tuple[list, list, list]:
        valid_entry_signals = []
        skip_conditions_log = []
        
        # Извлекаем нужную информацию из analysis_results
        current_context = analysis_results.get('primary_context')
        # derived_info = analysis_results.get('derived_info', {}) # Уже используется в _check_skip_conditions
        # asian_session_info = analysis_results.get('asian_session_analysis', {})

        if data_1h is None or data_1h.empty or data_3m is None or data_3m.empty or \
           current_context is None or current_context == 'neutral':
            self.logger.warning(f"Недостаточно данных или неопределен контекст ('{current_context}') для поиска сигналов.")
            return valid_entry_signals, skip_conditions_log, []

        # Получаем настройки для текущего символа
        symbol_settings = self.instrument_settings.get(current_symbol)
        if not symbol_settings:
            self.logger.error(f"Настройки для символа {current_symbol} не найдены в self.instrument_settings. Пропуск поиска сигналов.")
            return valid_entry_signals, skip_conditions_log, []
        point_size = symbol_settings['point_size']
        max_target_points = symbol_settings['max_target_points']
            
        bullish_fractals_1h, bearish_fractals_1h = self._identify_fractals(data_1h)
        
        target_fractals_today = []
        current_utc_time = datetime.now(timezone.utc)
        for session_name in ['frankfurt', 'london', 'newyork']:
            sess_fractals = self._filter_fractals_by_session(bullish_fractals_1h + bearish_fractals_1h, session_name, current_utc_time)
            target_fractals_today.extend(sess_fractals)
        
        unique_target_fractals = []
        seen_fractal_ts = set()
        for f in target_fractals_today:
            fractal_ts_orig = f['timestamp']
            # Проверка типа перед добавлением в set
            fractal_ts_comparable = pd.Timestamp(fractal_ts_orig) if not isinstance(fractal_ts_orig, pd.Timestamp) else fractal_ts_orig

            if fractal_ts_comparable not in seen_fractal_ts:
                unique_target_fractals.append(f)
                seen_fractal_ts.add(fractal_ts_comparable)
        
        self.logger.info(f"Всего найдено {len(unique_target_fractals)} фракталов в целевых сессиях для {current_utc_time.date()}.")

        atr_series_3m = calculate_atr(data_3m['high'], data_3m['low'], data_3m['close'])
        atr_value_3m = atr_series_3m.iloc[-1] if atr_series_3m is not None and not atr_series_3m.empty else None
        current_price_3m = data_3m['close'].iloc[-1]

        for fractal in unique_target_fractals:
            if not ((current_context == 'long' and fractal['type'] == 'bullish') or \
                    (current_context == 'short' and fractal['type'] == 'bearish')):
                continue

            reasons_to_skip = self._check_skip_conditions(
                fractal, data_1h, data_3m, current_context, 
                analysis_results, # <--- ПЕРЕДАЕМ ПОЛНЫЙ СЛОВАРЬ
                pdl, pdh,
                open_positions, point_size, max_target_points, atr_value_3m,
                current_price_3m, current_symbol
            )
            
            if reasons_to_skip:
                skip_info = {'fractal': fractal, 'reasons': reasons_to_skip, 'timestamp': datetime.now(timezone.utc)}
                skip_conditions_log.append(skip_info)
                self.logger.info(f"Сигнал по фракталу {fractal['timestamp']} ({fractal['type']}) пропущен: {reasons_to_skip}")
                continue

            entry_price = current_price_3m
            take_profit = self._calculate_target(data_1h, entry_price, current_context, point_size, max_target_points)
            stop_loss = self._calculate_stop_loss(entry_price, fractal['price'], atr_value_3m, current_context, point_size, current_symbol)

            if take_profit is None or stop_loss is None:
                self.logger.warning(f"Не удалось рассчитать TP или SL для фрактала {fractal['timestamp']}.")
                continue

            if abs(entry_price - stop_loss) < point_size: # Проверка на слишком близкий стоп
                self.logger.warning(f"Стоп-лосс ({stop_loss}) слишком близок к цене входа ({entry_price}) для фрактала {fractal['timestamp']}. Пропуск.")
                continue

            potential_reward = abs(take_profit - entry_price)
            potential_risk = abs(entry_price - stop_loss)
            rr_ratio = potential_reward / potential_risk if potential_risk > 0 else float('inf')


            if rr_ratio >= self.min_rr:
                signal = {
                    'symbol': current_symbol,
                    'fractal_info': fractal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': current_context,
                    'timestamp': datetime.now(timezone.utc),
                    'rr_ratio': rr_ratio,
                    'reason_for_entry': f"Fractal {fractal['type']} @ {fractal['price']:.{self._get_decimals_for_symbol(current_symbol)}f} ({pd.to_datetime(fractal['timestamp']).strftime('%Y-%m-%d %H:%M')})",
                    # Можно добавить информацию из analysis_results для логирования в БД
                    'analysis_snapshot': { # Урезанная копия для лога
                        'primary_context': current_context,
                        'asia_trend': analysis_results.get('asian_session_analysis',{}).get('asia_trend'),
                        'asia_rejection_skip': analysis_results.get('derived_info',{}).get('skip_trade_due_to_asia_rejection')
                    }
                }
                valid_entry_signals.append(signal)
                self.logger.info(f"Сгенерирован сигнал: {signal}")
            else:
                self.logger.info(f"Сигнал по фракталу {fractal['timestamp']} отфильтрован по R/R: {rr_ratio:.2f} < {self.min_rr}")
                skip_info = {'fractal': fractal, 'reasons': [f"Низкий R/R: {rr_ratio:.2f}"], 'timestamp': datetime.now(timezone.utc)}
                skip_conditions_log.append(skip_info)

        return valid_entry_signals, skip_conditions_log, unique_target_fractals

    def _get_decimals_for_symbol(self, symbol: str) -> int:
        """Возвращает количество знаков после запятой для форматирования цены."""
        instr_settings = self.instrument_settings.get(symbol, {})
        point_size = instr_settings.get('point_size', 0.0001)
        if point_size == 0.0001: return 5
        elif point_size == 0.01: return 2
        elif point_size == 0.1: return 2 # XAUUSD
        elif point_size == 1.0: return 1
        return 5