# ui/console_interface.py

import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    # Заглушки, если colorama не установлена
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = Style = DummyColor() # type: ignore

from config import settings # Для доступа к SESSIONS и другим настройкам
# Для аннотации типов, если check_entry_conditions_for_ui будет принимать экземпляр бота
# from core.bot import TradingBot1H3M # Раскомментируйте, если будете использовать


def clear_screen():
    """Очищает экран консоли."""
    os.system('cls' if os.name == 'nt' else 'clear')

def _get_price_format_string(point_size: float) -> str:
    """Определяет строку форматирования для цены на основе point_size."""
    if point_size == 0.0001: return "{:.5f}"  # Forex (EURUSD, GBPUSD)
    elif point_size == 0.01: return "{:.2f}" # JPY пары, некоторые акции
    elif point_size == 0.1: return "{:.2f}"   # XAUUSD (часто 2 знака после запятой, но цена может быть и xxx.x)
    elif point_size == 1: return "{:.1f}"     # Индексы типа GER40
    return "{:.2f}" # По умолчанию

def _format_price(price: Optional[float], point_size: float) -> str:
    """Форматирует цену для вывода."""
    if price is None:
        return "N/A"
    fmt = _get_price_format_string(point_size)
    return fmt.format(price)

def print_header(symbol: str):
    """Печатает заголовок приложения."""
    print(Style.BRIGHT + Fore.CYAN + "=" * 40)
    print(Style.BRIGHT + Fore.CYAN + f" Торговый Бот 1H3M [{symbol}] ".center(40))
    print(Style.BRIGHT + Fore.CYAN + "=" * 40 + Style.RESET_ALL)
    print(f"{Fore.BLUE}Текущее время UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}\n")


def print_market_context(context: Optional[str]):
    """Выводит текущий контекст рынка."""
    context_color = Fore.YELLOW
    if context == 'long':
        context_color = Fore.GREEN
    elif context == 'short':
        context_color = Fore.RED
    elif context == 'neutral':
        context_color = Fore.MAGENTA
    
    print(Style.BRIGHT + "--- Контекст Рынка ---")
    print(f"Текущий контекст: {context_color}{context.upper() if context else 'НЕ ОПРЕДЕЛЕН'}{Style.RESET_ALL}\n")


def print_condition(name: str, is_met: bool, description: str = ""):
    """Печатает состояние одного условия."""
    status_text = f"{Fore.GREEN}ДА" if is_met else f"{Fore.RED}НЕТ"
    description_text = f" ({description})" if description else ""
    print(f"{name}: {status_text}{Style.RESET_ALL}{description_text}")

def print_conditions_wrapper(conditions_data: List[Dict[str, Any]]):
    """Обертка для печати списка условий."""
    print(Style.BRIGHT + "--- Условия для Входа ---")
    if not conditions_data:
        print(f"{Fore.YELLOW}Данные по условиям не предоставлены.{Style.RESET_ALL}")
    for cond in conditions_data:
        print_condition(cond['name'], cond['is_met'], cond.get('description', ''))
    print("") # Пустая строка для отступа


def print_daily_limit(daily_limit: Optional[float], context: Optional[str], point_size: float):
    """Выводит информацию о дневном лимите."""
    print(Style.BRIGHT + "--- Дневной Лимит (DL) ---")
    if daily_limit is not None:
        dl_type = "Low (поддержка)" if context == 'long' else "High (сопротивление)" if context == 'short' else "N/A"
        print(f"Дневной лимит ({dl_type}): {_format_price(daily_limit, point_size)}")
    else:
        print(f"{Fore.YELLOW}Дневной лимит не установлен.{Style.RESET_ALL}")
    print("")


def print_fractal_levels(fractal_levels: List[Dict[str, Any]], current_context: Optional[str], point_size: float):
    """Выводит информацию о найденных фрактальных уровнях."""
    print(Style.BRIGHT + "--- Актуальные Фрактальные Уровни (1H) ---")
    if not fractal_levels:
        print(f"{Fore.YELLOW}Нет актуальных фрактальных уровней.{Style.RESET_ALL}")
    else:
        # Фильтруем фракталы, соответствующие текущему контексту
        relevant_fractals = [
            f for f in fractal_levels
            if (current_context == 'long' and f['type'] == 'bullish') or \
               (current_context == 'short' and f['type'] == 'bearish')
        ]
        if not relevant_fractals:
            print(f"{Fore.YELLOW}Нет фракталов, соответствующих текущему контексту ('{current_context}').{Style.RESET_ALL}")

        for i, f in enumerate(relevant_fractals):
            fractal_time = pd.to_datetime(f['timestamp']).strftime('%Y-%m-%d %H:%M')
            color = Fore.GREEN if f['type'] == 'bullish' else Fore.RED
            print(f"{i+1}. Тип: {color}{f['type'].capitalize()}{Style.RESET_ALL}, "
                  f"Цена: {_format_price(f['price'], point_size)}, "
                  f"Время: {fractal_time}")
    print("")


def print_session_status(sessions_config: Dict = settings.SESSIONS):
    """Выводит статус текущих торговых сессий."""
    print(Style.BRIGHT + "--- Торговые Сессии (UTC) ---")
    now_utc = datetime.now(timezone.utc)
    active_sessions = []
    for name, times in sessions_config.items():
        start_hour, end_hour = times['start'], times['end']
        # Проверка, активна ли сессия (end_hour не включается)
        is_active = False
        if start_hour <= end_hour: # Сессия не пересекает полночь
            if start_hour <= now_utc.hour < end_hour:
                is_active = True
        else: # Сессия пересекает полночь (например, Азия для некоторых таймзон)
            if now_utc.hour >= start_hour or now_utc.hour < end_hour:
                is_active = True
        
        status_text = f"{Fore.GREEN}АКТИВНА" if is_active else f"{Fore.RED}НЕАКТИВНА"
        print(f"{name.capitalize()}: ({start_hour:02d}:00 - {end_hour:02d}:00) - {status_text}{Style.RESET_ALL}")
        if is_active:
            active_sessions.append(name.capitalize())
    if active_sessions:
        print(f"Сейчас активны: {', '.join(active_sessions)}")
    else:
        print("Сейчас нет активных ключевых сессий.")
    print("")


def print_signal(signal: Dict[str, Any], symbol: str, point_size: float):
    """Выводит информацию о торговом сигнале."""
    print(Style.BRIGHT + Fore.MAGENTA + "--- Найден ТОРГОВЫЙ СИГНАЛ! ---" + Style.RESET_ALL)
    direction_color = Fore.GREEN if signal['direction'] == 'long' else Fore.RED
    print(f"Символ: {symbol}")
    print(f"Направление: {direction_color}{signal['direction'].upper()}{Style.RESET_ALL}")
    print(f"Цена входа: {_format_price(signal['entry_price'], point_size)}")
    print(f"Стоп-лосс: {_format_price(signal['stop_loss'], point_size)}")
    print(f"Тейк-профит: {_format_price(signal['take_profit'], point_size)}")
    print(f"R/R Ratio: {signal.get('rr_ratio', 'N/A'):.2f}")
    if signal.get('fractal_info'):
        f_info = signal['fractal_info']
        f_time = pd.to_datetime(f_info['timestamp']).strftime('%Y-%m-%d %H:%M')
        print(f"Основан на фрактале: {f_info['type']} @ {_format_price(f_info['price'], point_size)} ({f_time})")
    print(f"Причина для входа: {signal.get('reason_for_entry', 'N/A')}")
    print("-" * 30 + "\n")


def print_skip_conditions(skip_conditions: List[Dict[str, Any]]):
    """Выводит информацию о пропущенных сигналах."""
    print(Style.BRIGHT + "--- Пропущенные Сигналы/Условия ---")
    if not skip_conditions:
        print(f"{Fore.GREEN}Нет недавно пропущенных сигналов.{Style.RESET_ALL}")
    else:
        # Показываем последние N пропусков
        for i, skip in enumerate(reversed(skip_conditions[-5:])): # Последние 5
            fractal_info_str = "N/A"
            if skip.get('fractal'):
                f = skip['fractal']
                f_time = pd.to_datetime(f['timestamp']).strftime('%H:%M')
                fractal_info_str = f"{f['type']} @ {f['price']:.5f} ({f_time})" # Используем 5 знаков по умолчанию, если нет point_size
            
            reasons_str = ', '.join(skip.get('reasons', ['Не указана']))
            skip_time = pd.to_datetime(skip.get('timestamp', datetime.now())).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i+1}. {Fore.YELLOW}Пропуск ({skip_time}){Style.RESET_ALL}: Фрактал: {fractal_info_str}. Причины: {reasons_str}")
    print("")


def print_open_positions(open_positions: List[Dict[str, Any]], current_price: Optional[float], point_size: float):
    """Выводит информацию об открытых позициях."""
    print(Style.BRIGHT + "--- Открытые Позиции (Симуляция) ---")
    if not open_positions:
        print(f"{Fore.GREEN}Нет открытых позиций.{Style.RESET_ALL}")
    else:
        for i, pos in enumerate(open_positions):
            if pos.get('status') != 'open': continue # Показываем только реально открытые

            direction_color = Fore.GREEN if pos['direction'] == 'long' else Fore.RED
            entry_time_str = pd.to_datetime(pos['entry_time']).strftime('%Y-%m-%d %H:%M:%S')
            
            pnl_points_str = "N/A"
            pnl_currency_str = "N/A"
            
            if current_price is not None:
                pnl_mult = 1 if pos['direction'] == 'long' else -1
                pnl_points = (current_price - pos['entry_price']) * pnl_mult / point_size
                # Упрощенный PnL в "валюте" (предполагая, что position_size это нечто вроде лотов)
                # и 1 пункт изменения цены на 1 лот дает 1 "у.е." профита/убытка
                # Это очень грубая симуляция!
                pnl_currency = pnl_points * point_size * pos['position_size'] * (1/point_size) # (current_price - pos['entry_price']) * pos['position_size'] * pnl_mult
                
                pnl_points_str = f"{pnl_points:+.1f} пт"
                pnl_currency_str = f"{pnl_currency:+.2f} у.е."
                pnl_color = Fore.GREEN if pnl_points >= 0 else Fore.RED
                pnl_points_str = f"{pnl_color}{pnl_points_str}{Style.RESET_ALL}"
                pnl_currency_str = f"{pnl_color}{pnl_currency_str}{Style.RESET_ALL}"


            print(f"{i+1}. ID: {pos.get('id_internal', pos.get('id_db', 'N/A'))} "
                  f"Символ: {pos['symbol']} "
                  f"Направление: {direction_color}{pos['direction'].upper()}{Style.RESET_ALL}")
            print(f"   Вход: {_format_price(pos['entry_price'], point_size)} ({entry_time_str}) "
                  f"Размер: {pos['position_size']:.2f}")
            print(f"   SL: {_format_price(pos['stop_loss'], point_size)} "
                  f"TP: {_format_price(pos['take_profit'], point_size)}")
            print(f"   Текущий PnL: {pnl_points_str} / {pnl_currency_str}")
            print("-" * 20)
    print("")


def print_menu():
    """Печатает меню действий."""
    print(Style.BRIGHT + "--- Меню ---" + Style.RESET_ALL)
    print("1. Обновить данные и статус")
    print("2. Выполнить сделку по сигналу (если есть)")
    print("3. Показать график стратегии")
    print("4. Сменить торговый символ")
    print("0. Выход")
    print("-" * 20)

# Убедитесь, что pandas импортирован, если не был ранее в этом файле
try:
    import pandas as pd
except ImportError:
    # Обработка отсутствия pandas, если это возможно для каких-то функций
    # Но для print_fractal_levels и др. он нужен.
    pass


def check_entry_conditions_for_ui(bot_instance: Any) -> List[Dict[str, Any]]:
    """
    Собирает информацию о состоянии различных условий для входа из экземпляра бота
    и форматирует её для функции print_conditions_wrapper.

    Args:
        bot_instance: Экземпляр вашего класса TradingBot1H3M.
                      (Используем Any для избежания циклического импорта,
                       но лучше использовать 'core.bot.TradingBot1H3M' если настроено)
    Returns:
        Список словарей, каждый из которых описывает одно условие.
    """
    conditions = []
    
    # 1. Контекст рынка определен
    context_defined = bot_instance.current_context not in [None, 'neutral']
    conditions.append({
        'name': "Контекст рынка определен (Long/Short)",
        'is_met': context_defined,
        'description': f"Текущий: {bot_instance.current_context.upper() if bot_instance.current_context else 'N/A'}"
    })

    # 2. Горизонтальный тренд (пример)
    # Предположим, что бот хранит trend_info
    ht_info = getattr(bot_instance, 'horizontal_trend_info', {})
    is_not_strong_horizontal = not (ht_info.get('is_horizontal', False) and ht_info.get('confidence', 0.0) > 0.7)
    conditions.append({
        'name': "Отсутствие сильного горизонтального тренда",
        'is_met': is_not_strong_horizontal,
        'description': f"Горизонтальный: {ht_info.get('is_horizontal', 'N/A')}, Уверенность: {ht_info.get('confidence', 0.0):.2f}"
    })

    # 3. Реакция на FVG (пример)
    # Предположим, бот хранит analysis_results
    analysis_res = getattr(bot_instance, 'analysis_results', {})
    fvg_reacted = analysis_res.get('fvg_reacted', False)
    conditions.append({
        'name': "Реакция на FVG (если была)", # Или "Отсутствие реакции на FVG" в зависимости от стратегии
        'is_met': fvg_reacted, # Это условие может быть "НЕ должно быть реакции" или "ДОЛЖНА быть"
        'description': f"FVG отреагировал: {fvg_reacted}"
    })
    
    # 4. Снятие SSL/BSL (пример)
    ssl_bsl_removed = analysis_res.get('any_ssl_bsl_removed', False)
    conditions.append({
        'name': "Снятие SSL/BSL (если было)",
        'is_met': ssl_bsl_removed,
        'description': f"SSL/BSL снят: {ssl_bsl_removed}"
    })

    # 5. Наличие актуальных фракталов по контексту
    relevant_fractals_exist = False
    if bot_instance.fractal_levels and bot_instance.current_context:
        relevant_fractals_exist = any(
            (bot_instance.current_context == 'long' and f['type'] == 'bullish') or \
            (bot_instance.current_context == 'short' and f['type'] == 'bearish')
            for f in bot_instance.fractal_levels
        )
    conditions.append({
        'name': "Есть актуальные фракталы по контексту",
        'is_met': relevant_fractals_exist,
        'description': f"Найдено: {sum(1 for f in bot_instance.fractal_levels if (bot_instance.current_context == 'long' and f['type'] == 'bullish') or (bot_instance.current_context == 'short' and f['type'] == 'bearish')) if bot_instance.fractal_levels else 0}"
    })
    
    # 6. Не слишком близко к DL (пример)
    # Эта логика требует более точного определения "слишком близко"
    # current_price = bot_instance.get_current_price()
    # dl_ok = True
    # if bot_instance.daily_limit and current_price:
    #     atr = bot_instance.analysis_results.get('atr_value_3m') # Предположим, что ATR есть
    #     if atr:
    #         if abs(current_price - bot_instance.daily_limit) < atr * 0.5:
    #             dl_ok = False
    # conditions.append({
    #     'name': "Цена не слишком близко к DL",
    #     'is_met': dl_ok
    # })
    
    return conditions