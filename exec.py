#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import colorama
from colorama import Fore, Back, Style
from datetime import datetime
from main import TradingBot1H3M  # Импортируем ваш основной класс бота

# Инициализация colorama для работы с цветами в терминале
colorama.init(autoreset=True)

def clear_screen():
    """Очистка экрана терминала"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Печать заголовка программы"""
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}{'ТОРГОВЫЙ БОТ 1H3M':^80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{datetime.now().strftime('%d.%m.%Y %H:%M:%S'):^80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)

def print_market_context(context):
    """Печать контекста рынка с цветовым выделением"""
    if context is None:
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Контекст рынка: {Fore.YELLOW}Неопределен")
        return
    
    context_color = Fore.GREEN if context == 'long' else Fore.RED
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Контекст рынка: {context_color}{context.upper()}")

def print_condition(condition_name, is_met, description=""):
    """Печать условия с индикатором (галочка/крестик)"""
    indicator = f"{Fore.GREEN}✓" if is_met else f"{Fore.RED}✗"
    print(f"{indicator} {Fore.WHITE}{condition_name}{Fore.YELLOW} {description}")

def print_conditions_wrapper(conditions):
    """Обертка для безопасного вывода условий"""
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Условия для входа:")
    for condition in conditions:
        # Проверяем длину кортежа и применяем соответствующий формат
        if len(condition) == 3:
            name, is_met, description = condition
            print_condition(name, is_met, description)
        elif len(condition) == 2:
            name, is_met = condition
            print_condition(name, is_met)
        else:
            print(f"{Fore.RED}Ошибка формата условия: {condition}")

def print_signal(signal):
    """Печать информации о сигнале"""
    # Убедитесь, что colorama и ее компоненты импортированы в начале файла exec.py
    # Например:
    # import colorama
    # from colorama import Fore, Style
    # colorama.init(autoreset=True) # Инициализация colorama

    # Проверяем наличие необходимых ключей в словаре signal перед их использованием
    expected_keys = ['direction', 'entry_price', 'take_profit', 'stop_loss']
    missing_keys = [key for key in expected_keys if key not in signal]

    if missing_keys:
        # Выводим сообщение об ошибке, если каких-то ключей не хватает
        # Предполагается, что Style, Fore, Style.RESET_ALL из colorama импортированы
        print(f"\n{Style.BRIGHT}{Fore.RED}Ошибка: Словарь сигнала не содержит следующих ключей: {', '.join(missing_keys)}.{Style.RESET_ALL}")
        print(f"Полученный сигнал: {signal}")
        return

    direction_color = Fore.GREEN if signal['direction'] == 'long' else Fore.RED
    direction = signal['direction'].upper()
    entry_price = signal['entry_price']
    
    # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ 1: Используем 'take_profit' вместо 'target'
    target = signal['take_profit'] 
    
    point_size_local = 0.0001 # Значение по умолчанию для EURUSD-подобных пар
    symbol_local = "N/A"      # Значение по умолчанию

    # Предполагается, что 'bot' - это глобальная переменная в exec.py,
    # инициализированная экземпляром TradingBot1H3M.
    # Эта переменная должна быть доступна в области видимости этой функции.
    if 'bot' in globals() and hasattr(bot, 'symbol') and hasattr(bot, 'point_size'):
        point_size_local = bot.point_size
        symbol_local = bot.symbol
    elif 'bot' in globals() and hasattr(bot, 'symbol'): 
        # Если point_size не найден напрямую, пытаемся определить его по символу
        symbol_local = bot.symbol
        if 'USD' in symbol_local and 'XAU' not in symbol_local: # Для валютных пар типа EURUSD, GBPUSD
             point_size_local = 0.0001
        elif 'GER40' == symbol_local:
             point_size_local = 1 
        elif 'XAUUSD' == symbol_local:
             point_size_local = 0.1
        # Добавьте другие условия для разных символов, если необходимо
        else:
            # Предполагается, что Fore, Style, Style.RESET_ALL из colorama импортированы
            print(f"{Fore.YELLOW}Предупреждение: Не удалось автоматически определить point_size для символа {symbol_local}. Используется значение по умолчанию {point_size_local}.{Style.RESET_ALL}")
    else:
        # Предполагается, что Fore, Style, Style.RESET_ALL из colorama импортированы
        print(f"{Fore.YELLOW}Предупреждение: Глобальная переменная 'bot' или ее атрибуты 'symbol'/'point_size' не найдены. Расчет пунктов может быть неточным.{Style.RESET_ALL}")

    # Убедимся, что target и entry_price являются числами перед использованием в расчетах
    if not isinstance(target, (int, float)) or not isinstance(entry_price, (int, float)):
        # Предполагается, что Fore, Style.RESET_ALL из colorama импортированы
        print(f"{Fore.RED}Ошибка: target ({target}) или entry_price ({entry_price}) не являются числами.{Style.RESET_ALL}")
        return
        
    target_pips = abs(target - entry_price) / point_size_local
    
    # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ 2: Используем 'stop_loss' из словаря signal
    stop_loss = signal['stop_loss']

    # Убедимся, что stop_loss является числом
    if not isinstance(stop_loss, (int, float)):
        # Предполагается, что Fore, Style.RESET_ALL из colorama импортированы
        print(f"{Fore.RED}Ошибка: stop_loss ({stop_loss}) не является числом.{Style.RESET_ALL}")
        return
        
    risk_pips = abs(entry_price - stop_loss) / point_size_local
    
    # Избегаем деления на ноль для risk_reward
    risk_reward = target_pips / risk_pips if risk_pips > 0 else float('inf')
    
    # Предполагается, что Style, Fore, Style.RESET_ALL, Fore.MAGENTA из colorama импортированы
    print(f"\n{Style.BRIGHT}Сигнал для входа ({symbol_local}):")
    print(f"  Направление: {direction_color}{direction}{Style.RESET_ALL}")
    print(f"  Цена входа: {Fore.CYAN}{entry_price:.5f}{Style.RESET_ALL}")
    print(f"  Целевой уровень (Take Profit): {Fore.CYAN}{target:.5f}{Style.RESET_ALL} ({Fore.MAGENTA}{int(target_pips)} пунктов{Style.RESET_ALL})")
    print(f"  Стоп-лосс: {Fore.CYAN}{stop_loss:.5f}{Style.RESET_ALL} ({Fore.MAGENTA}{int(risk_pips)} пунктов{Style.RESET_ALL})")
    print(f"  Соотношение риск/прибыль: {Fore.YELLOW}{risk_reward:.2f}{Style.RESET_ALL}")


def print_open_positions(positions):
    """Печать информации об открытых позициях"""
    if not positions:
        print(f"\n{Fore.WHITE}Открытые позиции: {Fore.YELLOW}Нет")
        return
        
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Открытые позиции:")
    for pos in positions:
        direction_color = Fore.GREEN if pos['direction'] == 'long' else Fore.RED
        profit_pips = (bot.data_3m['close'].iloc[-1] - pos['entry_price']) / bot.point_size if pos['direction'] == 'long' else \
                      (pos['entry_price'] - bot.data_3m['close'].iloc[-1]) / bot.point_size
                      
        profit_color = Fore.GREEN if profit_pips > 0 else Fore.RED
        
        print(f"  {direction_color}{pos['direction'].upper()} {Fore.WHITE}вход: {pos['entry_price']:.5f}, "
              f"цель: {pos['target']:.5f}, стоп-лосс: {pos['stop_loss']:.5f}, "
              f"тек. P/L: {profit_color}{int(profit_pips)} пунктов")

def print_fractal_levels(levels):
    """Печать информации о фрактальных уровнях"""
    if not levels:
        print(f"\n{Fore.WHITE}Фрактальные уровни: {Fore.YELLOW}Не найдены")
        return
        
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Фрактальные уровни:")
    for level in levels:
        level_type_color = Fore.GREEN if level['type'] == 'bullish' else Fore.RED
        level_type = "Бычий" if level['type'] == 'bullish' else "Медвежий"
        print(f"  {level_type_color}{level_type} {Fore.WHITE}уровень: {level['price']:.5f} "
              f"({level['timestamp'].strftime('%d.%m %H:%M')})")

def print_skip_conditions(conditions):
    """Печать информации о пропущенных сигналах"""
    if not conditions:
        return
        
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Пропущенные сигналы:")
    for cond in conditions:
        reasons = ", ".join(cond['reasons'])
        print(f"  {Fore.WHITE}Фрактал {cond['fractal']['price']:.5f}: {Fore.YELLOW}{reasons}")

def print_daily_limit(limit, context):
    """Печать информации о дневном лимите"""
    if limit is None:
        print(f"\n{Fore.WHITE}Дневной лимит: {Fore.YELLOW}Не установлен")
        return
    
    try:    
        limit_type = "Поддержка" if context == 'long' else "Сопротивление"
        print(f"\n{Fore.WHITE}Дневной лимит ({limit_type}): {Fore.CYAN}{limit:.5f}")
    except (ValueError, TypeError):
        # Если не удалось отформатировать limit как число, выводим как есть
        print(f"\n{Fore.WHITE}Дневной лимит ({limit_type}): {Fore.CYAN}{limit}")

def print_session_status():
    """Печать информации о текущей торговой сессии"""
    now = datetime.now()
    hour = now.hour
    
    sessions = {
        'Азиатская': (1, 9),
        'Лондонская': (8, 16),
        'Нью-Йоркская': (13, 21),
        'Франкфуртская': (7, 15)
    }
    
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Статус торговых сессий:")
    
    for session_name, session_times in sessions.items():
        try:
            # Преобразуем значения времени в целые числа
            start = int(session_times[0]) if isinstance(session_times, tuple) and len(session_times) > 0 else 0
            end = int(session_times[1]) if isinstance(session_times, tuple) and len(session_times) > 1 else 24
            
            is_active = start <= hour < end
            status = f"{Fore.GREEN}Активна" if is_active else f"{Fore.YELLOW}Неактивна"
            print(f"  {session_name}: {status} ({start}:00-{end}:00 UTC)")
        except (ValueError, TypeError, IndexError):
            print(f"  {session_name}: {Fore.RED}Ошибка в формате сессии")

def print_menu():
    """Печать меню действий"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
    print(f"{Fore.WHITE}{Style.BRIGHT}Выберите действие:")
    print(f"{Fore.CYAN}[1] {Fore.WHITE}Обновить данные и проверить сигналы")
    print(f"{Fore.CYAN}[2] {Fore.WHITE}Выполнить сделку по сигналу (если есть)")
    print(f"{Fore.CYAN}[3] {Fore.WHITE}Показать график стратегии")
    print(f"{Fore.CYAN}[4] {Fore.WHITE}Установить другой символ")
    print(f"{Fore.CYAN}[0] {Fore.WHITE}Выход")
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)

def check_entry_conditions(bot):
    """Проверка всех условий для входа в позицию и подготовка для отображения"""
    conditions = []
    
    # 1. Проверка контекста рынка
    context = bot.current_context
    context_check = context is not None
    conditions.append(("Контекст рынка определен", context_check, f"{context.upper() if context else ''}"))
    
    # 2. Наличие подходящих фракталов
    fractals_found = len(bot.fractal_levels) > 0
    conditions.append(("Найдены подходящие фракталы", fractals_found, f"Найдено: {len(bot.fractal_levels)}"))
    
    # 3. Проверка текущего времени на попадание в сессию
    now = datetime.now()
    hour = now.hour
    in_trading_session = False
    current_session = "Нет активной сессии"
    
    # Проверяем, есть ли у бота атрибут sessions, если нет, создаем стандартный словарь сессий
    sessions = getattr(bot, 'sessions', {
        'Азиатская': (1, 9),
        'Лондонская': (8, 16),
        'Нью-Йоркская': (13, 21),
        'Франкфуртская': (7, 15)
    })
    
    for session_name, session_times in sessions.items():
        try:
            # Безопасное извлечение значений из кортежа или списка
            if isinstance(session_times, (tuple, list)) and len(session_times) >= 2:
                start = int(session_times[0])
                end = int(session_times[1])
            else:
                # Если не кортеж/список или недостаточно элементов, пропускаем
                continue
                
            if start <= hour < end:
                in_trading_session = True
                current_session = session_name
                break
        except (ValueError, TypeError) as e:
            print(f"{Fore.RED}Ошибка в формате сессии {session_name}: {e}")
            in_trading_session = True
            current_session = session_name
            break
    
    conditions.append(("Активная торговая сессия", in_trading_session, current_session))
    
    # 4. Проверка дневного лимита
    dl_set = bot.daily_limit is not None
    if dl_set:
        dl_value = f"{bot.daily_limit:.5f}"
    else:
        dl_value = ""
    conditions.append(("Установлен дневной лимит", dl_set, dl_value))
    
    # 5. Нет открытых позиций в том же направлении
    no_open_positions = not any(pos['direction'] == bot.current_context for pos in bot.open_positions)
    conditions.append(("Нет открытых позиций в том же направлении", no_open_positions, ""))  # Добавлен пустой description
    
    # 6. Целевое расстояние в пределах допустимого
    target_distance_ok = True
    target_distance_value = "Не рассчитано"
    
    if fractals_found:
        for fractal in bot.fractal_levels:
            # Проверяем, соответствует ли фрактал текущему контексту рынка
            if (bot.current_context == 'long' and fractal['type'] == 'bullish') or \
               (bot.current_context == 'short' and fractal['type'] == 'bearish'):
                try:
                    # Получаем расстояние и преобразуем его в число
                    distance = bot.calculate_target_distance(fractal)
                    # Если calculate_target_distance вернул строку, преобразуем в float
                    if isinstance(distance, str):
                        distance = float(distance.replace(',', '.'))
                    else:
                        distance = float(distance)
                    
                    # Получаем максимальное расстояние и преобразуем его в число
                    max_points = getattr(bot, 'max_target_points', float('inf'))
                    # Если max_target_points - строка, преобразуем в float
                    if isinstance(max_points, str):
                        max_points = float(max_points.replace(',', '.'))
                    else:
                        max_points = float(max_points)
                    
                    # Сравниваем расстояние с максимально допустимым
                    target_distance_ok = distance <= max_points
                    target_distance_value = f"{int(distance)} из {int(max_points)} пунктов"
                except (ValueError, TypeError, AttributeError) as e:
                    # Более подробное сообщение об ошибке для отладки
                    error_details = f"Тип distance: {type(distance) if 'distance' in locals() else 'не определен'}, "
                    error_details += f"Значение distance: {distance if 'distance' in locals() else 'не определено'}, "
                    error_details += f"Тип max_points: {type(max_points) if 'max_points' in locals() else 'не определен'}, "
                    error_details += f"Значение max_points: {max_points if 'max_points' in locals() else 'не определено'}"
                    print(f"{Fore.RED}Ошибка при вычислении расстояния до цели: {e}\n{error_details}")
                    
                    target_distance_ok = False
                    target_distance_value = "Ошибка расчета"
                break
    
    conditions.append(("Цель в пределах лимита", target_distance_ok, target_distance_value))
    
    return conditions

def main():
    """Основная функция программы"""
    global bot
    
    # Создаем бота для EURUSD (можете изменить на GER40)
    bot = TradingBot1H3M(symbol='EURUSD')
    
    while True:
        clear_screen()
        print_header()
        
        try:
            # Загрузка данных, только если еще не загружены
            if bot.data_1h is None or bot.data_3m is None:
                print(f"\n{Fore.YELLOW}Загрузка исторических данных...")
                bot.fetch_data()
                bot.analyze_market_context(bot.data_3m)
                bot.update_daily_limit()
            
            # Выводим информацию о контексте рынка
            print_market_context(bot.current_context)
            
            # Проверяем условия для входа
            conditions = check_entry_conditions(bot)
            
            print(f"\n{Fore.WHITE}{Style.BRIGHT}Условия для входа:")
            for name, is_met, description in conditions:
                print_condition(name, is_met, description)
            
            # Выводим информацию о дневном лимите
            print_daily_limit(bot.daily_limit, bot.current_context)
            
            # Выводим информацию о фрактальных уровнях
            print_fractal_levels(bot.fractal_levels)
            
            # Выводим информацию о текущей сессии
            print_session_status()
            
            # Проверяем наличие сигналов
            entry_signals = bot.find_entry_signals()
            
            # Выводим информацию о сигналах
            if entry_signals:
                for signal in entry_signals:
                    print_signal(signal)
            else:
                print(f"\n{Fore.YELLOW}Нет активных сигналов для входа")
            
            # Выводим информацию о пропущенных сигналах
            print_skip_conditions(bot.skip_conditions)
            
            # Выводим информацию об открытых позициях
            print_open_positions(bot.open_positions)
            
            # Управление открытыми позициями
            if bot.open_positions:
                bot.manage_open_positions()
            
            # Выводим меню
            print_menu()
            
            # Получаем ввод пользователя
            choice = input(f"{Fore.WHITE}Введите номер действия: ")
            
            if choice == '0':
                print(f"{Fore.GREEN}Выход из программы...")
                break
                
            elif choice == '1':
                print(f"{Fore.YELLOW}Обновление данных...")
                bot.fetch_data()
                bot.analyze_market_context(bot.data_3m)
                bot.update_daily_limit()
                bot.find_entry_signals()
                
            elif choice == '2':
                if entry_signals:
                    print(f"{Fore.YELLOW}Выполнение сделки по сигналу...")
                    for signal in entry_signals:
                        bot.execute_trade(signal)
                    time.sleep(2)
                else:
                    print(f"{Fore.RED}Нет активных сигналов для входа")
                    time.sleep(2)
                    
            elif choice == '3':
                print(f"{Fore.YELLOW}Создание графика стратегии...")
                bot.visualize_strategy(save_path="strategy_chart.png")
                print(f"{Fore.GREEN}График сохранен в файл strategy_chart.png")
                time.sleep(2)
                
            elif choice == '4':
                new_symbol = input(f"{Fore.WHITE}Введите символ (EURUSD или GER40): ").upper()
                if new_symbol in ['EURUSD', 'GER40']:
                    bot = TradingBot1H3M(symbol=new_symbol)
                    print(f"{Fore.GREEN}Символ изменен на {new_symbol}")
                    bot.fetch_data()
                    bot.analyze_market_context(bot.data_3m)
                    bot.update_daily_limit()
                else:
                    print(f"{Fore.RED}Неподдерживаемый символ. Используйте EURUSD или GER40")
                time.sleep(2)
                
            else:
                print(f"{Fore.RED}Неверный ввод")
                time.sleep(1)
                
        except Exception as e:
            print(f"{Fore.RED}Произошла ошибка: {e}")
            import traceback
            traceback.print_exc()  # Печать полного стека вызовов для отладки
            time.sleep(5)

if __name__ == "__main__":
    main()