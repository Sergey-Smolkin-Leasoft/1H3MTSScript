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
    context_color = Fore.GREEN if context == 'long' else Fore.RED
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Контекст рынка: {context_color}{context.upper()}")

def print_condition(condition_name, is_met, description=""):
    """Печать условия с индикатором (галочка/крестик)"""
    indicator = f"{Fore.GREEN}✓" if is_met else f"{Fore.RED}✗"
    print(f"{indicator} {Fore.WHITE}{condition_name}{Fore.YELLOW} {description}")

def print_signal(signal):
    """Печать информации о сигнале"""
    direction_color = Fore.GREEN if signal['direction'] == 'long' else Fore.RED
    direction = signal['direction'].upper()
    entry_price = signal['entry_price']
    target = signal['target']
    target_pips = abs(target - entry_price) / (0.0001 if 'USD' in bot.symbol else 1)
    
    # Рассчитываем риск
    stop_loss = signal['fractal']['price']
    risk_pips = abs(entry_price - stop_loss) / (0.0001 if 'USD' in bot.symbol else 1)
    risk_reward = target_pips / risk_pips if risk_pips > 0 else float('inf')
    
    print(f"\n{Style.BRIGHT}Сигнал для входа:")
    print(f"  Направление: {direction_color}{direction}")
    print(f"  Цена входа: {Fore.CYAN}{entry_price:.5f}")
    print(f"  Целевой уровень: {Fore.CYAN}{target:.5f} ({int(target_pips)} пунктов)")
    print(f"  Стоп-лосс: {Fore.CYAN}{stop_loss:.5f} ({int(risk_pips)} пунктов)")
    print(f"  Соотношение риск/прибыль: {Fore.YELLOW}{risk_reward:.2f}")

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
        
    limit_type = "Поддержка" if context == 'long' else "Сопротивление"
    print(f"\n{Fore.WHITE}Дневной лимит ({limit_type}): {Fore.CYAN}{limit:.5f}")

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
    
    for session_name, (start, end) in sessions.items():
        is_active = start <= hour < end
        status = f"{Fore.GREEN}Активна" if is_active else f"{Fore.YELLOW}Неактивна"
        print(f"  {session_name}: {status} ({start}:00-{end}:00 UTC)")

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
    
    for session_name, (start, end) in bot.sessions.items():
        if start <= hour < end:
            in_trading_session = True
            current_session = session_name
            break
    
    conditions.append(("Активная торговая сессия", in_trading_session, current_session))
    
    # 4. Проверка дневного лимита
    dl_set = bot.daily_limit is not None
    conditions.append(("Установлен дневной лимит", dl_set, f"{bot.daily_limit if dl_set else ''}"))
    
    # 5. Нет открытых позиций в том же направлении
    no_open_positions = not any(pos['direction'] == bot.current_context for pos in bot.open_positions)
    conditions.append(("Нет открытых позиций в том же направлении", no_open_positions))
    
    # 6. Целевое расстояние в пределах допустимого
    target_distance_ok = True
    target_distance_value = "Не рассчитано"
    
    if fractals_found:
        for fractal in bot.fractal_levels:
            if (bot.current_context == 'long' and fractal['type'] == 'bullish') or \
               (bot.current_context == 'short' and fractal['type'] == 'bearish'):
                distance = bot.calculate_target_distance(fractal)
                target_distance_ok = distance <= bot.max_target_points
                target_distance_value = f"{int(distance)} из {bot.max_target_points} пунктов"
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
                bot.determine_market_context()
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
            entry_signals = bot.find_entry_points()
            
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
                bot.determine_market_context()
                bot.update_daily_limit()
                bot.find_entry_points()
                
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
                    bot.determine_market_context()
                    bot.update_daily_limit()
                else:
                    print(f"{Fore.RED}Неподдерживаемый символ. Используйте EURUSD или GER40")
                time.sleep(2)
                
            else:
                print(f"{Fore.RED}Неверный ввод")
                time.sleep(1)
                
        except Exception as e:
            print(f"{Fore.RED}Произошла ошибка: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()