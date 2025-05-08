# 1h3m_trading_bot/main_runner.py

import time
import traceback
from datetime import datetime

# --- Импорты из ваших модулей ---
# Предполагается, что вы уже создали эти файлы и каталоги
# и разместили в них соответствующую логику

# Конфигурация и настройки
from config import settings

# Настройка логгера
from utils.logging_config import setup_logger

# Основной класс бота
from core.bot import TradingBot1H3M

# Функции консольного интерфейса
from ui.console_interface import (
    clear_screen,
    print_header,
    print_market_context,
    print_condition,
    print_conditions_wrapper, # Используем обертку
    print_signal,
    print_open_positions,
    print_fractal_levels,
    print_skip_conditions,
    print_daily_limit,
    print_session_status,
    print_menu,
    check_entry_conditions_for_ui # Функция для сбора данных для UI
)
# Цвета для UI (если они вынесены или нужны здесь)
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    print("Warning: colorama not installed. Console output might lack color.")
    # Определяем заглушки, если colorama нет
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = Style = DummyColor()

# -------------------------------------

def run_bot():
    """Основная функция запуска бота и интерфейса"""
    # 1. Настройка логгера
    logger = setup_logger()
    logger.info("Запуск основного приложения бота...")

    # 2. Инициализация бота с начальным символом из настроек
    try:
        initial_symbol = settings.DEFAULT_SYMBOL # Замените на имя вашей настройки
        bot = TradingBot1H3M(symbol=initial_symbol)
        logger.info(f"Бот успешно инициализирован для символа {initial_symbol}.")
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации бота: {e}", exc_info=True)
        print(f"{Fore.RED}Не удалось инициализировать бота: {e}. Проверьте логи.")
        return # Выход, если бот не инициализировался

    active_signals = [] # Храним активные сигналы между итерациями

    # 3. Основной цикл приложения
    while True:
        try:
            # Очистка экрана и заголовок
            clear_screen()
            print_header(bot.symbol if hasattr(bot, 'symbol') else "N/A") # Передаем символ в заголовок

            # --- Получение и обновление данных ---
            # Загрузка/обновление данных (бот сам решает, нужно ли обновлять)
            # Внутри bot.run_cycle() должны вызываться fetch_data, analyze_market_context, update_daily_limit и т.д.
            # Либо вызываем их явно:
            bot.fetch_data() # Метод fetch_data должен обрабатывать ошибки и логировать
            if bot.data_1h is None or bot.data_3m is None:
                logger.warning("Пропуск итерации: данные не загружены.")
                print(f"{Fore.YELLOW}Ожидание данных...")
                time.sleep(settings.DATA_FETCH_INTERVAL_SECONDS) # Используем интервал из настроек
                continue

            # --- Анализ и поиск сигналов ---
            bot.analyze_market_context() # Анализируем контекст
            bot.update_daily_limit() # Обновляем дневной лимит
            active_signals = bot.find_entry_signals() # Ищем сигналы

            # --- Отображение информации ---
            print_market_context(bot.current_context)
            print_daily_limit(bot.daily_limit, bot.current_context, bot.point_size) 
            print_fractal_levels(bot.fractal_levels, bot.current_context, bot.point_size)
            print_session_status()

            # Получаем и выводим условия для UI
            conditions_data = check_entry_conditions_for_ui(bot)
            print_conditions_wrapper(conditions_data) # Используем обертку

            # Вывод сигналов
            if active_signals:
                for signal in active_signals:
                     # Передаем bot.point_size для корректного расчета пунктов
                    print_signal(signal, bot.symbol, bot.point_size)
            else:
                print(f"\n{Fore.YELLOW}Нет активных сигналов для входа")

            print_skip_conditions(bot.skip_conditions)
            print_open_positions(bot.open_positions, bot.get_current_price(), bot.point_size) # Передаем текущую цену и размер пункта

            # --- Управление позициями ---
            bot.manage_open_positions()

            # --- Меню и ввод пользователя ---
            print_menu()
            choice = input(f"{Fore.WHITE}Введите номер действия: ")

            # --- Обработка выбора пользователя ---
            if choice == '0':
                logger.info("Пользователь выбрал выход.")
                print(f"{Fore.GREEN}Выход из программы...")
                break

            elif choice == '1':
                logger.info("Пользователь выбрал обновить данные.")
                print(f"\n{Fore.YELLOW}Обновление данных и повторный анализ...")
                # Принудительно вызываем цикл обновления и анализа в боте
                # bot.fetch_data() # Уже вызвали в начале цикла
                # bot.analyze_market_context()
                # bot.update_daily_limit()
                # active_signals = bot.find_entry_signals()
                # Нет необходимости в явных вызовах здесь, т.к. они уже произошли
                time.sleep(1) # Небольшая пауза для отображения сообщения

            elif choice == '2':
                logger.info("Пользователь выбрал выполнить сделку.")
                if active_signals:
                    print(f"\n{Fore.YELLOW}Попытка выполнения сделки по первому активному сигналу...")
                    # Обычно берем первый подходящий сигнал
                    signal_to_execute = active_signals[0]
                    executed = bot.execute_trade(signal_to_execute) # Метод должен возвращать True/False
                    if executed:
                        print(f"{Fore.GREEN}Сделка по сигналу отправлена (или симулирована).")
                        active_signals = [] # Сбрасываем сигналы после попытки исполнения
                    else:
                        print(f"{Fore.RED}Не удалось выполнить сделку.")
                    time.sleep(2)
                else:
                    print(f"\n{Fore.RED}Нет активных сигналов для выполнения сделки.")
                    time.sleep(2)

            elif choice == '3':
                logger.info("Пользователь выбрал показать график.")
                print(f"\n{Fore.YELLOW}Генерация графика...")
                try:
                    save_path = f"{settings.CHART_SAVE_PATH}strategy_chart_{bot.symbol}_{datetime.now():%Y%m%d_%H%M%S}.png" # Путь из настроек
                    bot.visualize_strategy(save_path=save_path)
                    print(f"{Fore.GREEN}График сохранен в файл: {save_path}")
                except Exception as viz_err:
                     logger.error(f"Ошибка при генерации или сохранении графика: {viz_err}", exc_info=True)
                     print(f"{Fore.RED}Ошибка визуализации: {viz_err}")
                input("Нажмите Enter для продолжения...") # Пауза, чтобы пользователь увидел сообщение

            elif choice == '4':
                logger.info("Пользователь выбрал сменить символ.")
                current_sym = bot.symbol
                supported_symbols = list(settings.INSTRUMENT_SETTINGS.keys()) # Получаем список из настроек
                print(f"Доступные символы: {', '.join(supported_symbols)}")
                new_symbol = input(f"{Fore.WHITE}Введите новый символ ({'/'.join(supported_symbols)}): ").upper()
                if new_symbol in supported_symbols:
                    if new_symbol != current_sym:
                        try:
                            print(f"\n{Fore.YELLOW}Смена символа на {new_symbol}...")
                            # Создаем новый экземпляр бота
                            bot = TradingBot1H3M(symbol=new_symbol)
                            active_signals = [] # Сбрасываем сигналы при смене символа
                            logger.info(f"Символ успешно изменен на {new_symbol}.")
                            print(f"{Fore.GREEN}Символ изменен. Данные будут загружены в следующей итерации.")
                        except Exception as e:
                            logger.critical(f"Критическая ошибка при переинициализации бота для {new_symbol}: {e}", exc_info=True)
                            print(f"{Fore.RED}Не удалось переключить символ: {e}.")
                            # Пытаемся вернуть старого бота, если возможно
                            try:
                                bot = TradingBot1H3M(symbol=current_sym)
                                logger.warning(f"Возвращен предыдущий символ {current_sym} из-за ошибки.")
                            except Exception:
                                logger.critical("Не удалось вернуть предыдущий символ. Завершение работы.")
                                print(f"{Fore.RED}Не удалось восстановить предыдущий символ. Выход.")
                                break # Выход из цикла, если все плохо
                    else:
                         print(f"{Fore.YELLOW}Выбран текущий символ.")
                else:
                    print(f"{Fore.RED}Неподдерживаемый символ.")
                time.sleep(2)

            else:
                logger.warning(f"Неверный ввод пользователя: {choice}")
                print(f"{Fore.RED}Неверный ввод.")
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Получен сигнал KeyboardInterrupt. Завершение работы...")
            print(f"\n{Fore.YELLOW}Завершение работы по запросу пользователя...")
            break
        except Exception as e:
            logger.error(f"Необработанная ошибка в главном цикле: {e}", exc_info=True)
            print(f"\n{Fore.RED}{Style.BRIGHT}Произошла критическая ошибка: {e}")
            print(traceback.format_exc())
            input("Нажмите Enter для попытки продолжения или Ctrl+C для выхода...")
            time.sleep(5) # Пауза перед следующей попыткой

    # Очистка ресурсов при выходе (если необходимо, например, закрытие соединений)
    # bot.cleanup() # Если есть такой метод

    logger.info("Приложение бота завершило работу.")
    print(f"{Fore.CYAN}Работа бота завершена.")

# --- Точка входа ---
if __name__ == "__main__":
    run_bot()