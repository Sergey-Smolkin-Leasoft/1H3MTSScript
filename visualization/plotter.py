# visualization/plotter.py
import logging
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime

class Plotter:
    def __init__(self, symbol: str, point_size: float):
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol
        self.point_size = point_size # Может понадобиться для контекста или форматирования

    def _get_price_format_string(self) -> str:
        if self.point_size == 0.0001: return "{:.5f}"
        elif self.point_size == 0.01: return "{:.2f}"
        elif self.point_size == 0.1: return "{:.2f}"
        elif self.point_size == 1: return "{:.1f}"
        return "{:.2f}"

    def plot_strategy(self, data_1h: pd.DataFrame | None, data_3m: pd.DataFrame | None,
                      fractals: list | None, daily_limit: float | None,
                      market_context: str | None, open_positions: list | None,
                      skipped_signals: list | None, pdl: float | None, pdh: float | None,
                      asia_high: float | None, asia_low: float | None, # Добавлено для отображения
                      save_path: str | None = None) -> plt.Figure | None:
        """
        Визуализирует текущее состояние стратегии.
        """
        self.logger.info(f"Попытка построения графика для {self.symbol}...")
        if data_1h is None or data_1h.empty:
            self.logger.warning("Нет данных 1H для построения графика.")
            return None
        if data_3m is None or data_3m.empty: # data_3m тоже важен для полной картины
             self.logger.warning("Нет данных 3M для построения графика.")
             # Можно строить только 1H, если 3M не критичен для этого конкретного вызова
             # return None # Раскомментируйте, если 3M обязателен

        # Используем копии данных для избежания SettingWithCopyWarning
        plot_data_1h = data_1h.tail(72).copy() # Например, последние 3 дня на 1H
        plot_data_3m = data_3m.tail(120).copy() if data_3m is not None else pd.DataFrame() # Последние N свечей 3M/5M

        # Переименование колонок для mplfinance
        for df in [plot_data_1h, plot_data_3m]:
            if not df.empty:
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)


        fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1, ax2 = axes[0], axes[1] # ax1 для 1H, ax2 для 3M

        mc_style = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc_style, gridstyle=':', y_on_right=False)

        # --- График 1H (ax1) ---
        ax1.set_title(f"Стратегия 1H3M для {self.symbol} (1H Контекст)", fontsize=14)
        if not plot_data_1h.empty:
            mpf.plot(plot_data_1h, type='candle', ax=ax1, style=s, ylabel="Цена (1H)", show_nontrading=False)

            # Отображение фракталов
            if fractals:
                for f in fractals:
                    f_time = pd.to_datetime(f['timestamp'])
                    if plot_data_1h.index.min() <= f_time <= plot_data_1h.index.max():
                        marker = '^' if f['type'] == 'bullish' else 'v'
                        color = 'lime' if f['type'] == 'bullish' else 'red'
                        ax1.scatter(f_time, f['price'], color=color, marker=marker, s=80, edgecolors='black', zorder=5,
                                    label=f"{f['type'].capitalize()} Fractal" if f"{f['type']}" not in [h.get_label() for h in ax1.legend().legendHandles] else "")
            # PDL/PDH
            price_fmt = self._get_price_format_string()
            if pdl is not None:
                ax1.axhline(y=pdl, color='blue', linestyle='--', linewidth=1, label=f'PDL: {price_fmt.format(pdl)}')
            if pdh is not None:
                ax1.axhline(y=pdh, color='magenta', linestyle='--', linewidth=1, label=f'PDH: {price_fmt.format(pdh)}')

            # Daily Limit
            if daily_limit is not None:
                ax1.axhline(y=daily_limit, color='purple', linestyle=':', linewidth=1.5, label=f'Daily Limit: {price_fmt.format(daily_limit)}')
            
            # Asia High/Low
            if asia_high is not None:
                ax1.axhline(y=asia_high, color='orange', linestyle='-.', linewidth=1, label=f'Asia High: {price_fmt.format(asia_high)}')
            if asia_low is not None:
                 ax1.axhline(y=asia_low, color='brown', linestyle='-.', linewidth=1, label=f'Asia Low: {price_fmt.format(asia_low)}')

            # Контекст рынка
            if market_context and not plot_data_1h.empty:
                y_text_pos = plot_data_1h['Low'].min() - (plot_data_1h['High'].max() - plot_data_1h['Low'].min()) * 0.05
                ctx_color = 'green' if market_context == 'long' else 'red' if market_context == 'short' else 'gray'
                ax1.text(plot_data_1h.index[0], y_text_pos, f"Контекст: {market_context.upper()}",
                         color=ctx_color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            ax1.legend(loc='best', fontsize='x-small')

        # --- График 3M/5M (ax2) ---
        ax2.set_title(f"{settings.TIMEFRAME_3M} график для входов", fontsize=12)
        if not plot_data_3m.empty:
            mpf.plot(plot_data_3m, type='candle', ax=ax2, style=s, ylabel=f"Цена ({settings.TIMEFRAME_3M})", show_nontrading=False)
            # Открытые позиции на 3М
            if open_positions:
                for pos in open_positions:
                    if pos.get('status') == 'open' and pos.get('symbol') == self.symbol:
                        entry_time = pd.to_datetime(pos['entry_time'])
                        if plot_data_3m.index.min() <= entry_time <= plot_data_3m.index.max():
                            color = 'green' if pos['direction'] == 'long' else 'red'
                            marker = '^' if pos['direction'] == 'long' else 'v'
                            ax2.scatter(entry_time, pos['entry_price'], color=color, marker=marker, s=120, edgecolors='black', zorder=6, label=f"Entry {pos['direction']}")
                            ax2.axhline(y=pos['stop_loss'], color='gray', linestyle=':', linewidth=1, label=f"SL {pos.get('id_internal', '')}")
                            ax2.axhline(y=pos['take_profit'], color='blue', linestyle=':', linewidth=1, label=f"TP {pos.get('id_internal', '')}")
            ax2.legend(loc='best', fontsize='x-small')


        fig.tight_layout()
        if save_path:
            try:
                # Создаем директорию, если ее нет
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                self.logger.info(f"График сохранен в {save_path}")
            except Exception as e:
                self.logger.error(f"Не удалось сохранить график в {save_path}: {e}")
        else:
            plt.show()
        
        return fig