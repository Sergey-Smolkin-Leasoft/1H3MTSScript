# analysis/indicators.py

import pandas as pd
import numpy as np
import logging
from ta.trend import SMAIndicator # Используем библиотеку ta для SMA
from ta.volatility import AverageTrueRange # Используем библиотеку ta для ATR

logger = logging.getLogger(__name__)

def calculate_sma(series: pd.Series, window: int) -> pd.Series | None:
    """
    Рассчитывает простую скользящую среднюю (SMA).

    Args:
        series (pd.Series): Временной ряд цен (например, 'close').
        window (int): Окно для расчета SMA.

    Returns:
        pd.Series | None: Временной ряд значений SMA или None при ошибке.
    """
    if not isinstance(series, pd.Series):
        logger.error("Для расчета SMA входные данные должны быть pd.Series.")
        return None
    if series.empty or len(series) < window:
        logger.warning(f"Недостаточно данных для SMA с окном {window}. Длина серии: {len(series)}.")
        return None
    try:
        sma_indicator = SMAIndicator(close=series, window=window, fillna=False)
        sma_series = sma_indicator.sma_indicator()
        logger.debug(f"SMA({window}) рассчитана. Последнее значение: {sma_series.iloc[-1] if not sma_series.empty else 'N/A'}")
        return sma_series
    except Exception as e:
        logger.error(f"Ошибка при расчете SMA({window}): {e}", exc_info=True)
        return None

def calculate_atr(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, window: int = 14) -> pd.Series | None:
    """
    Рассчитывает средний истинный диапазон (ATR).

    Args:
        high_series (pd.Series): Временной ряд максимумов.
        low_series (pd.Series): Временной ряд минимумов.
        close_series (pd.Series): Временной ряд цен закрытия.
        window (int, optional): Окно для расчета ATR. По умолчанию 14.

    Returns:
        pd.Series | None: Временной ряд значений ATR или None при ошибке.
    """
    if not (isinstance(high_series, pd.Series) and isinstance(low_series, pd.Series) and isinstance(close_series, pd.Series)):
        logger.error("Для расчета ATR все входные данные (high, low, close) должны быть pd.Series.")
        return None

    required_length = window + 1 # ta.volatility.AverageTrueRange требует n+1 для окна n
    if high_series.empty or len(high_series) < required_length or \
       low_series.empty or len(low_series) < required_length or \
       close_series.empty or len(close_series) < required_length:
        logger.warning(f"Недостаточно данных для ATR с окном {window}. Требуется {required_length} баров.")
        return None
    try:
        atr_indicator = AverageTrueRange(high=high_series, low=low_series, close=close_series, window=window, fillna=False)
        atr_series = atr_indicator.average_true_range()
        logger.debug(f"ATR({window}) рассчитан. Последнее значение: {atr_series.iloc[-1] if not atr_series.empty else 'N/A'}")
        return atr_series
    except Exception as e:
        logger.error(f"Ошибка при расчете ATR({window}): {e}", exc_info=True)
        return None

# Другие индикаторы можно добавить сюда по аналогии