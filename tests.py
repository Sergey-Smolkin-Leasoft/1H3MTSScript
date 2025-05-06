import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Import the trading bot class - assuming it's in trading_bot.py
# If it's in a different file, adjust the import accordingly
from main import TradingBot1H3M

class TestTradingBotInitialization(unittest.TestCase):
    """Tests for the initialization of the TradingBot1H3M class"""
    
    def test_init_eurusd(self):
        """Test initialization with EURUSD symbol"""
        bot = TradingBot1H3M(symbol='EURUSD')
        self.assertEqual(bot.symbol, 'EURUSD')
        self.assertEqual(bot.point_size, 0.0001)
        self.assertEqual(bot.max_target_points, 250)
    
    def test_init_ger40(self):
        """Test initialization with GER40 symbol"""
        bot = TradingBot1H3M(symbol='GER40')
        self.assertEqual(bot.symbol, 'GER40')
        self.assertEqual(bot.point_size, 1)
        self.assertEqual(bot.max_target_points, 400)
    
    def test_init_invalid_symbol(self):
        """Test initialization with invalid symbol"""
        with self.assertRaises(ValueError):
            TradingBot1H3M(symbol='INVALID')

    def test_init_sessions(self):
        """Test that sessions are correctly initialized"""
        bot = TradingBot1H3M(symbol='EURUSD')
        self.assertIn('asia', bot.sessions)
        self.assertIn('london', bot.sessions)
        self.assertIn('newyork', bot.sessions)
        self.assertIn('frankfurt', bot.sessions)
        
        # Check specific session times
        self.assertEqual(bot.sessions['asia']['start'], 1)
        self.assertEqual(bot.sessions['asia']['end'], 9)


class TestDataFetching(unittest.TestCase):
    """Tests for data fetching and generation methods"""
    
    def setUp(self):
        self.bot = TradingBot1H3M(symbol='EURUSD')
    
    @patch('twelvedata.TDClient')
    def test_fetch_historical_data(self, mock_td_client):
        """Test fetching historical data from Twelve Data API"""
        # Set up the mock
        mock_time_series = MagicMock()
        mock_td_client.return_value.time_series.return_value = mock_time_series
        
        # Create a sample DataFrame that the API would return
        sample_data = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=10, freq='1H'),
            'open': np.random.random(10) + 1,
            'high': np.random.random(10) + 1.1,
            'low': np.random.random(10) + 0.9,
            'close': np.random.random(10) + 1,
            'volume': np.random.randint(100, 1000, 10)
        })
        
        mock_time_series.as_pandas.return_value = sample_data
        
        # Call the method
        result = self.bot.fetch_historical_data('1h', 10)
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 10)
        self.assertTrue(all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        
        # Verify the API was called correctly
        mock_td_client.return_value.time_series.assert_called_once_with(
            symbol='EUR/USD',
            interval='1H',
            outputsize=10,
            timezone='UTC'
        )
    
    def test_generate_test_data_1h(self):
        """Test generating test data for 1h timeframe"""
        result = self.bot.generate_test_data('1h')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.index.freq, '1H')
        self.assertEqual(len(result), 240)  # 10 days * 24 hours
        self.assertTrue(all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']))
    
    def test_generate_test_data_3m(self):
        """Test generating test data for 3m timeframe"""
        result = self.bot.generate_test_data('3m')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.index.freq, '3min')
        self.assertEqual(len(result), 1440)  # 3 days * 480 3-minute intervals
        self.assertTrue(all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']))
    
    @patch.object(TradingBot1H3M, 'fetch_historical_data')
    @patch.object(TradingBot1H3M, 'generate_test_data')
    def test_fetch_data_with_exchange(self, mock_generate, mock_fetch):
        """Test fetch_data method with exchange"""
        # Setup
        self.bot.exchange = MagicMock()
        mock_data = pd.DataFrame({
            'open': [1.0, 1.1],
            'high': [1.2, 1.3],
            'low': [0.9, 0.8],
            'close': [1.1, 1.2],
            'volume': [100, 200]
        }, index=pd.date_range(start='2023-01-01', periods=2, freq='1H'))
        
        mock_fetch.return_value = mock_data
        
        # Call the method
        self.bot.fetch_data()
        
        # Verify
        mock_fetch.assert_any_call('1h')
        mock_fetch.assert_any_call('3m')
        self.assertEqual(mock_fetch.call_count, 2)
        mock_generate.assert_not_called()
        
        self.assertIsNotNone(self.bot.data_1h)
        self.assertIsNotNone(self.bot.data_3m)
    
    @patch.object(TradingBot1H3M, 'fetch_historical_data')
    @patch.object(TradingBot1H3M, 'generate_test_data')
    def test_fetch_data_without_exchange(self, mock_generate, mock_fetch):
        """Test fetch_data method without exchange (test mode)"""
        # Setup
        self.bot.exchange = None
        mock_data = pd.DataFrame({
            'open': [1.0, 1.1],
            'high': [1.2, 1.3],
            'low': [0.9, 0.8],
            'close': [1.1, 1.2],
            'volume': [100, 200]
        }, index=pd.date_range(start='2023-01-01', periods=2, freq='1H'))
        
        mock_generate.return_value = mock_data
        
        # Call the method
        self.bot.fetch_data()
        
        # Verify
        mock_generate.assert_any_call('1h')
        mock_generate.assert_any_call('3m')
        self.assertEqual(mock_generate.call_count, 2)
        mock_fetch.assert_not_called()
        
        self.assertIsNotNone(self.bot.data_1h)
        self.assertIsNotNone(self.bot.data_3m)


class TestMarketAnalysis(unittest.TestCase):
    """Tests for market analysis methods"""
    
    def setUp(self):
        self.bot = TradingBot1H3M(symbol='EURUSD')
        
        # Mock data for 1-hour timeframe
        self.mock_data_1h = pd.DataFrame({
            'open': np.random.normal(1.1, 0.01, 100),
            'high': np.random.normal(1.11, 0.01, 100),
            'low': np.random.normal(1.09, 0.01, 100),
            'close': np.random.normal(1.1, 0.01, 100),
            'volume': np.random.randint(100, 1000, 100)
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='1H'))
        
        # Explicitly create a pattern for fractal testing
        # Bullish fractal (V-shape)
        self.mock_data_1h.loc['2023-01-03 10:00:00', 'low'] = 1.05
        self.mock_data_1h.loc['2023-01-03 09:00:00', 'low'] = 1.07
        self.mock_data_1h.loc['2023-01-03 08:00:00', 'low'] = 1.08
        self.mock_data_1h.loc['2023-01-03 11:00:00', 'low'] = 1.07
        self.mock_data_1h.loc['2023-01-03 12:00:00', 'low'] = 1.08
        
        # Bearish fractal (Λ-shape)
        self.mock_data_1h.loc['2023-01-04 10:00:00', 'high'] = 1.15
        self.mock_data_1h.loc['2023-01-04 09:00:00', 'high'] = 1.13
        self.mock_data_1h.loc['2023-01-04 08:00:00', 'high'] = 1.12
        self.mock_data_1h.loc['2023-01-04 11:00:00', 'high'] = 1.13
        self.mock_data_1h.loc['2023-01-04 12:00:00', 'high'] = 1.12
        
        self.bot.data_1h = self.mock_data_1h
        
        # Mock data for 3-minute timeframe
        self.mock_data_3m = pd.DataFrame({
            'open': np.random.normal(1.1, 0.005, 1000),
            'high': np.random.normal(1.105, 0.005, 1000),
            'low': np.random.normal(1.095, 0.005, 1000),
            'close': np.random.normal(1.1, 0.005, 1000),
            'volume': np.random.randint(10, 100, 1000)
        }, index=pd.date_range(start='2023-01-04', periods=1000, freq='3min'))
        
        self.bot.data_3m = self.mock_data_3m
    
    def test_determine_market_context_long(self):
        """Test determining market context as long"""
        # Set closing prices to be above SMA
        self.bot.data_1h['close'] = 1.2
        
        result = self.bot.determine_market_context()
        
        self.assertEqual(result, 'long')
        self.assertEqual(self.bot.current_context, 'long')
    
    def test_determine_market_context_short(self):
        """Test determining market context as short"""
        # Set closing prices to be below SMA
        self.bot.data_1h['close'] = 1.0
        
        result = self.bot.determine_market_context()
        
        self.assertEqual(result, 'short')
        self.assertEqual(self.bot.current_context, 'short')
    
    def test_identify_fractals(self):
        """Test identifying fractals in price data"""
        bullish_fractals, bearish_fractals = self.bot.identify_fractals(self.mock_data_1h)
        
        # Verify we found at least one of each type
        self.assertGreaterEqual(len(bullish_fractals), 1)
        self.assertGreaterEqual(len(bearish_fractals), 1)
        
        # Check format of fractal data
        for fractal in bullish_fractals:
            self.assertIn('timestamp', fractal)
            self.assertIn('price', fractal)
            self.assertEqual(fractal['type'], 'bullish')
        
        for fractal in bearish_fractals:
            self.assertIn('timestamp', fractal)
            self.assertIn('price', fractal)
            self.assertEqual(fractal['type'], 'bearish')
        
        # Verify the specific fractals we created
        bullish_timestamp = pd.Timestamp('2023-01-03 10:00:00')
        bearish_timestamp = pd.Timestamp('2023-01-04 10:00:00')
        
        bullish_found = any(f['timestamp'] == bullish_timestamp for f in bullish_fractals)
        bearish_found = any(f['timestamp'] == bearish_timestamp for f in bearish_fractals)
        
        self.assertTrue(bullish_found, "Expected bullish fractal not found")
        self.assertTrue(bearish_found, "Expected bearish fractal not found")
    
    def test_filter_fractals_by_session(self):
        """Test filtering fractals by trading session"""
        # Create sample fractals at different hours
        fractals = [
            {'timestamp': pd.Timestamp('2023-01-01 03:00:00'), 'price': 1.1, 'type': 'bullish'},  # Asia session
            {'timestamp': pd.Timestamp('2023-01-01 10:00:00'), 'price': 1.12, 'type': 'bearish'},  # London session
            {'timestamp': pd.Timestamp('2023-01-01 15:00:00'), 'price': 1.09, 'type': 'bullish'},  # NY session
            {'timestamp': pd.Timestamp('2023-01-01 08:00:00'), 'price': 1.11, 'type': 'bearish'},  # Frankfurt session
            {'timestamp': pd.Timestamp('2023-01-01 23:00:00'), 'price': 1.10, 'type': 'bullish'}   # No session
        ]
        
        # Test Asia session filter
        asia_fractals = self.bot.filter_fractals_by_session(fractals, 'asia')
        self.assertEqual(len(asia_fractals), 1)
        self.assertEqual(asia_fractals[0]['timestamp'].hour, 3)
        
        # Test London session filter
        london_fractals = self.bot.filter_fractals_by_session(fractals, 'london')
        self.assertEqual(len(london_fractals), 2)  # Should include 10:00 and 15:00
        
        # Test NY session filter
        ny_fractals = self.bot.filter_fractals_by_session(fractals, 'newyork')
        self.assertEqual(len(ny_fractals), 1)
        self.assertEqual(ny_fractals[0]['timestamp'].hour, 15)
        
        # Test Frankfurt session filter
        frankfurt_fractals = self.bot.filter_fractals_by_session(fractals, 'frankfurt')
        self.assertEqual(len(frankfurt_fractals), 1)
        self.assertEqual(frankfurt_fractals[0]['timestamp'].hour, 8)
    
    def test_filter_fractals_by_session_with_days_ago(self):
        """Test filtering fractals by session with days_ago parameter"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        fractals = [
            {'timestamp': pd.Timestamp(today).replace(hour=3), 'price': 1.1, 'type': 'bullish'},
            {'timestamp': pd.Timestamp(yesterday).replace(hour=3), 'price': 1.09, 'type': 'bullish'}
        ]
        
        # Filter for today's Asia session
        today_fractals = self.bot.filter_fractals_by_session(fractals, 'asia', days_ago=0)
        self.assertEqual(len(today_fractals), 1)
        self.assertEqual(today_fractals[0]['timestamp'].date(), today)
        
        # Filter for yesterday's Asia session
        yesterday_fractals = self.bot.filter_fractals_by_session(fractals, 'asia', days_ago=1)
        self.assertEqual(len(yesterday_fractals), 1)
        self.assertEqual(yesterday_fractals[0]['timestamp'].date(), yesterday)


class TestTradingSignals(unittest.TestCase):
    """Tests for trading signal generation and validation"""
    
    def setUp(self):
        self.bot = TradingBot1H3M(symbol='EURUSD')
        
        # Mock data for 1-hour timeframe
        self.mock_data_1h = pd.DataFrame({
            'open': np.random.normal(1.1, 0.01, 100),
            'high': np.random.normal(1.11, 0.01, 100),
            'low': np.random.normal(1.09, 0.01, 100),
            'close': np.random.normal(1.1, 0.01, 100),
            'volume': np.random.randint(100, 1000, 100)
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='1H'))
        
        # Mock data for 3-minute timeframe
        self.mock_data_3m = pd.DataFrame({
            'open': np.random.normal(1.1, 0.005, 1000),
            'high': np.random.normal(1.105, 0.005, 1000),
            'low': np.random.normal(1.095, 0.005, 1000),
            'close': np.random.normal(1.1, 0.005, 1000),
            'volume': np.random.randint(10, 100, 1000)
        }, index=pd.date_range(start='2023-01-04', periods=1000, freq='3min'))
        
        self.bot.data_1h = self.mock_data_1h
        self.bot.data_3m = self.mock_data_3m
        self.bot.current_context = 'long'
        
        # Create sample fractals for testing
        self.bullish_fractal = {
            'timestamp': pd.Timestamp('2023-01-05 03:00:00'),  # Asia session
            'price': 1.095,
            'type': 'bullish'
        }
        
        self.bearish_fractal = {
            'timestamp': pd.Timestamp('2023-01-05 15:00:00'),  # NY session
            'price': 1.115,
            'type': 'bearish'
        }
    
    def test_check_fractal_breakout_bullish(self):
        """Test checking for bullish fractal breakout"""
        # Setup: Make sure there's a breakout in the 3m data
        self.bot.current_context = 'long'
        self.bot.data_3m.iloc[-5:, self.bot.data_3m.columns.get_loc('close')] = 1.1  # Above fractal price
        
        result = self.bot.check_fractal_breakout(self.bullish_fractal)
        self.assertTrue(result)
    
    def test_check_fractal_breakout_bearish(self):
        """Test checking for bearish fractal breakout"""
        # Setup: Make sure there's a breakout in the 3m data
        self.bot.current_context = 'short'
        self.bot.data_3m.iloc[-5:, self.bot.data_3m.columns.get_loc('close')] = 1.11  # Below fractal price
        
        result = self.bot.check_fractal_breakout(self.bearish_fractal)
        self.assertTrue(result)
    
    def test_check_fractal_breakout_no_breakout(self):
        """Test checking when there's no fractal breakout"""
        # Setup: No breakout
        self.bot.current_context = 'long'
        self.bot.data_3m.iloc[-5:, self.bot.data_3m.columns.get_loc('close')] = 1.09  # Below fractal price
        
        result = self.bot.check_fractal_breakout(self.bullish_fractal)
        self.assertFalse(result)
    
    def test_check_skip_conditions_old_fractal(self):
        """Test skip condition for old fractal"""
        # Create an old fractal (3 days ago)
        old_date = datetime.now() - timedelta(days=3)
        old_fractal = {
            'timestamp': old_date,
            'price': 1.1,
            'type': 'bullish'
        }
        
        skip_reasons = self.bot.check_skip_conditions(old_fractal)
        self.assertIn("Фрактал старше 2 дней", skip_reasons)
    
    def test_check_skip_conditions_context_mismatch(self):
        """Test skip condition for fractal contradicting market context"""
        self.bot.current_context = 'long'
        skip_reasons = self.bot.check_skip_conditions(self.bearish_fractal)
        self.assertIn("Фрактал противоречит текущему контексту рынка", skip_reasons)
        
        self.bot.current_context = 'short'
        skip_reasons = self.bot.check_skip_conditions(self.bullish_fractal)
        self.assertIn("Фрактал противоречит текущему контексту рынка", skip_reasons)
    
    def test_check_skip_conditions_existing_position(self):
        """Test skip condition for existing position in same direction"""
        # Setup: Add an open position
        self.bot.open_positions = [{
            'id': 'test-1',
            'direction': 'long',
            'entry_price': 1.1,
            'target': 1.12,
            'stop_loss': 1.09,
            'size': 1.0,
            'entry_time': datetime.now()
        }]
        
        self.bot.current_context = 'long'
        skip_reasons = self.bot.check_skip_conditions(self.bullish_fractal)
        self.assertIn("Уже открыта позиция в данном направлении", skip_reasons)
    
    def test_check_skip_conditions_dl_without_confirmation(self):
        """Test skip condition for daily limit broken without confirmation"""
        # Setup
        self.bot.current_context = 'short'
        self.bot.daily_limit = 1.12
        
        # Make one candle break the DL but not confirm it
        self.bot.data_3m.iloc[-5, self.bot.data_3m.columns.get_loc('low')] = 1.115  # Below DL
        self.bot.data_3m.iloc[-2:, self.bot.data_3m.columns.get_loc('close')] = 1.125  # Above DL
        
        skip_reasons = self.bot.check_skip_conditions(self.bearish_fractal)
        self.assertIn("Снятие DL без закрепления в шортовом контексте", skip_reasons)
    
    @patch.object(TradingBot1H3M, 'calculate_target_distance')
    def test_check_skip_conditions_target_too_far(self, mock_calc_distance):
        """Test skip condition for target being too far"""
        # Setup: Target is too far away
        mock_calc_distance.return_value = 300.0  # For EURUSD max is 250
        
        skip_reasons = self.bot.check_skip_conditions(self.bullish_fractal)
        self.assertTrue(any("Цель превышает максимальное расстояние" in r for r in skip_reasons))
    
    def test_calculate_target_distance_long(self):
        """Test calculating target distance for long position"""
        self.bot.current_context = 'long'
        
        # Setup: current price and a high to target
        self.bot.data_3m.iloc[-1, self.bot.data_3m.columns.get_loc('close')] = 1.10
        self.bot.data_1h.iloc[-10, self.bot.data_1h.columns.get_loc('high')] = 1.15
        
        distance = self.bot.calculate_target_distance(self.bullish_fractal)
        
        # Expected: (1.15 - 1.10) / 0.0001 = 500 points
        self.assertEqual(distance, 500.0)
    
    def test_calculate_target_distance_short(self):
        """Test calculating target distance for short position"""
        self.bot.current_context = 'short'
        
        # Setup: current price and a low to target
        self.bot.data_3m.iloc[-1, self.bot.data_3m.columns.get_loc('close')] = 1.15
        self.bot.data_1h.iloc[-10, self.bot.data_1h.columns.get_loc('low')] = 1.10
        
        distance = self.bot.calculate_target_distance(self.bearish_fractal)
        
        # Expected: (1.15 - 1.10) / 0.0001 = 500 points
        self.assertEqual(distance, 500.0)
    
    def test_calculate_target_long(self):
        """Test calculating target level for long position"""
        self.bot.current_context = 'long'
        
        # Setup: current price and highs within max distance
        current_price = 1.10
        self.bot.data_3m.iloc[-1, self.bot.data_3m.columns.get_loc('close')] = current_price
        
        # Create some high points in the historical data
        self.bot.data_1h.iloc[-48:, self.bot.data_1h.columns.get_loc('high')] = 1.09  # Default
        self.bot.data_1h.iloc[-10, self.bot.data_1h.columns.get_loc('high')] = 1.12  # Within range
        self.bot.data_1h.iloc[-5, self.bot.data_1h.columns.get_loc('high')] = 1.13  # Also within range
        
        target = self.bot.calculate_target(self.bullish_fractal)
        
        # Expected: The highest within range = 1.13
        self.assertEqual(target, 1.13)
    
    def test_calculate_target_short(self):
        """Test calculating target level for short position"""
        self.bot.current_context = 'short'
        
        # Setup: current price and lows within max distance
        current_price = 1.15
        self.bot.data_3m.iloc[-1, self.bot.data_3m.columns.get_loc('close')] = current_price
        
        # Create some low points in the historical data
        self.bot.data_1h.iloc[-48:, self.bot.data_1h.columns.get_loc('low')] = 1.16  # Default
        self.bot.data_1h.iloc[-10, self.bot.data_1h.columns.get_loc('low')] = 1.13  # Within range
        self.bot.data_1h.iloc[-5, self.bot.data_1h.columns.get_loc('low')] = 1.12  # Also within range
        
        target = self.bot.calculate_target(self.bearish_fractal)
        
        # Expected: The lowest within range = 1.12
        self.assertEqual(target, 1.12)
    
    def test_calculate_target_default_when_no_valid_targets(self):
        """Test calculating default target when no valid targets in historical data"""
        self.bot.current_context = 'long'
        
        # Setup: current price but no valid targets in history
        current_price = 1.10
        self.bot.data_3m.iloc[-1, self.bot.data_3m.columns.get_loc('close')] = current_price
        
        # All highs are way outside the valid range
        self.bot.data_1h.iloc[-48:, self.bot.data_1h.columns.get_loc('high')] = 1.20
        
        target = self.bot.calculate_target(self.bullish_fractal)
        
        # Expected: Current price + max target points
        expected_target = current_price + (self.bot.max_target_points * self.bot.point_size)
        self.assertEqual(target, expected_target)


class TestPositionManagement(unittest.TestCase):
    """Tests for position management functions"""
    
    def setUp(self):
        self.bot = TradingBot1H3M(symbol='EURUSD')
        
        # Mock data for 3-minute timeframe
        self.mock_data_3m = pd.DataFrame({
            'open': np.random.normal(1.1, 0.005, 20),
            'high': np.random.normal(1.105, 0.005, 20),
            'low': np.random.normal(1.095, 0.005, 20),
            'close': np.random.normal(1.1, 0.005, 20),
            'volume': np.random.randint(10, 100, 20)
        }, index=pd.date_range(start='2023-01-04', periods=20, freq='3min'))
        
        self.bot.data_3m = self.mock_data_3m
        
        # Sample entry signal
        self.entry_signal = {
            'fractal': {
                'timestamp': pd.Timestamp('2023-01-04 10:00:00'),
                'price': 1.095,
                'type': 'bullish'
            },
            'target': 1.12,
            'entry_price': 1.1,
            'direction': 'long'
        }
    
