import math
import json
import os
from warnings import deprecated
from sklearn.utils.extmath import weighted_mode
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
from datetime import datetime, timedelta, timezone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, mean_absolute_error, mean_squared_error, r2_score
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN
import supabase
from dotenv import load_dotenv

PREDICTION_LOG_PATH = os.environ.get(
    "LEVELEDGE_PREDICTION_LOG_PATH", "prediction_logs.jsonl"
)


class Predictor:

    def __init__(self, ticker_str: str, target_datetime: datetime, interval: str, price: float, end_datetime: datetime = None, data: pd.DataFrame = None) -> None:
        """
        Constructor creates object with a ticker, target datetime for prediction time
        and an interval to use to fetch data

        Parameters
        ==========
        ticker_str: str - lower or upper that holds the ticker symbol for the stock or crypto of interest
        target_datetime: datetime - datetime that represents when to predict for
        interval: str - string format of the interval to use for stock/crypto data
        end_datetime: datetime - if provided, treats this as a backtest (logs to backtest_logs)
        data: pd.DataFrame - optional pre-fetched OHLCV DataFrame (DatetimeIndex, US/Eastern).
                             When provided, the yfinance history fetch is skipped entirely.
                             Slice to end_datetime before passing in.

        Errors
        ======
        Raises ValueError if invalid interval is passed in
        Raises ValueError if Target Datetime is in the past

        Return
        ======
        None
        """

        self.price: float = price
        self.ticker_str: str = ticker_str.upper().strip()
        self.ticker: yf.ticker.Ticker = yf.ticker.Ticker(ticker_str)
        self.target_datetime: datetime = target_datetime
        self.interval: str = interval
        self.isCrypto: bool = '-' in self.ticker_str
        self.hasPrePost: bool = self.ticker.get_info()['hasPrePostMarketData'] # False for cryptos

        self.end_datetime = end_datetime if end_datetime is not None else datetime.now(tz=US_EASTERN)
        self.is_backtest = end_datetime is not None

        if self.interval not in ALLOWED_INTERVALS:
            raise ValueError("Invalid interval input.")

        if self.target_datetime < self.end_datetime:
            raise ValueError("Target Datetime Must be after end datetime.")

        if 'm' in self.interval:
            self.interval_min: int = int(self.interval[:-1])
        elif 'h' in self.interval:
            self.interval_min: int = 60 * int(self.interval[:-1])
        elif 'd' in self.interval:
            self.interval_min: int = 60 * 24 * int(self.interval[:-1])

        if data is not None:
            # Use caller-supplied DataFrame; skip yfinance fetch entirely.
            self.data: pd.DataFrame = data.copy()
        elif self.is_backtest:
            startDate = datetime.now(tz=US_EASTERN) - timedelta(days=59)  # most data we can get from yfinance
            self.data: pd.DataFrame = self.ticker.history(start=startDate, end=self.end_datetime, interval=self.interval, prepost=self.hasPrePost)
        else:
            self.data: pd.DataFrame = self.ticker.history(period='max', interval=self.interval, prepost=self.hasPrePost)

        self.current_price: float = self.data['Close'].iloc[-1]
        self.target_price_ratio: float = self.price / self.current_price
        self.candles_ahead: int = self._calculate_candles_ahead_crypto() if self.isCrypto else self._calculate_candles_ahead_stocks()
        self.features: pd.DataFrame = self.prepare_features()

        self._create_target_variable()

    def _log_prediction(self, prediction: float, is_backtest: bool = False) -> None:
        try:
            from dotenv import load_dotenv
            from supabase import create_client
            
            load_dotenv()
            
            supabase = create_client(
                os.environ["SUPABASE_URL"],
                os.environ["SUPABASE_KEY"]
            )

            if is_backtest:
                table_name = "backtest_logs"
                now_utc = self.end_datetime.astimezone(timezone.utc)
            else:
                table_name = "logs"
                now_utc = datetime.now(timezone.utc)

            record = {
                "logged_at_utc": now_utc.isoformat(),
                "ticker": self.ticker_str,
                "is_crypto": self.isCrypto,
                "interval": self.interval,
                "interval_minutes": getattr(self, "interval_min", None),
                "target_datetime": self.target_datetime.isoformat()
                    if isinstance(self.target_datetime, datetime)
                    else str(self.target_datetime),
                "price_level": self.price,
                "current_price": getattr(self, "current_price", None),
                "target_price_ratio": getattr(self, "target_price_ratio", None),
                "candles_ahead": getattr(self, "candles_ahead", None),
                "prediction": float(prediction) if prediction is not None else None,
                "model_type": "xgboost_classifier",
                "model_auc": self.xgb_expected_model_metrics[0]
                    if hasattr(self, "xgb_expected_model_metrics") and self.xgb_expected_model_metrics else None,
                "model_ps": self.xgb_expected_model_metrics[1]
                    if hasattr(self, "xgb_expected_model_metrics") and self.xgb_expected_model_metrics else None,
                "model_pr": self.xgb_expected_model_metrics[2]
                    if hasattr(self, "xgb_expected_model_metrics") and self.xgb_expected_model_metrics else None,
            }

            supabase.table(table_name).insert(record).execute()

        except Exception:
            # Logging must never break prediction flow.
            pass

    def _create_target_variable(self) -> None:
        """
        Creates target variable for each row
        (if close of n candles ahead is greater than inputted price)
        """

        data = self.data.copy()

        data['Future_close'] = data['Close'].shift(-self.candles_ahead)

        # data['Target'] = (data['Future_close'] > self.price).astype(int)

        # will future price
        data['Target'] = ((data['Future_close'] / data['Close']) > self.target_price_ratio).astype(int)

        # Continuous regression target: ratio of future close to current close
        data['Future_close_ratio'] = data['Future_close'] / data['Close']

        data['Datetime'] = data.index

        # print(f'Target Price Ratio: {self.target_price_ratio}')

        self.data_withna = data

        data = data.dropna().reset_index(drop=True)

        self.data = data


    def _calculate_candles_ahead_stocks(self) -> int:
        """Calculates how many candles ahead the target datetime is from current candle in market hours."""
        nyse = mcal.get_calendar('NYSE')
        current_datetime = self.data.index[-1]
        delta: timedelta = self.target_datetime - current_datetime

        non_market_hours_per_day = 8 if self.hasPrePost else 17.5

        market_days: int = 0
        non_market_days: int = 0

        # Iterate over every calendar date strictly after current_datetime up to
        # and including target_datetime. Using date arithmetic (rather than
        # range(1, delta.days+1)) correctly handles the case where delta < 24 h
        # but the window spans overnight non-market hours (delta.days == 0 even
        # though trading days are involved, e.g. predict at 3 PM for next day 10 AM).
        current_date = current_datetime.date()
        target_date = self.target_datetime.date()
        check_date = current_date + timedelta(days=1)
        while check_date <= target_date:
            ts = pd.Timestamp(check_date)
            schedule = nyse.schedule(start_date=ts, end_date=ts)
            if schedule.empty:
                non_market_days += 1
            else:
                market_days += 1
            check_date += timedelta(days=1)

        if self.interval_min == 1440:  # interval is 1d
            return market_days

        hours_to_subtract: float = market_days * non_market_hours_per_day + non_market_days * 24
        market_hours: float = (delta - timedelta(hours=hours_to_subtract, minutes=1)).total_seconds() / 60 / 60
        candles: int = int(market_hours / (self.interval_min / 60))

        if candles <= 0:
            raise ValueError(
                f"candles_ahead={candles}: end_datetime is too close to target_datetime "
                f"for interval '{self.interval}'. Use a later target or earlier end_datetime."
            )

        return candles

    def _calculate_candles_ahead_crypto(self) -> int:
        """Calculates how may candles ahead the target datetime is from the current candle for cryptos (24hr)."""

        current_datetime = self.data.index[-1]
        delta: timedelta = self.target_datetime - current_datetime - timedelta(minutes=1)

        return int(delta.total_seconds() / 60 / self.interval_min)
        
    def _calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate various technical indicators"""
        data = self.data.copy()
        
        # Simple Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['SMA_400'] = data['Close'].rolling(window=400).mean()
        
        # Exponential Moving Averages
        data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        
        # ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price change indicators
        data['Price_change'] = data['Close'].pct_change()
        data['Price_change_5'] = data['Close'].pct_change(periods=5)
        data['High_Low_ratio'] = data['High'] / data['Low']
        
        # Previous day high and low (works for any interval: group by date, then shift)
        date_norm = data.index.normalize()
        day_agg = data.groupby(date_norm).agg({'High': 'max', 'Low': 'min'})
        prev_day_agg = day_agg.shift(1)
        data['Prev_day_high'] = date_norm.map(prev_day_agg['High'])
        data['Prev_day_low'] = date_norm.map(prev_day_agg['Low'])
        
        # Previous hour high and low (skip for 1d interval; for intraday group by clock hour)
        if self.interval_min < 60 * 24:  # intraday only
            hour_floor = data.index.floor('h')
            hour_agg = data.groupby(hour_floor).agg({'High': 'max', 'Low': 'min'})
            prev_hour_agg = hour_agg.shift(1)
            data['Prev_hour_high'] = hour_floor.map(prev_hour_agg['High'])
            data['Prev_hour_low'] = hour_floor.map(prev_hour_agg['Low'])
        
        self.data = data
        return self.data
    
    def _calculate_candlestick_patterns(self) -> pd.DataFrame:
        """Calculate candlestick pattern features"""
        data = self.data.copy()
        
        # Body, Upper Shadow, Lower Shadow
        data['Body'] = abs(data['Close'] - data['Open'])
        data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
        data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        data['Total_Range'] = data['High'] - data['Low']
        
        # Body to range ratio
        data['Body_Ratio'] = data['Body'] / data['Total_Range']
        data['Upper_Shadow_Ratio'] = data['Upper_Shadow'] / data['Total_Range']
        data['Lower_Shadow_Ratio'] = data['Lower_Shadow'] / data['Total_Range']
        
        # Candlestick type
        data['Candle_Type'] = np.where(data['Close'] > data['Open'], 1, -1)
        
        # Patterns
        data['Doji'] = (data['Body_Ratio'] < 0.1).astype(int)
        data['Hammer'] = ((data['Body_Ratio'] < 0.3) & 
                          (data['Lower_Shadow_Ratio'] > 0.6) & 
                          (data['Upper_Shadow_Ratio'] < 0.2)).astype(int)
        data['Shooting_Star'] = ((data['Body_Ratio'] < 0.3) & 
                                (data['Upper_Shadow_Ratio'] > 0.6) & 
                                (data['Lower_Shadow_Ratio'] < 0.2)).astype(int)
        
        # Engulfing patterns
        prev_bullish = (data['Close'].shift(1) > data['Open'].shift(1)).astype(int)
        curr_bullish = (data['Close'] > data['Open']).astype(int)
        prev_body = abs(data['Close'].shift(1) - data['Open'].shift(1))
        curr_body = abs(data['Close'] - data['Open'])
        
        data['Bullish_Engulfing'] = ((prev_bullish == 0) & 
                                      (curr_bullish == 1) & 
                                      (data['Open'] < data['Close'].shift(1)) & 
                                      (data['Close'] > data['Open'].shift(1)) & 
                                      (curr_body > prev_body * 1.1)).astype(int)
        
        data['Bearish_Engulfing'] = ((prev_bullish == 1) & 
                                      (curr_bullish == 0) & 
                                      (data['Open'] > data['Close'].shift(1)) & 
                                      (data['Close'] < data['Open'].shift(1)) & 
                                      (curr_body > prev_body * 1.1)).astype(int)
        
        self.data = data
        return self.data
    
    def _calculate_residual_features(self) -> pd.DataFrame:
        """
        Calculate features that correlate with move magnitude rather than direction:
        VWAP, realized volatility, ATR-normalized returns, z-scores, lagged returns,
        gap, OBV, volume-weighted momentum, and price acceleration.
        """
        data = self.data.copy()

        log_ret = np.log(data['Close'] / data['Close'].shift(1))

        # Realized volatility (rolling std of log returns) — volatility clusters
        data['RVol_5']  = log_ret.rolling(5).std()
        data['RVol_10'] = log_ret.rolling(10).std()
        data['RVol_20'] = log_ret.rolling(20).std()

        # ATR-normalized return — return scaled by recent volatility
        data['ATR_norm_return'] = log_ret / data['ATR'].replace(0, np.nan)

        # Range expansion — how wide is this candle vs average (ATR)?
        candle_range = data['High'] - data['Low']
        data['Range_vs_ATR'] = candle_range / data['ATR'].replace(0, np.nan)

        # Z-score of close relative to rolling mean and std
        roll_mean = data['Close'].rolling(20).mean()
        roll_std  = data['Close'].rolling(20).std().replace(0, np.nan)
        data['Close_Zscore_20'] = (data['Close'] - roll_mean) / roll_std

        # VWAP (resets each calendar day for intraday; rolling for daily)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        tp_vol = typical_price * data['Volume']
        if self.interval_min < 1440:
            date_norm = data.index.normalize()
            data['VWAP'] = (tp_vol.groupby(date_norm).cumsum() /
                            data['Volume'].groupby(date_norm).cumsum())
        else:
            data['VWAP'] = tp_vol.rolling(20).sum() / data['Volume'].rolling(20).sum()

        # Distance from VWAP (normalized) — mean-reversion / trend signal
        data['VWAP_dist'] = (data['Close'] - data['VWAP']) / data['VWAP'].replace(0, np.nan)

        # Gap: open vs previous close (overnight or between-candle gap)
        data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)

        # Lagged log returns — autocorrelation structure
        for lag in [1, 2, 3, 5]:
            data[f'Ret_lag_{lag}'] = log_ret.shift(lag)

        # Price acceleration — change in momentum (2nd derivative of price)
        data['Price_accel'] = log_ret - log_ret.shift(1)

        # Volume-weighted momentum — big moves on high volume are more significant
        data['Vol_weighted_momentum'] = log_ret * data['Volume_ratio']

        # OBV (On-Balance Volume) — cumulative volume pressure
        obv = (np.sign(log_ret) * data['Volume']).fillna(0).cumsum()
        data['OBV'] = obv
        data['OBV_SMA'] = obv.rolling(20).mean()
        data['OBV_ratio'] = obv / data['OBV_SMA'].replace(0, np.nan)

        # Consecutive candle streak (positive = N bullish in a row, negative = bearish)
        direction = np.sign(log_ret).fillna(0)
        streak = []
        s = 0
        for d in direction:
            if d == 0:
                streak.append(s)
            elif (s >= 0 and d > 0) or (s <= 0 and d < 0):
                s += int(d)
                streak.append(s)
            else:
                s = int(d)
                streak.append(s)
        data['Candle_streak'] = streak

        self.data = data
        return self.data

    def prepare_features(self) -> pd.DataFrame:
        """Creates and adds features to data, then prepares feature set for the model"""

        # Calculate all indicators
        self._calculate_technical_indicators()
        self._calculate_candlestick_patterns()
        self._calculate_residual_features()

        # Define feature columns (Prev_hour_* only present for intraday intervals)
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'SMA_500'
            'EMA_5', 'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            'ATR',
            'Volume_SMA', 'Volume_ratio',
            'Price_change', 'Price_change_5', 'High_Low_ratio',
            'Prev_day_high', 'Prev_day_low',
            'Prev_hour_high', 'Prev_hour_low',
            'Body', 'Upper_Shadow', 'Lower_Shadow', 'Total_Range',
            'Body_Ratio', 'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
            'Candle_Type', 'Doji', 'Hammer', 'Shooting_Star',
            'Bullish_Engulfing', 'Bearish_Engulfing',
            # Residual / magnitude features
            'RVol_5', 'RVol_10', 'RVol_20',
            'ATR_norm_return', 'Range_vs_ATR',
            'Close_Zscore_20',
            'VWAP', 'VWAP_dist',
            'Gap',
            'Ret_lag_1', 'Ret_lag_2', 'Ret_lag_3', 'Ret_lag_5',
            'Price_accel',
            'Vol_weighted_momentum',
            'OBV', 'OBV_SMA', 'OBV_ratio',
            'Candle_streak',
        ]

        # Select available features (Prev_hour_* omitted for 1d) and drop NaN rows
        available_features = [col for col in feature_columns if col in self.data.columns]
        self.available_features = available_features
        features = self.data[available_features].copy()
        features = features.dropna()

        self.features = features
        return self.features

    def _walk_forward_split(self, train_size: int = 200, test_size: int = 40, step: int = 200, start: int = 0):
        splits = []

        # print(f'Data Length: {len(self.data)},  vs. {start + train_size + test_size}')

        while start + train_size + test_size <= len(self.data):
            train_idx = slice(start, start + train_size)
            test_idx = slice(start + train_size, start + train_size + test_size)
            splits.append((train_idx, test_idx))
            start += step

        return splits


    # def train_rfc(self) -> None:
    #     splits = self._walk_forward_split()
    #     auc_scores = []
    #     ps_scores = []
    #     models = []

    #     print(f"\nOverall class distribution:")
    #     print(self.data['Target'].value_counts())
    #     print(f"Overall positive rate: {self.data['Target'].mean():.2%}\n")

    #     for i, (train_idx, test_idx) in enumerate(splits):
    #         train = self.data.iloc[train_idx]
    #         test = self.data.iloc[test_idx]

    #         X_train = train[self.available_features]
    #         y_train = train['Target']
    #         X_test = test[self.available_features]
    #         y_test = test['Target']

    #         print(f'Split: {i+1}')
    #         print(f"  Train: {len(y_train)} samples, {y_train.sum()} positive ({y_train.mean():.2%})")
    #         print(f"  Test:  {len(y_test)} samples, {y_test.sum()} positive ({y_test.mean():.2%})")

    #         # Notify invalid splits
    #         if len(y_test.unique()) < 2:
    #             print(f"  ⚠️  IMBALANCE - only class {y_test.unique()[0]} in test set\n")
                
    #         if len(y_train.unique()) < 2:
    #             print(f"  ⚠️  IMBALANCE - only class {y_train.unique()[0]} in train set\n")
                
    #         # Calculate class weight
    #         n_negative = (y_train == 0).sum()
    #         n_positive = (y_train == 1).sum()
    #         scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1

    #         print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    #         model = RFC(
    #             n_estimators=100,
    #             max_depth=10,
    #             min_samples_split=5,
    #             min_samples_leaf=2,
    #             class_weight='balanced',  # This helps with imbalanced data
    #             random_state=42,
    #             n_jobs=-1
    #         )

    #         model.fit(X_train, y_train)
    #         preds = model.predict_proba(X_test)[:, 1]
    #         preds_binary = (preds >= .6).astype(int)
    #         auc = roc_auc_score(y_test, preds)
    #         ps = precision_score(y_test, preds_binary)
    #         auc_scores.append(auc)
    #         ps_scores.append(ps)
    #         print(f"  AUC: {auc:.4f}    PS: {ps:.4f}\n")
    #         models.append((model, ps*auc))

    #     self.rfc_models = models
        
    #     print(f"\n{'='*50}")
    #     print(f"Valid AUC, PS scores: {auc_scores}, {ps_scores}")
    #     if auc_scores:
    #         print(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    #         print(f"Mean PS: {np.mean(ps_scores):.4f} ± {np.std(ps_scores):.4f}")
    #     print(f"{'='*50}")




    def train_xgb(self, evaluate: bool = True) -> None:
        if evaluate:
            length = len(self.data)
            splits = self._walk_forward_split(int(length/4), 100, int(length/4), 0)
            auc_scores = []
            ps_scores = []
            pr_scores = []

            for i, (train_idx, test_idx) in enumerate(splits):
                train = self.data.iloc[train_idx]
                test = self.data.iloc[test_idx]

                X_train = train[self.available_features]
                y_train = train['Target']
                X_test = test[self.available_features]
                y_test = test['Target']

                if len(y_test.unique()) < 2 or len(y_train.unique()) < 2:
                    continue

                n_negative = (y_train == 0).sum()
                n_positive = (y_train == 1).sum()
                scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1

                model = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="binary:logistic", eval_metric="auc",
                    tree_method="hist", scale_pos_weight=scale_pos_weight
                )

                model.fit(X_train, y_train)
                preds = model.predict_proba(X_test)[:, 1]
                preds_binary = (preds >= .6).astype(int)
                auc_scores.append(roc_auc_score(y_test, preds))
                ps_scores.append(precision_score(y_test, preds_binary))
                pr_scores.append(average_precision_score(y_test, preds))

            avg_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0
            avg_ps = sum(ps_scores) / len(ps_scores) if ps_scores else 0
            avg_pr = sum(pr_scores) / len(pr_scores) if pr_scores else 0
            self.xgb_expected_model_metrics = (avg_auc, avg_ps, avg_pr)
        else:
            self.xgb_expected_model_metrics = (0, 0, 0)

        # Train model on all data
        X_train = self.data[self.available_features]
        y_train = self.data['Target']

        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1

        self.xgb_model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", scale_pos_weight=scale_pos_weight
        )

        self.xgb_model.fit(X_train, y_train)

    def print_xgb_model_metrics(self) -> None:
        print(f"Expected Model Metrics: AUC: {self.xgb_expected_model_metrics[0]:.4f}, PS: {self.xgb_expected_model_metrics[1]:.4f}, PR: {self.xgb_expected_model_metrics[2]:.4f}")

    def get_xgb_model_metrics(self) -> tuple[float, float, float]:
        return self.xgb_expected_model_metrics

    def print_candles_ahead(self) -> None:
        print(f'Predicting for {self.candles_ahead} candles ahead')
        print(f'Most recent candle datetime: {self.data_withna["Datetime"].iloc[-1]}')

    def predict_xgb(self) -> float:
        """
        Returns prediction using xgb for target datetime and price as a boolean
        (will ticker be above price level at target datetime?)

        Parameters
        ==========
        self - pass in object, has data, model, etc.

        Returns
        =======
        boolean - represents prediction
        """


        prediction = self.xgb_model.predict_proba(
            self.data_withna[self.available_features]
        )[-1][1]

        # Persist this prediction event for later analysis.
        self._log_prediction(prediction, self.is_backtest)

        return prediction

        # best_model = self.xgb_models[0][0]
        # best_psauc = self.xgb_models[0][1]

        # for (model, psauc) in self.xgb_models:
            # if psauc > best_psauc:
                # best_model = model
                # best_psauc = psauc
        
        # predictions = best_model.predict_proba(self.data_withna[self.available_features])

        # counter: int = 0
        # mean_prediction: float = 0

        # for (model, psauc) in self.xgb_models:
        #     if psauc and not math.isnan(psauc) and psauc != 0:
        #         counter += 1
        #         mean_prediction += model.predict_proba(self.data_withna[self.available_features])[-1][1]

        # mean_prediction /= counter



        # # print('Prediction')
        # # print(f'P(<{self.price}), P(>{self.price})')
        # # print(f'{predictions[-1][0]:.2%}, {predictions[-1][1]:.4%}')

        # return mean_prediction


    def train_regression(self) -> None:
        """
        Train an XGBRegressor to predict the future close / current close ratio
        using the same walk-forward split strategy as train_xgb.

        Stores:
            self.regression_model               — final model trained on all data
            self.regression_expected_model_metrics — (mae, rmse, r2) averaged over splits
        """
        length = len(self.data)
        splits = self._walk_forward_split(int(length / 4), 100, int(length / 4), 0)
        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for train_idx, test_idx in splits:
            train = self.data.iloc[train_idx]
            test = self.data.iloc[test_idx]

            X_train = train[self.available_features]
            y_train = train['Future_close_ratio']
            X_test = test[self.available_features]
            y_test = test['Future_close_ratio']

            if y_train.isna().all() or y_test.isna().all():
                continue

            model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae_scores.append(mean_absolute_error(y_test, preds))
            rmse_scores.append(mean_squared_error(y_test, preds) ** 0.5)
            r2_scores.append(r2_score(y_test, preds))

        avg_mae = sum(mae_scores) / len(mae_scores) if mae_scores else 0.0
        avg_rmse = sum(rmse_scores) / len(rmse_scores) if rmse_scores else 0.0
        avg_r2 = sum(r2_scores) / len(r2_scores) if r2_scores else 0.0
        self.regression_expected_model_metrics = (avg_mae, avg_rmse, avg_r2)

        # Final model trained on all available data
        X_all = self.data[self.available_features]
        y_all = self.data['Future_close_ratio']

        self.regression_model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
        )
        self.regression_model.fit(X_all, y_all)

    def predict_regression(self) -> float:
        """
        Returns the predicted future_close / current_close ratio for the most
        recent candle using the trained regression model.

        Returns
        =======
        float — predicted price ratio (e.g. 1.05 means +5% expected move)
        """
        prediction = float(
            self.regression_model.predict(
                self.data_withna[self.available_features].iloc[[-1]]
            )[0]
        )
        return prediction

    def get_regression_model_metrics(self) -> tuple[float, float, float]:
        """Returns (MAE, RMSE, R²) averaged over walk-forward splits."""
        return self.regression_expected_model_metrics

    def print_regression_model_metrics(self) -> None:
        mae, rmse, r2 = self.regression_expected_model_metrics
        print(f"Regression Model Metrics: MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")





