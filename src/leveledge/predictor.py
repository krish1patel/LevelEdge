import math
from warnings import deprecated
from sklearn.utils.extmath import weighted_mode
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score
from zoneinfo import ZoneInfo

ALLOWED_INTERVALS: list[str] = ["1m", "2m", "5m", "10m", "15m", "30m", "1h", "90m", "1d"]


class Predictor:

    def __init__(self, ticker_str: str, target_datetime: datetime, interval: str, price: float) -> None:
        """
        Constructor creates object with a ticker, target datetime for prediction time
        and an interval to use to fetch data

        Parameters
        ==========
        ticker_str: str - lower or upper that holds the ticker symbol for the stock or crypto of interest
        target_datetime: datetime - datetime that represents when to predict for
        interval: str - string format of the interval to use for stock/crypto data

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

        if self.interval not in ALLOWED_INTERVALS:
            raise ValueError("Invalid interval input.")
        if self.target_datetime < datetime.now(tz=ZoneInfo('EST')):
            raise ValueError("Target Datetime Must be in the future.")

        if 'm' in self.interval:
            self.interval_min: int = int(self.interval[:-1])
        elif 'h' in self.interval:
            self.interval_min: int = 60 * int(self.interval[:-1])
        elif 'd' in self.interval:
            self.interval_min: int = 60 * 24 * int(self.interval[:-1])

        self.data: pd.DataFrame = self.ticker.history(period='max', interval=self.interval)
        self.current_price: float = self.data['Close'].iloc[-1]
        self.target_price_ratio: float = self.price / self.current_price
        self.candles_ahead: int = self._calculate_candles_ahead_crypto() if self.isCrypto else self._calculate_candles_ahead_stocks()
        self.features: pd.DataFrame = self.prepare_features()

        self._create_target_variable()

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

        market_days: int = 0
        non_market_days: int = 0

        for num_days in range(1, delta.days+1, 1):
            date: pd.Timestamp = pd.Timestamp(current_datetime + timedelta(days=num_days)).normalize()
            schedule = nyse.schedule(start_date=date, end_date=date)
            if schedule.empty:
                # Not a trading day
                non_market_days += 1
            else:
                market_days += 1

        if self.interval_min == 1440: # interval is 1d
            return market_days
        
        hours_to_subtract: float = market_days*17.5 + non_market_days*24

        market_hours: float = (delta - timedelta(hours=hours_to_subtract, minutes=1)).total_seconds() / 60 / 60

        candles: int = int(market_hours / (self.interval_min/60))

        return candles

    def _calculate_candles_ahead_crypto(self) -> int:
        """Calculates how may candles ahead the target datetime is from the current candle for cryptos (24hr)."""

        current_datetime = self.data.index[-1]
        delta: timedelta = self.target_datetime - current_datetime - timedelta(minutes=1)

        return int(delta.total_seconds / 60 / self.interval_min)
        
    @deprecated('Logical Errors')
    def _calculate_candles_ahead(self) -> int:
        """Calculates how many candles ahead the target datetime is from current candle.

        For daily intervals ('1d') this uses business-day counting (US Federal holidays)
        so weekends and federal holidays are skipped. For intraday intervals it
        converts the time delta to minutes and estimates the number of candles,
        subtracting the usual market-closed hours for non-crypto tickers.
        """

        last_candle_timestamp: pd.Timestamp = self.data.index[-1]
        target_timestamp: pd.Timestamp = pd.Timestamp(self.target_datetime)

        # Normalize to same tz-aware timestamps where possible
        # Ensure timestamps are timezone-aware in US/Eastern
        last_ts = pd.Timestamp(last_candle_timestamp)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('US/Eastern')
        else:
            last_ts = last_ts.tz_convert('US/Eastern')

        target_ts = pd.Timestamp(target_timestamp)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize('US/Eastern')
        else:
            target_ts = target_ts.tz_convert('US/Eastern')

        # If daily interval, count business days between last candle date and target date
        if self.interval.endswith('d'):
            # Use pandas CustomBusinessDay with US Federal holidays
            from pandas.tseries.holiday import USFederalHolidayCalendar
            from pandas.tseries.offsets import CustomBusinessDay

            cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

            # Start from the day after the last candle's date up to and including target date
            start_date = last_ts.date()
            end_date = target_ts.date()

            if end_date <= start_date:
                return 0

            # Build date range using business day frequency and count
            bday_range = pd.date_range(start=start_date + pd.Timedelta(days=1), end=end_date, freq=cbd)
            # number of business days ahead
            business_days_ahead = len(bday_range)

            # Subtract 1 to predict the candle before the target time (preserve original behavior)
            candles_ahead = max(0, business_days_ahead - 1)
            return int(candles_ahead)

        # Otherwise (intraday), compute minutes difference safely
        delta_seconds = (target_ts - last_ts).total_seconds()
        delta_minutes = delta_seconds / 60.0

        # Subtract 1 to ensure predicting for candle before target
        if not self.isCrypto:
            # Approximate non-trading hours removal: subtract 17.5 hours (market closed time)
            # This remains an approximation for intraday intervals; can be improved later.
            closed_minutes = 17.5 * 60
            candles_ahead = int((delta_minutes - 1 - closed_minutes) / self.interval_min)
        else:
            candles_ahead = int((delta_minutes - 1) / self.interval_min)

        return max(0, int(candles_ahead))

    def _calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate various technical indicators"""
        data = self.data.copy()
        
        # Simple Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
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
    
    def prepare_features(self) -> pd.DataFrame:
        """Creates and adds features to data, then prepares feature set for the model"""
        
        # Calculate all indicators
        self._calculate_technical_indicators()
        self._calculate_candlestick_patterns()
        
        # Define feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            'ATR',
            'Volume_SMA', 'Volume_ratio',
            'Price_change', 'Price_change_5', 'High_Low_ratio',
            'Body', 'Upper_Shadow', 'Lower_Shadow', 'Total_Range',
            'Body_Ratio', 'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
            'Candle_Type', 'Doji', 'Hammer', 'Shooting_Star',
            'Bullish_Engulfing', 'Bearish_Engulfing'
        ]
        
        # Select available features and drop NaN rows
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




    def train_xgb(self) -> None:
        splits = self._walk_forward_split(400, 80, 100, 0)
        auc_scores = []
        ps_scores = []
        pr_scores = []
        
        # Add diagnostic info
        # print(f"\nOverall class distribution:")
        # print(self.data['Target'].value_counts())
        # print(f"Overall positive rate: {self.data['Target'].mean():.2%}\n")

        for i, (train_idx, test_idx) in enumerate(splits):
            train = self.data.iloc[train_idx]
            test = self.data.iloc[test_idx]

            X_train = train[self.available_features]
            y_train = train['Target']
            X_test = test[self.available_features]
            y_test = test['Target']
            
            # Diagnostic prints
            # print(f"Split {i+1}:")
            # print(f"  Train: {len(y_train)} samples, {y_train.sum()} positive ({y_train.mean():.2%})")
            # print(f"  Test:  {len(y_test)} samples, {y_test.sum()} positive ({y_test.mean():.2%})")
            
            # Skip invalid splits
            if len(y_test.unique()) < 2:
                # print(f"  ⚠️  SKIPPED - only class {y_test.unique()[0]} in test set\n")
                continue
            
            if len(y_train.unique()) < 2:
                # print(f"  ⚠️  SKIPPED - only class {y_train.unique()[0]} in train set\n")
                continue

            # Calculate class weight
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()
            scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1
            
            # print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

            model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight  # Add this!
            )

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            preds_binary = (preds >= .6).astype(int)
            auc = roc_auc_score(y_test, preds)
            pr = average_precision_score(y_test, preds)
            ps = precision_score(y_test, preds_binary)
            auc_scores.append(auc)
            ps_scores.append(ps)
            pr_scores.append(pr)
            # print(f"  AUC: {auc:.4f}    PS: {ps:.4f}\n")

        # Model metrics are the mean of the scores for all models
        avg_auc = sum(auc_scores) / len(auc_scores) if len(auc_scores) > 0 else 0
        avg_ps = sum(ps_scores) / len(ps_scores) if len(ps_scores) > 0 else 0
        avg_pr = sum(pr_scores) / len(pr_scores) if len(pr_scores) > 0 else 0
        self.xgb_expected_model_metrics = (avg_auc, avg_ps, avg_pr)

        # Train model on all data
        X_train = self.data[self.available_features]
        y_train = self.data['Target']

        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1

        self.xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight  # Add this!
        )

        self.xgb_model.fit(X_train, y_train)
        # self.xgb_model = model


        # print(f"\n{'='*50}")
        # print(f"Valid AUC, PS scores: {auc_scores}, {ps_scores}")
        # if auc_scores:
            # print(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
            # print(f"Mean PS: {np.mean(ps_scores):.4f} ± {np.std(ps_scores):.4f}")
        # print(f"{'='*50}")

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


        prediction = self.xgb_model.predict_proba(self.data_withna[self.available_features])[-1][1]
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







