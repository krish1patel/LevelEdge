from leveledge import Predictor
from zoneinfo import ZoneInfo
from datetime import datetime

ticker = input('Enter the ticker you wish to predictor for(e.g. SPY, ETH-USD): ').strip()
tgt_datetime_str = input('Enter the datetime you wish to predict for in EST (YYYY-MM-DD HH:MM:SS): ')
intvl = input('Enter the interval you would like to analyze (e.g. 5m, 15m, 1h, 4h, 1d): ')
price_level = float(input('Enter the price level you wish to predict for: $'))

format_string = "%Y-%m-%d %H:%M:%S"
tgt_datetime = datetime.strptime(tgt_datetime_str, format_string).replace(tzinfo=ZoneInfo('EST'))
predictor = Predictor(ticker, tgt_datetime, intvl, price_level)

predictor.train_xgb()

prediction = predictor.predict_xgb()

print('\n\n')
print('='*30)
print('Model Metrics')
print('='*30)
print()
predictor.print_xgb_model_metrics()


print('\n\n')
print('='*30)
print('Prediction Results')
print('='*30)
print()
print(f'{ticker.upper()} has a {prediction:.2%} chance of being above {price_level} at {tgt_datetime_str}')
