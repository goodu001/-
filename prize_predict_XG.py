import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime
import numpy as np

df = pd.read_csv('prize_results.csv')
df.columns = ['date','prize','front3_1','front3_2','last3_1','last3_2','last2']
df['date'] = pd.to_datetime(df['date'])
df['date_ordinal'] = df['date'].map(datetime.toordinal)
df = df.sort_values('date')

for i in range(1, 4):
    df[f'prize_lag{i}'] = df['prize'].shift(i)
df = df.dropna()

X = df[['date_ordinal','prize_lag1','prize_lag2','prize_lag3']]
y = df['prize']

X_train, y_train = X.iloc[:-1], y.iloc[:-1]

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# เตรียม feature สำหรับงวดที่จะออก (1 พ.ย.)
next_date = df['date'].max() + pd.Timedelta(days=16)
next_ordinal = next_date.toordinal()
next_row = df.iloc[-1]
next_features = np.array([
    next_ordinal,
    next_row['prize'],
    next_row['prize_lag1'],
    next_row['prize_lag2']
]).reshape(1,-1)

predicted_prize = model.predict(next_features)
print(int(round(predicted_prize[0])))
