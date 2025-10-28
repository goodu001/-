import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import numpy as np

# โหลดข้อมูล
df = pd.read_csv('prize_results.csv')
df.columns = ['date','prize','front3_1','front3_2','last3_1','last3_2','last2']
df['date'] = pd.to_datetime(df['date'])
df['date_ordinal'] = df['date'].map(datetime.toordinal)
df = df.sort_values('date')

# สร้าง lagged features สำหรับแต่ละรางวัล
lag_cols = ['prize','front3_1','front3_2','last3_1','last3_2','last2']
for col in lag_cols:
    for i in range(1, 4):
        df[f'{col}_lag{i}'] = df[col].shift(i)
df = df.dropna()

# features ที่ใช้
feature_cols = ['date_ordinal']
for col in lag_cols:
    feature_cols += [f'{col}_lag1', f'{col}_lag2', f'{col}_lag3']

results = {}

# ทำนายแต่ละรางวัล
for target in lag_cols:
    X = df[feature_cols]
    y = df[target]
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    X_next = X.iloc[-1:]
    pred = model.predict(X_next)
    results[target] = int(round(pred[0]))

print(results)
