import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# โหลดข้อมูล
df = pd.read_csv('prize_results.csv')
df.columns = ['date','prize','front3_1','front3_2','last3_1','last3_2','last2']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# รายการรางวัลที่ต้องการทำนาย
target_cols = ['prize','front3_1','front3_2','last3_1','last3_2','last2']

results = {}

# ทำนายแต่ละรางวัลด้วย Prophet
for target in target_cols:
    # เตรียมข้อมูลในรูปแบบ Prophet
    prophet_df = df[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    
    # สร้างโมเดล Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(prophet_df)
    
    # ทำนายไปข้างหน้า 15 วัน (งวดถัดไป)
    future = model.make_future_dataframe(periods=15)
    forecast = model.predict(future)
    
    # ค่าทำนายสุดท้าย (วันที่ล่าสุด)
    pred_value = forecast.tail(1)['yhat'].values[0]
    
    results[target] = int(round(pred_value))

# วันที่ถัดไป (งวด 1 พ.ย.)
next_date = df['date'].max() + timedelta(days=15)

print(f"🔮 Predicted Lottery Results for {next_date.date()} (Prophet Model)\n")
for k, v in results.items():
    print(f"{k:12s}: {v}")
