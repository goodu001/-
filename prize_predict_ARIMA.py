import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# โหลดข้อมูล
df = pd.read_csv('prize_results.csv')
df.columns = ['date','prize','front3_1','front3_2','last3_1','last3_2','last2']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ตั้ง index เป็นวันที่
df = df.set_index('date')

# รายการรางวัลที่จะทำนาย
target_cols = ['prize','front3_1','front3_2','last3_1','last3_2','last2']

results = {}

# ทำนายแต่ละรางวัลด้วย SARIMA
for target in target_cols:
    series = df[target]
    
    # ใช้ parameter SARIMA เบื้องต้น (p,d,q,P,D,Q,s)
    # กำหนด s=2 เพราะหวยออกเดือนละ 2 ครั้ง (1 และ 16)
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,2), enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = model.fit(disp=False)
    
    # ทำนายค่าถัดไป 1 งวด (15 วันข้างหน้า)
    forecast = sarima_fit.forecast(steps=1)
    
    # เก็บผลลัพธ์
    results[target] = int(round(forecast.iloc[0]))

# วันที่ถัดไป (งวด 1 พ.ย.)
next_date = df.index.max() + timedelta(days=15)

print(f"🔮 Predicted Lottery Results for {next_date.date()} (Statistical Forecast)\n")
for k,v in results.items():
    print(f"{k:12s}: {v}")