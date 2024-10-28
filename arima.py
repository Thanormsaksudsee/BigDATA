# นำเข้าไลบรารีที่จำเป็นสำหรับการวิเคราะห์ข้อมูล, การสร้างกราฟ และการสร้างโมเดล ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # Updated import statement
from statsmodels.tsa.stattools import adfuller

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('year_sales.csv')

# แสดงข้อมูล 5 แถวแรกเพื่อตรวจสอบโครงสร้างของข้อมูล
print(df.head())

# แปลงคอลัมน์ 'Year' เป็น datetime format โดยใช้รูปแบบปีและเดือน (เช่น '2023-01')
df['Year'] = pd.to_datetime(df['Year'], format='%Y-%m')

# ตั้งค่า 'Year' เป็นดัชนี (index) ของ DataFrame เพื่อให้ง่ายต่อการทำงานกับข้อมูลเวลา
df.set_index('Year', inplace=True)

# วาดกราฟแสดงข้อมูล Time Series ของยอดขายในแต่ละปี
plt.figure(figsize=(12, 6))
plt.plot(df, label='Original Data', color='blue', linestyle='-', linewidth=2)
plt.title('Yearly Sales')        # กำหนดหัวข้อกราฟ
plt.xlabel('Year')               # กำหนดชื่อแกน x
plt.ylabel('Sales')              # กำหนดชื่อแกน y
plt.legend()
# plt.show()  # หากต้องการแสดงกราฟ ให้เอาคอมเมนต์ออก

# ทดสอบความนิ่งของข้อมูล (stationarity) โดยใช้ ADF test
result = adfuller(df['Sales'])
print('ADF Statistic:', result[0])  # แสดงค่าสถิติ ADF
print('p-value:', result[1])       # แสดงค่า p-value เพื่อตัดสินใจว่าข้อมูลนิ่งหรือไม่

# แบ่งข้อมูลออกเป็นชุดฝึกสอน (train) 80% และชุดทดสอบ (test) 20%
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# กำหนดและปรับแต่งโมเดล ARIMA โดยใช้ค่า order = (1,1,1) (สามารถปรับค่า order ตามความเหมาะสม)
model = ARIMA(train, order=(1,1,1))  # (p,d,q): p = AR, d = differencing, q = MA
model_fit = model.fit()

# แสดงสรุปผลการฝึกสอนโมเดล ARIMA
print(model_fit.summary())

# สร้างการทำนายสำหรับช่วงข้อมูล test
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
predictions = pd.DataFrame(predictions, index=test.index, columns=['Predicted'])  # กำหนด DataFrame สำหรับการทำนาย

# วาดกราฟแสดงข้อมูลชุดฝึกสอน (train), ชุดทดสอบ (test) และค่าที่ทำนาย (predicted)
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train', color='green', linestyle='--', linewidth=2)   # ข้อมูลชุดฝึกสอน
plt.plot(test, label='Test', color='orange', linestyle='-.', linewidth=2)    # ข้อมูลชุดทดสอบ
plt.plot(predictions, label='Predicted', color='red', linestyle='-', linewidth=2)  # ข้อมูลที่ทำนาย
plt.title('Train, Test, and Predicted Data')   # หัวข้อกราฟ
plt.xlabel('Year')                             # ชื่อแกน x
plt.ylabel('Sales')                            # ชื่อแกน y
plt.legend()
plt.show()   # แสดงกราฟเปรียบเทียบข้อมูลจริงกับข้อมูลที่ทำนาย
