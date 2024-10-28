# นำเข้าไลบรารีที่จำเป็นสำหรับการสร้าง Spark session, ALS และการประเมินผล
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# สร้าง Spark session สำหรับการทำงานกับ PySpark
spark = SparkSession.builder.appName("BookRecommendations").getOrCreate()

# โหลดชุดข้อมูลจากไฟล์ CSV โดยให้ PySpark ทำการกำหนด schema ให้อัตโนมัติ
df = spark.read.csv('book_ratings.csv', header=True, inferSchema=True)

# แสดง schema ของข้อมูลเพื่อยืนยันโครงสร้างว่าโหลดข้อมูลถูกต้อง
df.printSchema()

# กำหนดโมเดล ALS (Alternating Least Squares) สำหรับการสร้างระบบแนะนำหนังสือ
als = ALS(
    maxIter=10,                 # จำนวน iteration สูงสุดที่ ALS จะรันเพื่อหาผลลัพธ์
    regParam=0.1,               # ค่าพารามิเตอร์การปรับให้เหมาะสมเพื่อลด overfitting
    userCol="user_id",           # คอลัมน์ที่ใช้แทนผู้ใช้
    itemCol="book_id",           # คอลัมน์ที่ใช้แทนไอเท็ม (ในที่นี้คือหนังสือ)
    ratingCol="rating",         # คอลัมน์ที่ใช้แทนค่าคะแนนที่ให้โดยผู้ใช้
    coldStartStrategy="drop"    # วิธีการจัดการกับค่า NaN ในการทำนาย (ใช้ "drop" เพื่อละทิ้งค่า NaN)
)

# แบ่งข้อมูลออกเป็นชุดข้อมูลฝึกสอน (training) 80% และชุดข้อมูลทดสอบ (test) 20%
(training_data, test_data) = df.randomSplit([0.8, 0.2])

# ฝึกโมเดล ALS บนชุดข้อมูลฝึกสอน
model = als.fit(training_data)

# สร้างการทำนายผลลัพธ์บนชุดข้อมูลทดสอบ
predictions = model.transform(test_data)

# กำหนดตัวประเมินผลโดยใช้ RMSE (Root Mean Square Error) เป็น metric
evaluator = RegressionEvaluator(
    metricName="rmse",          # ใช้ RMSE เป็น metric ในการประเมินผล
    labelCol="rating",          # คอลัมน์ของค่าคะแนนจริง
    predictionCol="prediction"  # คอลัมน์ของค่าคะแนนที่โมเดลทำนาย
)

# ประเมินผลโมเดลโดยคำนวณ RMSE ซึ่งแสดงถึงความคลาดเคลื่อนของการทำนาย
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error (RMSE): {rmse}")

# กรองข้อมูลเพื่อแสดงข้อมูล User ID, Book ID, Rating และ Prediction สำหรับผู้ใช้ที่ระบุ (ตัวอย่างเช่น User ID = 53)
user_id = 53
filtered_user_data = df.filter(df['user_id'] == user_id)

# แสดงข้อมูลการให้คะแนนจริงของผู้ใช้ที่เลือก
filtered_user_data.show()

# แสดงการทำนายผลลัพธ์สำหรับผู้ใช้ที่เลือก
user_predictions = model.transform(filtered_user_data)
user_predictions.orderBy('prediction', ascending=False).show()

# แสดงหนังสือที่แนะนำ 5 เล่มสำหรับผู้ใช้ทุกคน
model.recommendForAllUsers(5).show(truncate=False)

# แสดงผู้ใช้ 5 คนที่แนะนำให้หนังสือทุกเล่ม
model.recommendForAllItems(5).show(truncate=False)

# ปิด Spark session หลังจากทำงานเสร็จสิ้น
spark.stop()
