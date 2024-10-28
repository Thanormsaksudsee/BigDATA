# Import necessary libraries for Spark, ML, and plotting
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType

# สร้าง SparkSession สำหรับการใช้งาน PySpark
spark = SparkSession.builder \
    .appName("Linear Regression Analysis") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV และตั้งค่าให้มี header และกำหนด schema ให้อัตโนมัติ
data = spark.read.csv('fb_live_thailand.csv', header=True, inferSchema=True)
data.show()  # แสดงข้อมูลตัวอย่างเพื่อให้เห็นภาพรวมของ dataset

# สร้าง VectorAssembler เพื่อรวมคอลัมน์ 'num_reactions' และ 'num_loves' เป็นคอลัมน์ 'features'
assembler = VectorAssembler(
    inputCols=['num_reactions', 'num_loves'],
    outputCol='features'
)

# แปลงข้อมูลโดยใช้ assembler เพื่อสร้างคอลัมน์ 'features' ที่ใช้ในการวิเคราะห์ regression
data_assembled = assembler.transform(data)
data_assembled.show()  # แสดงข้อมูลที่มีคอลัมน์ 'features'

# สร้างโมเดล Linear Regression โดยกำหนดให้ 'num_loves' เป็น label และ 'features' เป็น input
linear_regression = LinearRegression(
    labelCol='num_loves',        # คอลัมน์ที่เป็น label หรือค่าผลลัพธ์ที่ต้องการพยากรณ์
    featuresCol='features',      # คอลัมน์ที่เป็น input feature
    maxIter=10,                  # กำหนดจำนวนรอบการทำงานสูงสุดของการ train
    regParam=0.3,                # ค่าพารามิเตอร์การปกติ (regularization) เพื่อป้องกันการ overfitting
    elasticNetParam=0.8          # พารามิเตอร์ ElasticNet ใช้ผสมระหว่าง Ridge และ Lasso regression
)

# กำหนด Pipeline ซึ่งช่วยในการเรียงลำดับการทำงานของขั้นตอนการเตรียมและสร้างโมเดล
pipeline = Pipeline(stages=[linear_regression])

# แบ่งข้อมูลออกเป็น train (80%) และ test (20%) สำหรับการเทรนและการทดสอบโมเดล
train_data, test_data = data_assembled.randomSplit([0.8, 0.2], seed=42)

# ฝึกโมเดลด้วยข้อมูล train
pipeline_model = pipeline.fit(train_data)
# ทำนายผลลัพธ์ของข้อมูล test และแสดงคอลัมน์ num_loves, features, และ prediction
predictions = pipeline_model.transform(test_data)
predictions.select('num_loves', 'features', 'prediction').show(5)  # แสดงตัวอย่างผลการพยากรณ์

# สร้าง evaluator สำหรับประเมินโมเดลโดยใช้ Mean Squared Error (MSE) และ R-squared (R2)
evaluator = RegressionEvaluator(
    labelCol='num_loves',         # ค่าที่แท้จริงของ label
    predictionCol='prediction'    # ค่าที่โมเดลพยากรณ์ได้
)

# คำนวณ Mean Squared Error (MSE) ของโมเดล
mse = evaluator.setMetricName("mse").evaluate(predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# คำนวณค่า R-squared (R2) ของโมเดล ซึ่งแสดงประสิทธิภาพของการพยากรณ์
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2: {r2:.4f}")

# แปลงผลลัพธ์การพยากรณ์เป็น Pandas DataFrame เพื่อใช้กับ Matplotlib
pandas_df = predictions.select('num_loves', 'prediction').toPandas()

# สร้างกราฟ scatter plot เพื่อแสดงความสัมพันธ์ระหว่าง num_loves กับ prediction
plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_loves', y='prediction', data=pandas_df)
plt.title('Scatter Plot of num_loves vs Prediction')
plt.xlabel('num_loves')
plt.ylabel('Prediction')
plt.show()

# เลือกคอลัมน์ num_loves และ prediction เปลี่ยนเป็น IntegerType และเรียงข้อมูลตาม prediction ในลำดับมากไปน้อย
selected_data = predictions.select(
    col('num_loves').cast(IntegerType()).alias('num_loves'),
    col('prediction').cast(IntegerType()).alias('prediction')
).orderBy(col('prediction').desc())

# แปลงข้อมูลที่เลือกเป็น Pandas DataFrame เพื่อใช้สำหรับการวิเคราะห์ต่อ
pandas_df = selected_data.toPandas()

# สร้างกราฟ Linear Regression plot ระหว่าง num_loves และ prediction
plt.figure(figsize=(10, 6))
sns.lmplot(x='num_loves', y='prediction', data=pandas_df, aspect=1.5)
plt.title('Linear Regression: num_loves vs Prediction')
plt.xlabel('num_loves')
plt.ylabel('Prediction')
plt.show()
