# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.types import DoubleType

# สร้าง SparkSession สำหรับการใช้งาน PySpark
spark = SparkSession \
    .builder \
    .appName("testKMeans") \
    .getOrCreate()

# อ่านไฟล์ CSV ที่มี header โดยข้อมูลจะถูกเก็บใน DataFrame ชื่อ `df`
df = spark.read.format("csv").option("header", True).load("fb_live_thailand.csv")

# เลือกเฉพาะคอลัมน์ `num_sads` และ `num_reactions` แล้วแปลงข้อมูลเป็นชนิด `Double`
df = df.select(df.num_sads.cast(DoubleType()), df.num_reactions.cast(DoubleType()))

# รวมคอลัมน์ `num_sads` และ `num_reactions` เป็นคอลัมน์ `features` ที่ใช้ในการวิเคราะห์กลุ่ม
vec_assembler = VectorAssembler(inputCols=["num_sads", "num_reactions"], outputCol="features")

# ใช้ StandardScaler เพื่อทำให้ค่าของแต่ละคอลัมน์มีมาตรฐานและเปรียบเทียบกันได้ง่ายขึ้น
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# สร้างลิสต์สำหรับเก็บค่า Silhouette Score ของแต่ละค่า k
k_values = []

# ลูปหาค่า k ที่เหมาะสมในช่วง 2 ถึง 5
for i in range(2, 5):
    # สร้างโมเดล K-means โดยใช้ค่า k=i
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction_col", k=i)
    
    # รวม stages ของการแปลงและการสเกลข้อมูลลงใน Pipeline
    pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])
    
    # ฝึก pipeline model ด้วยข้อมูล `df`
    model = pipeline.fit(df)
    
    # แปลงข้อมูลด้วยโมเดลที่ฝึกแล้วเพื่อให้ได้ prediction
    output = model.transform(df)
    
    # ประเมินโมเดลด้วย Silhouette Score โดยใช้ระยะ Squared Euclidean
    evaluator = ClusteringEvaluator(predictionCol="prediction_col", featuresCol="scaledFeatures", metricName="silhouette", distanceMeasure="squaredEuclidean")
    score = evaluator.evaluate(output)
    
    # บันทึกคะแนน silhouette score ลงในลิสต์ k_values
    k_values.append(score)
    print("Silhouette Score:", score)

# เลือก k ที่ให้ค่า Silhouette Score สูงสุด
best_k = k_values.index(max(k_values)) + 2  # บวก 2 เพราะเริ่มจาก k=2
print("The best k", best_k, max(k_values))

# สร้างโมเดล K-means โดยใช้ k ที่ดีที่สุดที่ได้จากการประเมิน
kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction_col", k=best_k)

# รวม stages ของการแปลงและการสเกลข้อมูลลงใน Pipeline
pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])

# ฝึก pipeline model ด้วยข้อมูล `df`
model = pipeline.fit(df)

# ทำนายผลการจัดกลุ่มของข้อมูล
predictions = model.transform(df)

# ประเมิน Silhouette Score ของโมเดลที่ได้
evaluator = ClusteringEvaluator(predictionCol="prediction_col", featuresCol="scaledFeatures", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance =", silhouette)

# แปลง DataFrame ของผลการทำนายเป็น Pandas DataFrame เพื่อใช้งานกับ Matplotlib
clustered_data_pd = predictions.toPandas()

# สร้างกราฟกระจายเพื่อแสดงผลการจัดกลุ่ม
plt.scatter(clustered_data_pd["num_reactions"], clustered_data_pd["num_sads"], c=clustered_data_pd["prediction_col"])
plt.xlabel("num_reactions")
plt.ylabel("num_sads")
plt.title("K-means Clustering")
plt.colorbar().set_label("Cluster")
plt.show()
