# 1. Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# สร้าง SparkSession สำหรับการทำงานกับ PySpark
spark = SparkSession.builder \
    .appName("DecisionTreeRegressionExample") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV โดยกำหนดให้มี header และให้ PySpark กำหนด schema ให้อัตโนมัติ
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# แปลงข้อมูลคอลัมน์ 'num_reactions' เป็นดัชนีตัวเลขแทนข้อความ (StringIndexer)
indexer_reactions = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")
indexer_loves = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

# แปลงข้อมูลดัชนี (index) เป็นเวกเตอร์แบบ One-Hot Encoding เพื่อให้เป็นข้อมูลเชิงตัวเลขสำหรับการวิเคราะห์
encoder_reactions = OneHotEncoder(inputCols=["num_reactions_ind"], outputCols=["num_reactions_vec"])
encoder_loves = OneHotEncoder(inputCols=["num_loves_ind"], outputCols=["num_loves_vec"])

# สร้าง VectorAssembler เพื่อรวมคอลัมน์เวกเตอร์ของ 'num_reactions_vec' และ 'num_loves_vec' เป็นฟีเจอร์เดียว
assembler = VectorAssembler(inputCols=["num_reactions_vec", "num_loves_vec"], outputCol="features")

# สร้าง Pipeline ที่ประกอบไปด้วยขั้นตอนการแปลงข้อมูลทั้งหมด
pipeline = Pipeline(stages=[indexer_reactions, indexer_loves, encoder_reactions, encoder_loves, assembler])

# ฝึก Pipeline โมเดลด้วยข้อมูลที่มีอยู่
pipeline_model = pipeline.fit(data)

# แปลงข้อมูลด้วย pipeline ที่ฝึกมาแล้ว เพื่อเตรียมข้อมูลสำหรับการทำนาย
transformed_data = pipeline_model.transform(data)

# แบ่งข้อมูลออกเป็น train (80%) และ test (20%) สำหรับการทดสอบ
train_data, test_data = transformed_data.randomSplit([0.8, 0.2])

# สร้างโมเดล Decision Tree Regressor โดยใช้ 'num_loves_ind' เป็น label และ 'features' เป็น input
dt = DecisionTreeRegressor(labelCol="num_loves_ind", featuresCol="features")

# ฝึก Decision Tree โมเดลด้วยข้อมูล train
dt_model = dt.fit(train_data)

# ทำนายผลลัพธ์จากโมเดลโดยใช้ข้อมูลทดสอบ (test data)
predictions = dt_model.transform(test_data)

# สร้าง evaluator เพื่อวัดผลลัพธ์ของโมเดล โดยเลือกใช้ R2 Score ในการประเมิน
evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

# คำนวณค่า R2 Score ของโมเดล R2  = ค่าสัมประสิทธิ์การกำหนด
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2 score: {r2}")

# ปิด SparkSession หลังใช้งานเสร็จสิ้น
spark.stop()
