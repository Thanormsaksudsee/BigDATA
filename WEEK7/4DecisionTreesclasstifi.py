from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# สร้าง SparkSession เพื่อทำงานกับ PySpark
spark = SparkSession.builder.appName("FBLiveTH").getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV โดยกำหนดให้มี header และ inferSchema เพื่อให้ PySpark กำหนด schema ให้อัตโนมัติ
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# ใช้ StringIndexer เพื่อแปลงข้อมูลในคอลัมน์ 'status_type' และ 'status_published' เป็นข้อมูลเชิงตัวเลข
status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

# ใช้ OneHotEncoder เพื่อแปลงข้อมูลที่แปลงเป็นตัวเลขแล้วให้เป็นการเข้ารหัสแบบ one-hot
status_type_encoder = OneHotEncoder(inputCol="status_type_ind", outputCol="status_type_encoded")
status_published_encoder = OneHotEncoder(inputCol="status_published_ind", outputCol="status_published_encoded")

# รวมคอลัมน์ที่เข้ารหัสแล้วลงในคอลัมน์เดียว 'features' สำหรับใช้ในโมเดล
assembler = VectorAssembler(inputCols=["status_type_encoded", "status_published_encoded"], outputCol="features")

# สร้าง Pipeline รวมทุกขั้นตอนการเตรียมข้อมูลและการแปลงข้อมูลที่ต้องใช้
pipeline = Pipeline(stages=[status_type_indexer, status_published_indexer, status_type_encoder, status_published_encoder, assembler])

# ฝึก Pipeline model บนชุดข้อมูล
pipeline_model = pipeline.fit(data)

# แปลงข้อมูลด้วย pipeline model เพื่อให้ได้ข้อมูลพร้อมสำหรับฝึกโมเดล
transformed_data = pipeline_model.transform(data)

# แบ่งข้อมูลออกเป็นชุดฝึกสอน (train) 80% และชุดทดสอบ (test) 20%
train_data, test_data = transformed_data.randomSplit([0.8, 0.2])

# สร้าง Decision Tree Classifier โดยกำหนด 'features' เป็น input และ 'status_type_ind' เป็น label
decision_tree = DecisionTreeClassifier(labelCol="status_type_ind", featuresCol="features")

# ฝึก Decision Tree model บนชุดข้อมูลฝึกสอน
decision_tree_model = decision_tree.fit(train_data)

# ทำการทำนายผลลัพธ์บนชุดข้อมูลทดสอบ
predictions = decision_tree_model.transform(test_data)

# ประเมินผลลัพธ์ของโมเดลด้วย metric ต่างๆ เช่น accuracy, precision, recall, และ f1
evaluator = MulticlassClassificationEvaluator(labelCol="status_type_ind", predictionCol="prediction")

# คำนวณ accuracy ของโมเดล
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})

# คำนวณ weighted precision ของโมเดล
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})

# คำนวณ weighted recall ของโมเดล
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

# คำนวณ F1 measure ของโมเดล
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

# คำนวณ test error (ความคลาดเคลื่อนของโมเดล)
test_error = 1.0 - accuracy

# แสดงผลลัพธ์ metric ต่างๆ
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Measure: {f1}")
print(f"Test Error: {test_error}")
