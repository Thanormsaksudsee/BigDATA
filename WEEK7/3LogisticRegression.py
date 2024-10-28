# 1. Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
# สร้าง SparkSession เพื่อทำงานกับ PySpark
spark = SparkSession.builder \
    .appName("LogisticRegressionExample") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV โดยมี header และให้ PySpark ทำการกำหนด schema ให้อัตโนมัติ
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# แปลงข้อมูลในคอลัมน์ 'status_type' เป็นตัวเลขโดยใช้ StringIndexer
status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

# ใช้ StringIndexer ที่สร้างขึ้นกับข้อมูลจริง โดยแปลง 'status_type' และ 'status_published' ให้เป็นข้อมูลเชิงตัวเลข
data_indexed = status_type_indexer.fit(data).transform(data)
data_indexed = status_published_indexer.fit(data_indexed).transform(data_indexed)

# สร้าง VectorAssembler เพื่อรวมคอลัมน์ 'status_type_ind' และ 'status_published_ind' เป็นฟีเจอร์เดียวที่เรียกว่า 'features'
assembler = VectorAssembler(inputCols=["status_type_ind", "status_published_ind"], outputCol="features")

# กำหนด Logistic Regression โมเดลโดยใช้ 'features' เป็น input และ 'status_type_ind' เป็น label
lr = LogisticRegression(featuresCol="features", labelCol="status_type_ind")
lr.setMaxIter(100).setRegParam(0.1).setElasticNetParam(0.5)  # ตั้งค่าการทำซ้ำ (iterations), ค่าปรับค่า (regularization), และ ElasticNet

# สร้าง Pipeline ที่รวมขั้นตอน assembler และ logistic regression เข้าไว้ด้วยกัน
pipeline = Pipeline(stages=[assembler, lr])

# แบ่งข้อมูลออกเป็นชุดฝึกสอน (train) 80% และชุดทดสอบ (test) 20%
train_data, test_data = data_indexed.randomSplit([0.8, 0.2])

# ฝึก Pipeline โมเดลด้วยชุดข้อมูลฝึกสอน
pipeline_model = pipeline.fit(train_data)

# ใช้โมเดลที่ฝึกแล้วทำการทำนายผลลัพธ์บนชุดข้อมูลทดสอบ
predictions = pipeline_model.transform(test_data)

# แสดงตัวอย่างผลลัพธ์ โดยจะแสดงค่า status_type_ind, prediction, และ probability
predictions.select("status_type_ind", "prediction", "probability").show(5)

# สร้าง evaluator เพื่อประเมินผลลัพธ์ของโมเดลด้วยการคำนวณค่า accuracy, precision, recall และ F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="status_type_ind", predictionCol="prediction")

# คำนวณค่า accuracy ของโมเดล
accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
print(f"Accuracy: {accuracy}")

# คำนวณค่า precision ของโมเดล (เฉลี่ยแบบถ่วงน้ำหนัก)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
print(f"Precision: {precision}")

# คำนวณค่า recall ของโมเดล (เฉลี่ยแบบถ่วงน้ำหนัก)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
print(f"Recall: {recall}")

# คำนวณค่า F1 ของโมเดล
f1 = evaluator.setMetricName("f1").evaluate(predictions)
print(f"F1 measure: {f1}")

# ปิด SparkSession หลังจากใช้งานเสร็จ
spark.stop()
