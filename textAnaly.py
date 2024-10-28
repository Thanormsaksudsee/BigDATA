from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# สร้าง SparkSession สำหรับการวิเคราะห์ข้อมูล
spark = SparkSession.builder.appName("TextAnalytics").getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV (แทนที่ 'reviews_rated.csv' ด้วยไฟล์ที่ต้องการ)
data = spark.read.csv("reviews_rated.csv", header=True, inferSchema=True)

# เลือกเฉพาะคอลัมน์ "Review Text" และ "Rating" แล้วเปลี่ยนชื่อคอลัมน์ "Review Text" เป็น "review_text" และแปลงประเภทข้อมูล "Rating" เป็น IntegerType
data = data.select(data["Review Text"].alias("review_text"), data["Rating"].cast(IntegerType()).alias("rating"))

# ลบแถวที่มีค่า missing
data = data.na.drop()
data.show(5)

# ขั้นตอนที่ 1: Tokenizer แยกคำในรีวิวออกเป็นคำ (tokens)
tokenizer = Tokenizer(inputCol="review_text", outputCol="words")

# ขั้นตอนที่ 2: StopWordsRemover ลบคำที่ใช้บ่อยแต่ไม่มีความหมายสำคัญ (เช่น "the", "is")
stopword_remover = StopWordsRemover(inputCol="words", outputCol="meaningful_words")

# ขั้นตอนที่ 3: HashingTF แปลงคำที่มีความหมายให้เป็นตัวเลข (features) โดยใช้ Term Frequency
hashing_tf = HashingTF(inputCol="meaningful_words", outputCol="features")

# สร้าง Pipeline เพื่อเรียกใช้ขั้นตอนการประมวลผลที่กำหนดไว้ตามลำดับ
pipeline = Pipeline(stages=[tokenizer, stopword_remover, hashing_tf])

# แบ่งข้อมูลออกเป็นชุดข้อมูลฝึก (80%) และชุดข้อมูลทดสอบ (20%)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# เรียกใช้ Pipeline กับชุดข้อมูลฝึก
pipeline_model = pipeline.fit(train_data)

# แปลงข้อมูลฝึกและข้อมูลทดสอบด้วย pipeline ที่ได้สร้างไว้
train_transformed = pipeline_model.transform(train_data)
test_transformed = pipeline_model.transform(test_data)

# แสดงตัวอย่างข้อมูลที่แปลงแล้วของชุดข้อมูลฝึก
train_transformed.select("meaningful_words", "features", "rating").show(5)

# สร้างโมเดล Logistic Regression สำหรับการจำแนกประเภทโดยใช้คอลัมน์ 'features' เป็นตัวทำนายและ 'rating' เป็นตัว label
log_reg = LogisticRegression(labelCol="rating", featuresCol="features")

# ฝึกโมเดล Logistic Regression ด้วยชุดข้อมูลฝึกที่แปลงแล้ว
log_reg_model = log_reg.fit(train_transformed)

# ทำนายผลลัพธ์ด้วยชุดข้อมูลทดสอบที่แปลงแล้ว
predictions = log_reg_model.transform(test_transformed)

# แสดงผลการทำนาย
predictions.select("meaningful_words", "rating", "prediction").show(5)

# ประเมินโมเดลโดยใช้ MulticlassClassificationEvaluator เพื่อคำนวณค่าความแม่นยำ
evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")

# คำนวณความแม่นยำของโมเดล
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
