from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import trim, split, collect_list, array_distinct, expr

# Step 1: Initialize Spark Session with more memory
# สร้าง Spark Session พร้อมการกำหนดหน่วยความจำเพิ่มเติมสำหรับการประมวลผลข้อมูลขนาดใหญ่
spark = SparkSession.builder \
    .appName("FPGrowth Example") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.task.maxFailures", "5") \
    .getOrCreate()

# Step 2: Read CSV data
# โหลดข้อมูลจากไฟล์ CSV โดยกำหนดให้ inferSchema เป็น True เพื่อให้ Spark ตรวจสอบประเภทข้อมูลอัตโนมัติ
data = spark.read.csv("groceries_data.csv", header=True, inferSchema=True)

# Step 3: Use trim to clean any leading/trailing spaces in 'itemDescription'
# Also replace 'rolls/buns' with 'rolls,buns'
# ใช้ฟังก์ชัน split ในการแยกข้อมูลที่มี '/' ในคอลัมน์ 'itemDescription' เพื่อให้แยกเป็นรายการเดี่ยว ๆ
data = data.withColumn("itemDescription", split(data["itemDescription"], "/"))

# Step 4: Split 'rolls,buns' into separate items
# ใช้ฟังก์ชัน split แยกคำใน 'itemDescription' โดยใช้ ',' เป็นตัวแบ่ง
# data = data.withColumn("itemDescription", split("itemDescription", ","))

# Step 5: Group by 'Member_number' and collect unique items into a basket
# รวมรายการซื้อของสมาชิกแต่ละคนไว้ใน 'basket' โดยใช้ collect_list และ array_distinct เพื่อให้รายการไม่ซ้ำกันในแต่ละตะกร้า
grouped_data = data.groupBy("Member_number").agg(collect_list("itemDescription").alias("basket"))
grouped_data = grouped_data.withColumn("basket", array_distinct("basket"))

# Step 6: Initialize and fit FPGrowth model
# สร้างและฝึกฝนโมเดล FPGrowth โดยกำหนด minSupport (การสนับสนุนต่ำสุด) และ minConfidence (ความเชื่อมั่นต่ำสุด) สำหรับกฎการเชื่อมโยง
fp = FPGrowth(minSupport=0.01, minConfidence=0.3, itemsCol="basket", predictionCol="prediction")
model = fp.fit(grouped_data)

# Step 7: Show frequent itemsets
# แสดงชุดไอเทมที่พบว่ามีความถี่สูงในการซื้อร่วมกัน
model.freqItemsets.show(10, truncate=False)

# Step 8: Show association rules
# แสดงกฎการเชื่อมโยงที่มีค่าความเชื่อมั่นสูงกว่า 0.5 (เช่น เมื่อซื้อสินค้าหนึ่งมักจะซื้ออีกสินค้าร่วมด้วย)
model.associationRules.filter(model.associationRules.confidence > 0.5).show(truncate=False)

# Step 9: Create new data for predictions
# สร้างข้อมูลตัวอย่างใหม่สำหรับการทำนายเพื่อดูว่าโมเดลจะแนะนำสินค้าชิ้นใด
new_data = spark.createDataFrame([
    (["vegetable juice", "frozen fruits", "packaged fruit"],),
    (["mayonnaise", "butter", "rolls"],)  # Separate 'rolls' from 'buns'
], ["basket"])

# Step 10: Transform the model to make predictions
# ทำการทำนายโดยใช้ข้อมูลตัวอย่างใหม่เพื่อแนะนำสินค้าที่น่าจะซื้อร่วมกัน
predictions = model.transform(new_data)
predictions.show(truncate=False)

# Stop Spark session
# ปิด Spark Session หลังจากจบการทำงาน
spark.stop()
