from pyspark.sql import SparkSession
from pyspark.sql.functions import split, current_timestamp
import pyspark.sql.functions as F

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()

# สร้าง DataFrame ที่แทนสตรีมข้อมูลจาก socket
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# แยกข้อความและเพิ่มคอลัมน์ timestamp
words = lines.withColumn("date", \
    split(lines["value"], " ").getItem(1)). \
    withColumn("timestamp", F.current_timestamp()). \
    withWatermark("timestamp", "10 seconds")

# การจัดกลุ่มและนับจำนวนคำในช่วงเวลา
windowedCounts = words.groupBy(
    F.window(words["timestamp"], "10 seconds", "5 seconds"),
    words["date"]
).count()

# เขียนผลลัพธ์ออกมาที่คอนโซล
query = windowedCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# รอให้การประมวลผลสตรีมมิ่งเสร็จสิ้น
query.awaitTermination()
