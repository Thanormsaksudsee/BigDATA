from pyspark.sql import SparkSession
from pyspark.sql.functions import split, current_timestamp
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()

lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

words = lines.withColumn("date", \
    split(lines["value"], " ").getItem(1)). \
    withColumn("timestamp", F.current_timestamp()). \
    withWatermark("timestamp", "10 seconds")

windowedCounts = words.groupBy(
    F.window(words["timestamp"], "10 seconds", "5 seconds"),
    words["date"]
).count()

query = windowedCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
