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


words = lines.withColumn("word", split(lines["value"], " ")) \
    .withColumn("word", F.explode(F.col("word"))) \
    .withColumn("timestamp", F.current_timestamp()) \
    .withWatermark("timestamp", "10 seconds")
    

windowedCounts = words.groupBy(
    F.window(words["timestamp"], "10 seconds", "5 seconds"),
    words["word"]

).count()

query = windowedCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", False) \
    .start()
    
query.awaitTermination()


