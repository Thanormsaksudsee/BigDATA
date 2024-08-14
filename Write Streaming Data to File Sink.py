from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, StructField
from pyspark.sql.functions import split, current_timestamp

spark = SparkSession.builder.appName("FileSourceStreamingExample").getOrCreate()

file_schema = StructType([
    StructField("status_id", StringType(), True),
    StructField("status_type", StringType(), True),
    StructField("status_published", StringType(), True),
    StructField("num_reactions", StringType(), True),
    StructField("num_comments", StringType(), True),
    StructField("num_shares", StringType(), True),
    StructField("num_likes", StringType(), True),
    StructField("num_loves", StringType(), True),
    StructField("num_wows", StringType(), True),
    StructField("num_hahas", StringType(), True),
    StructField("num_sads", StringType(), True),
    StructField("num_angrys", StringType(), True)
])

lines = spark \
    .readStream \
    .format("csv") \
    .option("maxFilesPerTrigger", 1) \
    .option("header", True) \
    .option("path", "./data/stream") \
    .schema(file_schema) \
    .load()

words = lines.withColumn("date", split(lines["status_published"], " ").getItem(1)) \
             .withColumn("timestamp", current_timestamp()) \
             .withWatermark("timestamp", "10 seconds")

wordCounts = words.groupBy("date", "status_type", "timestamp").count()

wordCounts.writeStream \
    .format("csv") \
    .option("path", "./save") \
    .trigger(processingTime='5 seconds') \
    .option("checkpointLocation","./save") \
    .outputMode("append") \
    .option("truncate", False) \
    .start().awaitTermination()


