from pyspark.sql import SparkSession
from pyspark.sql.functions import split, current_timestamp
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()

# Create a DataFrame representing the stream of input lines from connection to localhost:9999
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Process the lines: Split lines into words and add timestamp
words = lines.withColumn("word", split(lines["value"], " ")) \
    .withColumn("word", F.explode(F.col("word"))) \
    .withColumn("timestamp", F.current_timestamp()) \
    .withWatermark("timestamp", "10 seconds")

# Group by window and word and count occurrences
windowedCounts = words.groupBy(
    F.window(words["timestamp"], "10 seconds", "5 seconds"),
    words["word"]
).count()

# Output the results to the console
query = windowedCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# Await termination of the stream
query.awaitTermination()
