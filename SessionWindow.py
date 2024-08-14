from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, window, split, current_timestamp

spark = SparkSession.builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()

lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 5000) \
    .load()

words = lines.select(
    explode(
        split(lines.value, " ")
    ).alias("word")
)

words = words.withColumn("timestamp", current_timestamp())

windowCounts = words.groupBy(
    window(words.timestamp, "10 seconds", "5 seconds"),
    words.word
).count()

query = windowCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
