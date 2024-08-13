from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, window

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()

# Create a DataFrame representing the stream of input lines from connection to localhost:9999
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 5000) \
    .load()

# Split the lines into words
words = lines.select(
    explode(
        split(lines.value, " ")
    ).alias("word")
)

# Add a timestamp column (if not already present)
words = words.withColumn("timestamp", current_timestamp())

# Group the data by window and word and compute the counts
windowCounts = words.groupBy(
    window(words.timestamp, "10 seconds", "5 seconds"),
    words.word
).count()

# Write the output to the console
query = windowCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
