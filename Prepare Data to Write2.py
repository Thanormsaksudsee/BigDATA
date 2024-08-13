from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, current_timestamp

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GroupByExample") \
    .getOrCreate()

# Define the schema for the CSV files
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

# Read streaming data from CSV files in a directory
lines = spark \
    .readStream \
    .format("csv") \
    .option("maxFilesPerTrigger", 1) \
    .option("header", True) \
    .schema(file_schema) \
    .load("../data/stream")

# Add date and timestamp columns, and apply watermark
words = lines \
    .withColumn("date", split(col("status_published"), " ").getItem(0)) \
    .withColumn("timestamp", current_timestamp()) \
    .withWatermark("timestamp", "10 seconds")

# Group by date, status_type, and timestamp, and count the occurrences
wordCounts = words \
    .groupBy("date", "status_type", "timestamp") \
    .count()

# For demonstration, let's write the output to the console
query = wordCounts.writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

# Await termination of the query
query.awaitTermination()
