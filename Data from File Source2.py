from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StreamingCSVExample") \
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

# For demonstration, let's write the output to the console
query = lines.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# Await termination of the query
query.awaitTermination()
