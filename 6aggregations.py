from pyspark.sql.types import *
from pyspark.sql import SparkSession
# List of CSV file paths
spark = SparkSession.builder.getOrCreate()
file_paths = [
    "fb_live_thailand.csv",
    "fb_live_thailand2.csv",
    "fb_live_thailand3.csv"
]

# Read multiple CSV files
read_file = spark.read.format("csv") \
    .option("header", "true") \
    .load(file_paths)  

# Create a temporary view
read_file.createOrReplaceTempView("fb_live_thailand_view")

# Select all data from the temporary view
result = spark.sql("SELECT * FROM fb_live_thailand_view")

# Show the result
result.show()