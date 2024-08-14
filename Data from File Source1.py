from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SchemaExample") \
    .getOrCreate()

# Define the schema
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

# Load the datasets with the defined schema
df_part1 = spark.read \
    .format("csv") \
    .option("header", "true") \
    .schema(file_schema) \
    .load("fb1.csv")

df_part2 = spark.read \
    .format("csv") \
    .option("header", "true") \
    .schema(file_schema) \
    .load("fb2.csv")

# Show the first few rows of each DataFrame
df_part1.show()
df_part2.show()
