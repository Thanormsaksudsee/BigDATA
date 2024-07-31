from pyspark.sql.types import *
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
read_file = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand.csv")

sqlDF = read_file.select("status_id", "num_reactions") \
    .filter(read_file["num_reactions"].cast(IntegerType()) > 3000) \
    .withColumnRenamed("num_reactions", "reactions") \
    .orderBy("num_reactions")

sqlDF.show(3)