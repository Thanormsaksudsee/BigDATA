from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

read_file = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand.csv")

read_file.printSchema()