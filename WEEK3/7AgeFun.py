from pyspark.sql.functions import count
from pyspark.sql import SparkSession    
# Create a SparkSession
spark = SparkSession.builder.getOrCreate()


read_file = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand.csv")

read_file.createOrReplaceTempView("fb_live_thailand_view")

result = spark.sql("SELECT * FROM fb_live_thailand_view")


sqlDF = read_file.select(count("status_id"))

sqlDF = spark.sql("SELECT * FROM fb_live_thailand.csv where "status_type" == 'photo'")

sqlDF.select(count("status_published"))