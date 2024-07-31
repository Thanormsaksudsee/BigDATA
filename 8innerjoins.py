from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

sqlDF1 = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand2.csv")

sqlDF2 = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand3.csv")

joined_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "inner")

joined_df.show()
