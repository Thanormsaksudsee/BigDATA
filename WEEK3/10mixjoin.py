from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()

sqlDF1 = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand2.csv")

sqlDF2 = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand3.csv")


inner_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "inner")
inner_join_df.show()

outer_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "outer")
outer_join_df.show()

left_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "left")
left_join_df.show()

right_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "right")
right_join_df.show()
