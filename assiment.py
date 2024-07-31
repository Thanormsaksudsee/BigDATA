from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import count, countDistinct, first, last, min, max, sum, sumDistinct, col


spark = SparkSession.builder.getOrCreate()


df1 = spark.read.csv('fb_live_thailand2.csv', header=True, inferSchema=True)
df2 = spark.read.csv('fb_live_thailand3.csv', header=True, inferSchema=True)

sqlDF1 = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand2.csv")

sqlDF2 = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand3.csv")

read_file = spark.read.format("csv") \
    .option("header", "true") \
    .load("fb_live_thailand.csv")

sqlDF = read_file.select("status_id", "num_reactions") \
    .filter(read_file["num_reactions"].cast(IntegerType()) > 3000) \
    .withColumnRenamed("num_reactions", "reactions") \
    .orderBy("num_reactions")

# 1
read_file.printSchema()
# 2
sqlDF.show(3)
# 3
print("Count of status_published:")
sqlDF.select(count("status_published")).show()

# 4
print("Count of distinct status_type:")
sqlDF.select(countDistinct("status_type")).show()

#5
print("First and last status_published:")
sqlDF.select(
    first("status_published").alias("first_status_published"),
    last("status_published").alias("last_status_published")
).show()

#6
print("Min and max num_reactions:")
sqlDF.select(min("num_reactions_int").alias("min_reactions"), max("num_reactions_int").alias("max_reactions")).show()


#7
print("Sum of num_reactions:")
sqlDF.select(sum("num_reactions_int")).show()
#8
print("Sum of distinct num_reactions:")
sqlDF.select(sumDistinct("num_reactions_int")).show()

# 23-28
inner_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "inner")
inner_join_df.show()

outer_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "outer")
outer_join_df.show()

left_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "left")
left_join_df.show()

right_join_df = sqlDF1.join(sqlDF2, sqlDF1["status_id"] == sqlDF2["status_id"], "right")
right_join_df.show()
