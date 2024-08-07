from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split
from pyspark.sql.types import StructType, StringType, StructField

spark = SparkSession.builder.appName("FileSourceStreamingExample").getOrCreate()

schema = StructType([
    StructField("status_id", StringType(), True),
    StructField("status_type", StringType(), True),
    StructField("status_published", StringType(), True)
    
])

lines = spark.readStream.format("csv") \
    .option("maxFilesPerTrigger", 1) \
    .option("header", True) \
    .option("path", "C:\\Users\\ADMIN\\Downloads\\BigDATA\\file") \
    .schema(schema) \
    .load()

words = lines.withColumn("date", split(lines["status_published"], " ").getItem(0))

wordCounts = words.groupBy("date", "status_type").count()

query = wordCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()