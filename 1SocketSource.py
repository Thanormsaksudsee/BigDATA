from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col

spark = (SparkSession
    .builder
    .appName("StructuredStreaming")
    .getOrCreate())

host, port = "localhost", 5000

lines = (spark
    .readStream
    .format("socket")
    .option("host", host)
    .option("port", port)
    .load())

print("lines isStreaming: ", lines.isStreaming)

words = lines.select(
    explode(
        split(lines.value, " ")
    ).alias("word")
)

word_counts = words.groupBy("word").count()

query = (word_counts
    .writeStream
    .outputMode("complete")
    .format("console")
    .option("truncate", False)
    .option("numRows", 1000)
    .start())

spark.streams.awaitAnyTermination()

spark.close()
