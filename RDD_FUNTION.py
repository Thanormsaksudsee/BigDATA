from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

rdd = spark.sparkContext.textFile('fb_live_thailand.csv', 5)

count_distinct = rdd.distinct().count()
print('Number of distinct records: ', count_distinct)


filter_rdd = rdd.filter(lambda x: x.split(',')[1] == 'link').collect()
print(filter_rdd)