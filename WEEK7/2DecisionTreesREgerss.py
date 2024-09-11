# 1. Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("DecisionTreeRegressionExample") \
    .getOrCreate()

data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

indexer_reactions = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")
indexer_loves = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

encoder_reactions = OneHotEncoder(inputCols=["num_reactions_ind"], outputCols=["num_reactions_vec"])
encoder_loves = OneHotEncoder(inputCols=["num_loves_ind"], outputCols=["num_loves_vec"])

assembler = VectorAssembler(inputCols=["num_reactions_vec", "num_loves_vec"], outputCol="features")

pipeline = Pipeline(stages=[indexer_reactions, indexer_loves, encoder_reactions, encoder_loves, assembler])

pipeline_model = pipeline.fit(data)

transformed_data = pipeline_model.transform(data)

train_data, test_data = transformed_data.randomSplit([0.8, 0.2])

dt = DecisionTreeRegressor(labelCol="num_loves_ind", featuresCol="features")

dt_model = dt.fit(train_data)

predictions = dt_model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2 score: {r2}")

spark.stop()
