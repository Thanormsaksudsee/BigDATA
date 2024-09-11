from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType

spark = SparkSession.builder \
    .appName("Linear Regression Analysis") \
    .getOrCreate()

data = spark.read.csv('fb_live_thailand.csv', header=True, inferSchema=True)
data.show()

assembler = VectorAssembler(
    inputCols=['num_reactions', 'num_loves'],
    outputCol='features'
)

data_assembled = assembler.transform(data)
data_assembled.show()

linear_regression = LinearRegression(
    labelCol='num_loves',  # Label column for the regression
    featuresCol='features',    # Features column for the regression
    maxIter=10,                # Set maximum number of iterations (adjust as needed)
    regParam=0.3,              # Set regularization parameter (0...1, adjust as needed)
    elasticNetParam=0.8        # Set ElasticNet mixing parameter (0...1, adjust as needed)
)

pipeline = Pipeline(stages=[linear_regression])

train_data, test_data = data_assembled.randomSplit([0.8, 0.2], seed=42)

pipeline_model = pipeline.fit(train_data)

predictions = pipeline_model.transform(test_data)

predictions.select('num_loves', 'features', 'prediction').show(5)

evaluator = RegressionEvaluator(
    labelCol='num_loves',  
    predictionCol='prediction'  
)

mse = evaluator.setMetricName("mse").evaluate(predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")

r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2: {r2:.4f}")

pandas_df = predictions.select('num_loves', 'prediction').toPandas()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_loves', y='prediction', data=pandas_df)
plt.title('Scatter Plot of num_loves vs Prediction')
plt.xlabel('num_loves')
plt.ylabel('Prediction')
plt.show()

selected_data = predictions.select(
    col('num_loves').cast(IntegerType()).alias('num_loves'),
    col('prediction').cast(IntegerType()).alias('prediction')
).orderBy(col('prediction').desc())

pandas_df = selected_data.toPandas()

plt.figure(figsize=(10, 6))
sns.lmplot(x='num_loves', y='prediction', data=pandas_df, aspect=1.5)

plt.title('Linear Regression: num_loves vs Prediction')
plt.xlabel('num_loves')
plt.ylabel('Prediction')
plt.show()