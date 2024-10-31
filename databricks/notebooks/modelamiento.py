# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Modelamiento

# COMMAND ----------

import mlflow
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# COMMAND ----------

modeling_dataframe = spark.sql("SELECT * FROM hive_metastore.feature_store.hotel_booking_modeling")

train_df, test_df = modeling_dataframe.randomSplit(weights=[0.8, 0.2], seed=4200)

# COMMAND ----------

mlflow.create_experiment(name="/Users/miguel.arquez12@gmail.com/DSRP Pyspark")

# COMMAND ----------

mlflow.autolog()
mlflow.set_experiment(experiment_name="/Users/miguel.arquez12@gmail.com/DSRP Pyspark")
with mlflow.start_run(run_name="dsrp booking") as run:
  rfc = RandomForestClassifier(labelCol="is_canceled",featuresCol="modeling_features")
  rfc_fitted = rfc.fit(train_df)
  predictions = rfc_fitted.transform(test_df)


  evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
  print(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
  

# COMMAND ----------


