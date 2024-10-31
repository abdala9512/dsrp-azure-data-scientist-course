# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Feature Engineering

# COMMAND ----------

import pyspark
from pyspark.ml.feature import (
  VarianceThresholdSelector, 
  OneHotEncoder,
  StandardScaler, 
  VectorAssembler,
  StringIndexer
)
from pyspark.sql import functions as F

# COMMAND ----------


dataframe_reservas = spark.table("hive_metastore.raw.hotel_bookings")

# COMMAND ----------

from typing import Union, List
from pyspark.ml import Pipeline

class DistributedDataProcessor:

  def __init__(self, data: pyspark.sql.DataFrame) -> None:
    self.data = data


  def __get_encoders(self):
    ohe_column_list = [
            "arrival_date_year", 
            "arrival_date_month",
            "meal",
            "market_segment",
            "distribution_channel", 
            "reserved_room_type",
            "deposit_type",
            "customer_type"
        ]
    indexers = [
      StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
      for c in ohe_column_list
    ]

    encoders = [
      OneHotEncoder(
        dropLast=False,
        inputCol=indexer.getOutputCol(),
        outputCol=f"{indexer.getOutputCol()}_indexed"
      )
      for indexer in indexers
    ]

    return indexers, encoders

  def __encode_with_ohe(self, col_name: Union[str, List[str]]):
    """
    ONE HOT ENCODING DE CUALQUIER COLUMNA

    col_names: LIST DE COLUMNAS PARA APLICAR OHE
    """
    encoder = OneHotEncoder(inputCol=col_name, outputCol=f"encoded_{col_name}").fit(self.data)
    self.data = encoder.transform(self.data)

  def process(self) -> None:

    self.data = self.data.withColumn(
      "is_city_hotel", F.udf(lambda x: int(x == "City Hotel"))("hotel")
    ).drop("hotel")
    self.data = self.data.withColumn(
      "diff_room", 
      F.when(F.col("assigned_room_type") == F.col("reserved_room_type"), 1).otherwise(0)
    )
  
    indexers, encoders = self.__get_encoders()
    #Scaling
    std_vars = ["lead_time", "adr", "days_in_waiting_list"]
    pipeline = Pipeline(
      stages= 
        indexers 
      + encoders
      + [
        VectorAssembler(
          inputCols=[encoder.getOutputCol() for encoder in encoders],
          outputCol="ohe_features"
        )
      ]
      + [
          VectorAssembler(inputCols=std_vars, outputCol="scaling_features")
        ] 
      + [
        StandardScaler(inputCol="scaling_features", outputCol="scaled_features")
      ]
      + [
        VectorAssembler(
          inputCols=["scaled_features", "ohe_features"],
          outputCol="modeling_features"
        )
      ]
    ).fit(self.data)

    self.data = pipeline.transform(self.data)

    # COLUMNAS QUE VAMOS A ELIMINAR
    drop_list = [
            "country", 
            "company", 
            "agent",
            "reservation_status", 
            "reservation_status_date",
            "name", 
            "email", 
            "phone-number", 
            "credit_card",
            "assigned_room_type"
    ]
    self.data = self.data.drop(*drop_list)
    
  def write(self, table_name: str):

    self.data.write.saveAsTable(f"hive_metastore.feature_store.{table_name}")

    


# COMMAND ----------

processor = DistributedDataProcessor(data=dataframe_reservas)
processor.process()
processor.write(table_name="hotel_booking_modeling")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM hive_metastore.feature_store.hotel_booking_modeling LIMIT 10

# COMMAND ----------


