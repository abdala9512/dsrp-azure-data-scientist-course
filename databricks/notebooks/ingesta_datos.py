# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Ingesta Datos

# COMMAND ----------

raw_dataframe = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/hotel_booking.csv")
raw_dataframe.write.saveAsTable("hive_metastore.raw.hotel_bookings")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM hive_metastore.raw.hotel_bookings

# COMMAND ----------


