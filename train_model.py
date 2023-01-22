# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC #Model Training
# MAGIC 
# MAGIC This notebook trains various models for the GA + Linkedin Analytics dataset, using MLflow API.  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Data Cleaning

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC **Load the dataset from the Workspace directory (repo)**

# COMMAND ----------

ga_li_df = (
    spark.read.parquet('file:/Workspace/Repos/rossequ@cronos.be/databricks-ml-associate/data/ga_li_prediction.parquet')
    .drop('__index_level_0__')
)

# Drop columns with Campaign in them (irrelevant)
for cname in [cname for cname in ga_li_df.columns if "Campaign" in cname]: 
    ga_li_df = ga_li_df.drop(cname)

display(ga_li_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC **Explicitely infer dtypes** 

# COMMAND ----------

numeric_cols = [(cname, dtype) for cname, dtype in ga_li_df.dtypes if dtype in ['double', 'bigint', 'long']] 
string_cols = [cname for cname, dtype in ga_li_df.dtypes if dtype in ['string']] 
double_cols = [cname for cname, dtype in numeric_cols if 'rate' in cname or 'Avg' in cname]
bigint_cols = [cname for cname, dtype in numeric_cols if cname not in double_cols]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC **Recast** 

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, DoubleType, IntegerType

for dc in double_cols: 
     ga_li_df = ga_li_df.withColumn(dc, col(dc).cast(DoubleType()))

for bi in bigint_cols: 
     ga_li_df = ga_li_df.withColumn(bi, col(bi).cast(IntegerType()))
        
# Fill na's 
ga_li_df = ga_li_df.fillna(.0, subset=double_cols)
ga_li_df = ga_li_df.fillna(0, subset=bigint_cols)
ga_li_df = ga_li_df.fillna("None", string_cols)

display(ga_li_df)

# COMMAND ----------


