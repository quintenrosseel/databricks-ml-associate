# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC #Model Training
# MAGIC 
# MAGIC This notebook trains various models for the GA + Linkedin Analytics dataset, using MLflow API.  

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Constants 

# COMMAND ----------

RANDOM_SEED = 42

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Data Cleaning

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Load the dataset from the Workspace directory (repo)

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
# MAGIC ## Explicitely infer dtypes

# COMMAND ----------

numeric_cols = [(cname, dtype) for cname, dtype in ga_li_df.dtypes if dtype in ['double', 'bigint', 'long']] 
string_cols = [cname for cname, dtype in ga_li_df.dtypes if dtype in ['string']] 
double_cols = [cname for cname, dtype in numeric_cols if 'rate' in cname or 'Avg' in cname]
bigint_cols = [cname for cname, dtype in numeric_cols if cname not in double_cols]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Recast

# COMMAND ----------

import pyspark.sql.functions as F
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

# Shuffle randomly 
ga_li_df = ga_li_df.orderBy(F.rand(seed=RANDOM_SEED))

display(ga_li_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Training
# MAGIC 
# MAGIC ## Transformers and Estimators
# MAGIC 
# MAGIC Spark ML standardizes APIs for machine learning algorithms to make it easier to combine multiple algorithms into a single pipeline, or workflow. Let's cover two key concepts introduced by the Spark ML API: **`transformers`** and **`estimators`**.
# MAGIC 
# MAGIC **Transformer**: Transforms one DataFrame into another DataFrame. It accepts a DataFrame as input, and returns a new DataFrame with one or more columns appended to it. Transformers do not learn any parameters from your data and simply apply rule-based transformations. It has a **`.transform()`** method.
# MAGIC 
# MAGIC **Estimator**: An algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model. It has a **`.fit()`** method because it learns (or "fits") parameters from your DataFrame.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Spark NLP (transformers)
# MAGIC 
# MAGIC Taken from 
# MAGIC - Installation: https://nlp.johnsnowlabs.com/docs/en/install#databricks-support
# MAGIC - Tokenization: https://nlp.johnsnowlabs.com/docs/en/annotators#tokenizer
# MAGIC - Document Assembler: https://medium.com/spark-nlp/spark-nlp-101-document-assembler-500018f5f6b5
# MAGIC - Roberta Sentence Embeddings: https://nlp.johnsnowlabs.com/docs/en/transformers#robertasentenceembeddings

# COMMAND ----------

string_cols

# COMMAND ----------

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssemblerPost = (
    DocumentAssembler()
    .setInputCol('Post title')           
    .setOutputCol("Post title doc")
    .setCleanupMode("shrink")
)

postTokenizer = (
    Tokenizer()
    .setInputCols(["Post title doc"])
    .setOutputCol("Post title token").fit(ga_li_df)
)

postEmbedder = (
    RoBertaEmbeddings.pretrained()
    .setInputCols("Post title doc", "Post title token")
    .setOutputCol("Post title embedding")
)


documentAssemblerAuthor = (
     DocumentAssembler()
    .setInputCol('Posted by')           
    .setOutputCol("Posted by doc")
    .setCleanupMode("shrink")
)

authorTokenizer = (
    Tokenizer()
    .setInputCols(["Posted by doc"])
    .setOutputCol("Posted by token").fit(ga_li_df)
)

authorEmbedder = (
    RoBertaEmbeddings.pretrained()
    .setInputCols("Posted by doc", "Posted by token")
    .setOutputCol("Posted by embedding")
)

finisher = (
    Finisher() 
    .setInputCols(["Post title doc", "Posted by doc"])
    .setIncludeMetadata(True)
)

embeddingsFinisher = (
     EmbeddingsFinisher()
    .setInputCols("Post title embedding", "Posted by embedding")
    .setOutputCols("Post title embedding vector", "Posted by embedding vector")
    .setOutputAsVector(True)
    .setCleanAnnotations(False)
)

pipeline = (
    Pipeline().setStages([
        documentAssemblerPost, 
        documentAssemblerAuthor, 
        postTokenizer,
        authorTokenizer,
        postEmbedder,
        authorEmbedder, 
        embeddingsFinisher
    ]
    ).fit(ga_li_df)
)

nlp_features_df = (
    pipeline.transform(ga_li_df)
    .select([
        #"Post title doc", 
        #"Post title token", 
        # "Posted by doc",
        # "Posted by token",
        # "Post title embedding",
        # "Posted by embedding"
        "Post title embedding vector",
        "Posted by embedding vector"
    ])
)

display(nlp_features_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Join numeric & nlp features (dataframes)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

ga_li_nlp_df = (
    ga_li_df.withColumn("id", monotonically_increasing_id()).join(
        (nlp_features_df.withColumn("id", monotonically_increasing_id())), 
        on="id", 
        how="inner"
    )
    .drop("id")
    .drop(*string_cols)
   
    # Unpack Array of Vectors
    .withColumn("Post title embedding vector", nlp_features_df["Post title embedding vector"].getItem(0))
    .withColumn("Posted by embedding vector", nlp_features_df["Posted by embedding vector"].getItem(0))
)

display(ga_li_nlp_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Vector Assembler
# MAGIC The Linear Regression estimator (.fit()) expected a column of Vector type as input.
# MAGIC 
# MAGIC We can easily get the values from the bedrooms column into a single vector using VectorAssembler. VectorAssembler is an example of a transformer. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.
# MAGIC 
# MAGIC You can see an example of how to use VectorAssembler on the ML Programming Guide.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

target_cols = [c for c in ga_li_nlp_df.columns if "GA" in c]
feature_cols = [c for c in ga_li_nlp_df.columns if c not in target_cols]
numeric_columns = [cname for cname, dtype in ga_li_df.dtypes if dtype == "int" or dtype == "double"]

train_df, test_df = ga_li_nlp_df.randomSplit([.8, .2], seed=RANDOM_SEED)

vec_assembler = VectorAssembler(
    inputCols=["Post title embedding vector", "Posted by embedding vector"] + numeric_columns, 
    outputCol="features"
)

vec_train_df = vec_assembler.transform(train_df)

display(vec_train_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Linear Regresion

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="GA number of users")
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Inspect the model

# COMMAND ----------

lr_model.coefficients

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test the model

# COMMAND ----------

vec_test_df = vec_assembler.transform(test_df)
pred_df = lr_model.transform(vec_test_df)

display(pred_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Evaluate the model

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(
    predictionCol="prediction", 
    labelCol="GA number of users", 
    metricName="rmse"
)

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")

# COMMAND ----------


