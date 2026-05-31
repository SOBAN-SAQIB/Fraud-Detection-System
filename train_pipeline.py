from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, abs
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType
)

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml import Pipeline

import os

# ============================================================
# Spark Session
# ============================================================

spark = SparkSession.builder \
    .appName("PaySim-Fraud-Training") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ============================================================
# Dataset Schema
# ============================================================

schema = StructType([
    StructField("step", IntegerType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True)
])

# ============================================================
# Load Dataset
# ============================================================

#  The correct absolute container path
csv_path = "/app/datasets/paysim_data.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

print("Loading dataset...")

df = spark.read.csv(
    csv_path,
    header=True,
    schema=schema
)

# ============================================================
# Feature Engineering
# ============================================================

print("Engineering features...")

df = df.withColumn(
    "label",
    col("isFraud").cast(DoubleType())
)

df = df.withColumn(
    "hour_of_day",
    col("step") % 24
)

df = df.withColumn(
    "day",
    (col("step") / 24).cast(IntegerType())
)

df = df.withColumn(
    "is_transfer_cashout",
    when(
        (col("type") == "TRANSFER") |
        (col("type") == "CASH_OUT"),
        1
    ).otherwise(0)
)

df = df.withColumn(
    "balance_change",
    abs(
        col("oldbalanceOrg") -
        col("newbalanceOrig")
    )
)

df = df.withColumn(
    "amount_ratio",
    col("amount") /
    (col("oldbalanceOrg") + 1)
)

# ============================================================
# Train / Validation / Test Split
# ============================================================

train_df, val_df, test_df = df.randomSplit(
    [0.7, 0.15, 0.15],
    seed=42
)

# Optimize memory caching strategy
val_df.cache()
test_df.cache()

# ============================================================
# Save Test Data For Streaming (FIXED: Drops all ground truth)
# ============================================================

stream_path = "test_stream_source"

print("Saving clean, unlabeled test rows for streaming simulation...")
test_df.drop("label", "isFraud", "isFlaggedFraud") \
       .write \
       .mode("overwrite") \
       .json(stream_path)

# ============================================================
# Handle Class Imbalance
# ============================================================

fraud_df = train_df.filter(col("label") == 1)
normal_df = train_df.filter(col("label") == 0)

fraud_count = fraud_df.count()
normal_count = normal_df.count()

print(f"Fraud Records  : {fraud_count}")
print(f"Normal Records : {normal_count}")

ratio = normal_count / fraud_count

fraud_upsampled = fraud_df.sample(
    withReplacement=True,
    fraction=ratio,
    seed=42
)

balanced_train = normal_df.union(fraud_upsampled)
balanced_train.cache() # Cache right here before the model reads it!

# ============================================================
# ML Pipeline
# ============================================================

indexer = StringIndexer(
    inputCol="type",
    outputCol="typeIndex",
    handleInvalid="keep"
)

feature_columns = [
    "step",
    "hour_of_day",
    "day",
    "typeIndex",
    "amount",
    "amount_ratio",
    "balance_change",
    "is_transfer_cashout"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    maxIter=20,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

pipeline = Pipeline(
    stages=[
        indexer,
        assembler,
        gbt
    ]
)

# ============================================================
# Train Model
# ============================================================

print("Training model...")

model = pipeline.fit(balanced_train)

# ============================================================
# Validation Metrics
# ============================================================

print("Evaluating model...")

val_predictions = model.transform(val_df)

roc_auc = BinaryClassificationEvaluator(
    labelCol="label"
).evaluate(val_predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(val_predictions)

recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(val_predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
).evaluate(val_predictions)

print("\n===== Validation Metrics =====")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ============================================================
# Test Evaluation
# ============================================================

test_predictions = model.transform(test_df)

test_auc = BinaryClassificationEvaluator(
    labelCol="label"
).evaluate(test_predictions)

print(f"\nTest ROC-AUC : {test_auc:.4f}")

# ============================================================
# Save Model
# ============================================================

model_path = "fraud_detection_model"

model.write() \
     .overwrite() \
     .save(model_path)

print(f"\nModel saved to: {model_path}")
print("Training completed successfully.")