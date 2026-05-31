from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when, abs as spark_abs
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml import PipelineModel
import os
import json

spark = SparkSession.builder \
    .appName("PaySim-Inference-Processor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# FIXED PATH: Points exactly to where your script saved it
# ============================================================
model_path = "/app/fraud_detection_model"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model brain folder not found at {model_path}! Run training first.")
    
print("🧠 Loading trained GBT model architecture pipeline...")
trained_model = PipelineModel.load(model_path)

schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("step", IntegerType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True)
])

kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "transactions") \
    .load()

parsed_stream = kafka_stream \
    .selectExpr("CAST(value AS STRING) as json_payload") \
    .select(from_json(col("json_payload"), schema).alias("data")) \
    .select("data.*")

# ============================================================
# FIXED: Adding the exact feature engineering steps from training
# ============================================================
stream_features_df = parsed_stream \
    .withColumn("hour_of_day", col("step") % 24) \
    .withColumn("day", (col("step") / 24).cast(IntegerType())) \
    .withColumn("is_transfer_cashout", when((col("type") == "TRANSFER") | (col("type") == "CASH_OUT"), 1).otherwise(0)) \
    .withColumn("balance_change", spark_abs(col("oldbalanceOrg") - col("newbalanceOrig"))) \
    .withColumn("amount_ratio", col("amount") / (col("oldbalanceOrg") + 1))

# FIXED PATH: Matches our docker-compose volume mounting layout
JSON_OUTPUT_PATH = "/app/data/shared/live_logs.json"

def apply_model_prediction_batch(df, batch_id):
    if df.count() > 0:
        # Pass streaming rows through the ML model Pipeline
        predictions_df = trained_model.transform(df)
        
        # Extract fields along with the ML model's prediction column
        output_records = predictions_df.select(
            "transaction_id", "step", "type", "amount", "prediction"
        ).collect()
        
        records_list = []
        for row in output_records:
            d = row.asDict()
            # Explicitly label the fraud property as an integer for the JS UI
            d["is_fraud"] = int(row["prediction"])
            del d["prediction"]
            records_list.append(d)
            
        existing_records = []
        if os.path.exists(JSON_OUTPUT_PATH):
            try:
                with open(JSON_OUTPUT_PATH, "r") as f:
                    existing_records = json.load(f)
            except:
                pass
        
        combined = records_list + existing_records
        # Cap the output at 40 rows to ensure zero latency at the web frontend layer
        combined = combined[:40]
        
        with open(JSON_OUTPUT_PATH, "w") as f:
            json.dump(combined, f, indent=4)
        print(f"⚡ Live Batch {batch_id} processed via GBT Model. File updated.", flush=True)

# Use our dataframe containing engineered streaming features
query = stream_features_df.writeStream \
    .foreachBatch(apply_model_prediction_batch) \
    .option("checkpointLocation", "/app/spark_checkpoints") \
    .start()

query.awaitTermination()