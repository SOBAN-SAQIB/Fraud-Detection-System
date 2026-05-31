import json
import time
import glob
import uuid
from kafka import KafkaProducer

print("⏳ Establishing connection bounds to Kafka pipeline...")
while True:
    try:
        producer = KafkaProducer(
            bootstrap_servers=['kafka:29092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        break
    except Exception:
        time.sleep(2)

# Point to the json files outputted during the training script run
json_shards = glob.glob("/app/test_stream_source/*.json")
if not json_shards:
    raise FileNotFoundError("❌ No test stream data available! Run train_pipeline.py first.")

print("🚀 Production streaming active. Transmitting unseen test events...")

for shard in json_shards:
    with open(shard, "r") as f:
        for line in f:
            if not line.strip():
                continue
            txn_payload = json.loads(line.strip())
            
            # Map an ad-hoc UUID so rows possess transaction identification markers
            txn_payload["transaction_id"] = f"TXN-{uuid.uuid4().hex[:6].upper()}"
            
            # Send clean, unlabeled transaction records directly to Kafka
            producer.send('transactions', value=txn_payload)
            
            # FIXED: Removed the missing 'isFraud' key to prevent a script crash
            print(f"📡 Dispatched: ID={txn_payload['transaction_id']} | Type={txn_payload['type']} | Amt=${txn_payload['amount']:.2f}", flush=True)
            
            time.sleep(0.4)