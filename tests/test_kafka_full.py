import asyncio
import json
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from strategy_service.config import Settings
import time

async def test_kafka_communication():
    cfg = Settings()
    
    print("[TEST] Starting Kafka communication test...")
    
    # Consumer setup
    consumer = AIOKafkaConsumer(
        cfg.topic,
        bootstrap_servers=cfg.kafka_brokers,
        group_id="test_group",
        auto_offset_reset='latest'  # 最新のメッセージから開始
    )
    
    # Producer setup
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka_brokers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    await consumer.start()
    await producer.start()
    
    try:
        print("[TEST] Kafka services started")
        
        # テストメッセージ送信
        test_data = {
            "open_time": int(time.time() * 1000),
            "close": 90000.0,
            "symbol": "BTCUSDT"
        }
        
        print(f"[TEST] Sending message: {test_data}")
        await producer.send_and_wait(cfg.topic, test_data)
        print("[TEST] Message sent")
        
        # メッセージ受信待ち
        print("[TEST] Waiting for message...")
        
        async for msg in consumer:
            print(f"[TEST] Received: {msg.value}")
            data = json.loads(msg.value)
            print(f"[TEST] Parsed data: {data}")
            break  # 最初のメッセージを受信したら終了
            
    finally:
        await consumer.stop()
        await producer.stop()
        print("[TEST] Kafka test completed")

if __name__ == "__main__":
    asyncio.run(test_kafka_communication())