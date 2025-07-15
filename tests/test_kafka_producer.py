import asyncio
import json
from aiokafka import AIOKafkaProducer
from strategy_service.config import Settings
import time

async def send_test_message():
    cfg = Settings()
    
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka_brokers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    await producer.start()
    
    try:
        # テスト用のダミー価格データ
        test_data = {
            "open_time": int(time.time() * 1000),  # 現在時刻をミリ秒で
            "close": 90000.0,  # BTC価格
            "symbol": "BTCUSDT"
        }
        
        print(f"[PRODUCER] Sending test message: {test_data}")
        
        await producer.send_and_wait(
            cfg.topic,
            test_data
        )
        
        print("[PRODUCER] Test message sent successfully!")
        
    finally:
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(send_test_message())