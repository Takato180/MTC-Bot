import asyncio
import json
from aiokafka import AIOKafkaProducer
from strategy_service.config import Settings
import time
import subprocess
import threading

def run_live_engine():
    """live_engine.pyを別スレッドで実行"""
    try:
        result = subprocess.run(
            ['python', '-m', 'src.strategy_service.live_engine'],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"[LIVE ENGINE] STDOUT: {result.stdout}")
        print(f"[LIVE ENGINE] STDERR: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("[LIVE ENGINE] Timeout - this is expected")
    except Exception as e:
        print(f"[LIVE ENGINE] Error: {e}")

async def send_test_messages():
    """テストメッセージを送信"""
    cfg = Settings()
    
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka_brokers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    await producer.start()
    
    try:
        # 複数のメッセージを送信してSMAクロスオーバーを発生させる
        messages = [
            {"open_time": int(time.time() * 1000), "close": 89000.0, "symbol": "BTCUSDT"},
            {"open_time": int(time.time() * 1000) + 1000, "close": 90000.0, "symbol": "BTCUSDT"},
            {"open_time": int(time.time() * 1000) + 2000, "close": 91000.0, "symbol": "BTCUSDT"},
        ]
        
        for i, msg in enumerate(messages):
            print(f"[PRODUCER] Sending message {i+1}: {msg}")
            await producer.send_and_wait(cfg.topic, msg)
            await asyncio.sleep(1)  # 1秒間隔で送信
            
        print("[PRODUCER] All messages sent")
        
    finally:
        await producer.stop()

async def main():
    print("[MAIN] Starting live engine test...")
    
    # live_engine.pyを別スレッドで起動
    engine_thread = threading.Thread(target=run_live_engine)
    engine_thread.start()
    
    # 少し待ってからメッセージを送信
    await asyncio.sleep(3)
    
    await send_test_messages()
    
    # エンジンスレッドの終了を待つ
    engine_thread.join()
    
    print("[MAIN] Test completed")

if __name__ == "__main__":
    asyncio.run(main())