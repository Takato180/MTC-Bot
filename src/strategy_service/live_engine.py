import asyncio
import json
from aiokafka import AIOKafkaConsumer
import yaml
import pandas as pd
from pathlib import Path

from strategy_service.config import Settings
from strategy_dsl import RuleStrategy, SMA
from broker_adapters.bybit_adapter import BybitAdapter

# ブローカーアダプターと設定の初期化
adapter = BybitAdapter()
cfg = Settings()

# テスト用に閾値を小さく
short, long = 1, 2

# 戦略テンプレート読み込み
conf = yaml.safe_load(
    Path("src/strategy_dsl/examples/sma_crossover.yml")
    .read_text(encoding="utf-8")
)
strat = RuleStrategy(**conf)

history = pd.DataFrame(columns=["close"])

async def run():
    consumer = AIOKafkaConsumer(
        cfg.topic,
        bootstrap_servers=cfg.kafka_brokers,
        group_id="live_sma_group",
    )
    await consumer.start()
    print("[INFO] Live engine started. Waiting for Kafka messages...")
    try:
        async for msg in consumer:
            print(f"[MSG] Received raw: {msg.value}")
            k = json.loads(msg.value)
            t = pd.to_datetime(k["open_time"], unit="ms")
            history.loc[t] = k["close"]

            if len(history) >= max(short, long):
                history["SMA_short"] = SMA(history["close"], short)
                history["SMA_long"]  = SMA(history["close"], long)
                pos = strat.signal(history)
                latest, prev = pos.iloc[-1], pos.iloc[-2]

                # 正しいシンボルを使って成行注文を発注
                if prev == 0 and latest == 1:
                    adapter.place_market_order(
                        cfg.trade_symbol,  # "BTCUSDT" のようなシンボル
                        "Buy",
                        0.0001
                    )
                    print("[ORDER] Placed BUY market order")
                elif prev == 1 and latest == 0:
                    adapter.place_market_order(
                        cfg.trade_symbol,
                        "Sell",
                        0.0001
                    )
                    print("[ORDER] Placed SELL market order")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(run())
