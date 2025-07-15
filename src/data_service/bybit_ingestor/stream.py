import yaml, time, platform, datetime as dt
from pathlib import Path
from pybit.unified_trading import WebSocket
from .producer import send

# --- 設定読込 ---
cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
topic   = cfg["topic"]
symbol  = cfg["symbol"]
interval= cfg["interval"]      # "60" など文字列
testnet = cfg.get("testnet", False)

# --- WebSocket 初期化 ---
ws = WebSocket(channel_type="linear", testnet=testnet)

def on_msg(msg: dict):
    # print("RAW", msg)
    if "data" not in msg:                  # ping/pong や subscribe 成功メッセ
        return
    bar_raw = msg["data"][0]               # [start, open, high, low, close, volume, turnover]
    bar = {
        "start":    int(bar_raw[0]),       # ms
        "open":     float(bar_raw[1]),
        "high":     float(bar_raw[2]),
        "low":      float(bar_raw[3]),
        "close":    float(bar_raw[4]),
        "volume":   float(bar_raw[5]),
        "turnover": float(bar_raw[6]),
        "ingest_ts": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    send(topic, bar)

print(f"WS subscribe {symbol} {interval}m → Kafka {topic}")
ws.kline_stream(symbol=symbol, interval=interval, callback=on_msg)

# --- Windows 用の“無限待機” ---
print("Ingestor running. Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(3600)   # 1 時間ごとに起きて再 loop
except KeyboardInterrupt:
    print("Stopped by user")
