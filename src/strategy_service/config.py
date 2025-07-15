from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # カンマ区切りでブローカー指定
    kafka_brokers: str  = "host.docker.internal:19092"
    topic:         str  = "kline_1h"
    bybit_key:     str
    bybit_secret:  str
    trade_symbol:  str  = "BTCUSDT"    # ← 追加

    class Config:
        env_file            = ".env"
        env_file_encoding   = "utf-8"    # ← 文字コード指定
