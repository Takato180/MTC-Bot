from pybit.unified_trading import HTTP
from strategy_service.config import Settings

class BybitAdapter:
    def __init__(self):
        cfg = Settings()
        self.client = HTTP(
            api_key=cfg.bybit_key,
            api_secret=cfg.bybit_secret,
            testnet=False,  # 本番環境を利用
            recv_window=10000       # timestampずれ対策にリクエスト有効時間を延長
        )

    def place_market_order(self, symbol: str, side: str, qty: float):
        """
        symbol: 'BTCUSDT'
        side: 'Buy' or 'Sell'
        qty: 注文数量
        """
        return self.client.place_order(
            category="linear",           # USDT無期限合約の場合
            symbol=symbol,               # 例: "BTCUSDT"
            side=side.title(),           # 'Buy' または 'Sell'   
            order_type="Market",
            qty=qty,
            time_in_force="GoodTillCancel"
        )