import mlflow
import yaml, pandas as pd, matplotlib.pyplot as plt
from strategy_dsl import SMA, RuleStrategy
from pathlib import Path

# 1) 最適化結果読み込み
best = yaml.safe_load(Path("optuna_best.json").read_text())
short, long = best["short"], best["long"]

# 2) データ読み込み
df = pd.read_csv("data/BTCUSDT_1h.csv", parse_dates=["open_time"]).set_index("open_time")
df["SMA_short"] = SMA(df["close"], short)
df["SMA_long"]  = SMA(df["close"], long)

# 3) シグナル生成
with open("src/strategy_dsl/examples/sma_crossover.yml", encoding="utf-8") as f:
    conf = yaml.safe_load(f)
strat = RuleStrategy(**conf)
pos   = strat.signal(df)

# 4) パフォーマンス計算
entries = pos.shift(1).fillna(False).astype(bool)
exits   = pos == 0
equity_bh = (df["close"].pct_change()+1).cumprod()
equity_strat = (df["close"].pct_change()*entries + 1).cumprod()

# 5) MLflow ログ
mlflow.set_experiment("sma_crossover_backtest")
with mlflow.start_run():
    mlflow.log_params({"short": short, "long": long})
    mlflow.log_metric("total_return", equity_strat.iloc[-1])
    # 6) プロット生成
    plt.figure()
    equity_bh.plot(label="Buy & Hold")
    equity_strat.plot(label="SMA Strat")
    plt.legend(); plt.title("Equity Curve"); plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "equity_curve.png")
    plt.close()
