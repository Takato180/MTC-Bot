from pathlib import Path
import json, yaml, optuna, vectorbt as vbt, pandas as pd
from strategy_dsl import SMA, RuleStrategy

DATA_CSV = "data/BTCUSDT_1h.csv"  # 事前に用意

df = pd.read_csv(DATA_CSV, parse_dates=["open_time"]).set_index("open_time")
price = df["close"]  # 価格データを抽出

"""
Objective function for optimizing the SMA crossover trading strategy.
Overview:
    この関数はシンプル移動平均（SMA）クロスオーバー戦略のハイパーパラメータ（短期と長期のSMA期間）を最適化するために設計されている。
    Optunaのtrialオブジェクトを使用して、'short'（短期SMA期間）と'long'（長期SMA期間）の値を探索する。
    指定されたYAML設定ファイルから戦略のルールを読み込み、この設定を用いてトレーディングシグナルを生成する。
    生成されたシグナルに基づき、vectorbtライブラリを利用してポートフォリオシミュレーションを行い、
    最終的にシミュレーションされたポートフォリオの総リターンを返す。
Parameters:
    trial (optuna.Trial): ハイパーパラメータ探索のためのOptunaトライアルオブジェクト。'short'と'long'の期間がサジェストされる。
Returns:
    float: シミュレートされたポートフォリオの総リターン。
"""
def objective(trial: optuna.Trial):

    short = trial.suggest_int("short", 5, 100)
    long  = trial.suggest_int("long", 50, 400)

    local_df = df.copy()
    local_df["SMA_short"] = SMA(local_df["close"], short)
    local_df["SMA_long"]  = SMA(local_df["close"], long)
    
    # YAML をそのまま読み込むだけで OK
    conf = yaml.safe_load(
        Path("src/strategy_dsl/examples/sma_crossover.yml").read_text()
    )
    strat = RuleStrategy(**conf)
    position = strat.signal(local_df)
    # -----------------------------------------------

    # ポートフォリオを計算
    # entries: position.shift(1).astype(bool) で前日のシグナル
    # exits: position == 0 でポジションを閉じる
    # ここでは position が 0 以外のときにエントリー
    # つまり position が 0 のときはエグジット
    # これでポジションを持たない状態に戻る
    port = vbt.Portfolio.from_signals(
        local_df["close"],
        entries=position.shift(1).astype(bool),
        exits=position == 0
    )
    return port.total_return()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("best params:", study.best_params)
Path("optuna_best.json").write_text(json.dumps(study.best_params, indent=2))
