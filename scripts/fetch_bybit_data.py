"""
fetch_bybit_data.py
Bybit から履歴 Kline を取得し Parquet 保存
"""
import argparse, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from pybit.unified_trading import HTTP

BATCH = 1000  # API 1 回の上限本数

def fetch(http, sym, interval, start_ts, end_ts):
    rows = []
    cursor = start_ts
    while cursor < end_ts:
        res = http.get_kline(
            symbol=sym,
            interval=str(interval),
            start_time=cursor,
            end_time=min(cursor + interval*60*BATCH, end_ts),
            limit=BATCH,
        )
        data = res["result"]["list"]
        if not data:
            break
        rows.extend(data)
         # data[-1][0] == 最後のローの start_time (Unix 秒)
        cursor = int(data[-1][0]) + interval * 60 * 1000 #ミリ秒:60*1000
        time.sleep(0.2)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end",   required=True)
    ap.add_argument("--out",   default="data/raw")
    args = ap.parse_args()

    http   = HTTP(testnet=False)
    s_dt   = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    e_dt   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    rows = fetch(http, args.symbol, args.interval,
                 int(s_dt.timestamp()), int(e_dt.timestamp()))
    if not rows:
        print("No data returned.")
        return

    cols = ["start","open","high","low","close","volume","turnover"]
    df = pd.DataFrame(rows, columns=cols)

    #ミリ秒へ変換
    df["open_time"] = pd.to_datetime(df["start"], unit="ms", utc=True)
    out_path = Path(args.out)                      #  ファイルパス
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)               #  CSV で保存
    print(f"Saved {len(df)} rows → {out_path}")

if __name__ == "__main__":
    main()
