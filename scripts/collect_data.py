#!/usr/bin/env python3
"""
MTC-Bot用データ収集スクリプト

このスクリプトはBybitから過去の暗号通貨データを収集し、
PatchTSTモデルの学習に適した形式で保存します。
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path
import json
from typing import List, Dict, Any

# srcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from pybit.unified_trading import HTTP
except ImportError:
    print("エラー: pybitがインストールされていません。以下でインストールしてください: pip install pybit")
    sys.exit(1)


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='MTC-Bot用暗号通貨データの収集')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='取引シンボル (デフォルト: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='60',
                       choices=['1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D'],
                       help='足間隔（分） (デフォルト: 60)')
    parser.add_argument('--days', type=int, default=365,
                       help='収集する日数 (デフォルト: 365)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='出力ディレクトリ (デフォルト: data)')
    parser.add_argument('--testnet', action='store_true',
                       help='メインネットの代わりにテストネットを使用')
    parser.add_argument('--validate', action='store_true',
                       help='収集したデータを検証')
    
    return parser.parse_args()


class DataCollector:
    """Bybit APIを使用した暗号通貨データコレクター"""
    
    def __init__(self, testnet: bool = False):
        """
        データコレクターの初期化
        
        引数:
            testnet: テストネットを使用するかどうか
        """
        self.client = HTTP(testnet=testnet)
        self.testnet = testnet
        
    def collect_klines(self, symbol: str, interval: str, days: int) -> List[Dict[str, Any]]:
        """
        Bybitから足データを収集する
        
        引数:
            symbol: 取引シンボル
            interval: 足間隔
            days: 収集する日数
        
        戻り値:
            足データを保持する辞書のリスト
        """
        print(f"{symbol}の{days}日分の{interval}分足データを収集中...")
        
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        
        # 日ごとの足数を定義
        intervals_per_day = {
            '1': 1440, '3': 480, '5': 288, '15': 96, '30': 48,
            '60': 24, '120': 12, '240': 6, '360': 4, '720': 2, 'D': 1
        }
        
        limit = 1000  # Bybit APIの1リクエストあたりの最大データ数
        total_intervals = days * intervals_per_day[interval]
        total_requests = (total_intervals + limit - 1) // limit
        
        print(f"必要なリクエスト総数: {total_requests}")
        
        for request_num in range(total_requests):
            try:
                print(f"Request {request_num + 1}/{total_requests}", end=' ')
                
                response = self.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    end=end_time,
                    limit=limit
                )
                
                if response['retCode'] != 0:
                    print(f"\nError: {response['retMsg']}")
                    break
                
                klines = response['result']['list']
                
                if not klines:
                    print("\nNo more data available")
                    break
                
                all_data.extend(klines)
                print(f"- Collected {len(klines)} klines")
                
                # 次のリクエスト用にend_timeを更新
                end_time = int(klines[-1][0]) - 1
                
                # レート制限対策のために待機
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nError collecting data: {e}")
                break
        
        print(f"\nTotal klines collected: {len(all_data)}")
        return all_data
    
    def process_klines(self, klines: List[List[str]], symbol: str) -> pd.DataFrame:
        """
        APIから取得した生のklineデータをDataFrameに変換する
        
        引数:
            klines: APIから取得した生のklineデータ
            symbol: 取引シンボル
        
        戻り値:
            整形済みのDataFrame（テーブル形式のデータ）
        """
        print("klineデータの整形処理を開始...")
        
        # DataFrameに変換
        df = pd.DataFrame(klines, columns=[
            'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # データ型を変換（オーバーフロー対策）
        try:
            df['start_time'] = pd.to_datetime(df['start_time'].astype(np.int64), unit='ms')
        except (ValueError, OverflowError):
            # 文字列として数値に変換してからタイムスタンプに変換
            df['start_time'] = pd.to_datetime(pd.to_numeric(df['start_time'], errors='coerce'), unit='ms')
        
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
        
        # シンボル情報を追加
        df['symbol'] = symbol
        
        # タイムスタンプで並び替え
        df = df.sort_values('start_time').reset_index(drop=True)
        
        # start_time列の名前をtimestampに変更
        df = df.rename(columns={'start_time': 'timestamp'})
        
        print(f"整形済みデータの件数: {len(df)}")
        print(f"期間: {df['timestamp'].min()} から {df['timestamp'].max()} まで")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        収集したデータの検証を行う
        
        引数:
            df: 整形済みデータのDataFrame
        
        戻り値:
            検証結果を保持する辞書
        """
        print("データの検証を開始...")
        
        validation_results = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_timestamps': df.duplicated(subset=['timestamp']).sum(),
            'price_anomalies': 0,
            'volume_anomalies': 0,
            'gaps_in_data': 0
        }
        
        # 価格の急激な変動（50%以上の変化）を検出
        df['price_change'] = df['close'].pct_change()
        price_anomalies = (abs(df['price_change']) > 0.5).sum()  # 50%超の変化
        validation_results['price_anomalies'] = price_anomalies
        
        # 売買量の異常値を検出
        volume_q99 = df['volume'].quantile(0.99)
        volume_anomalies = (df['volume'] > volume_q99 * 10).sum()
        validation_results['volume_anomalies'] = volume_anomalies
        
        # データの連続性（間隔）の検証
        time_diffs = df['timestamp'].diff()
        expected_interval = time_diffs.mode()[0]
        gaps = (time_diffs > expected_interval * 1.5).sum()
        validation_results['gaps_in_data'] = gaps
        
        # 検証結果の概要を表示
        print("検証結果:")
        for key, value in validation_results.items():
            print(f"  {key}: {value}")
        
        return validation_results


def main():
    """データ収集のメイン関数"""
    args = parse_arguments()
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # データコレクターを初期化
    collector = DataCollector(testnet=args.testnet)
    
    # データの収集
    klines = collector.collect_klines(args.symbol, args.interval, args.days)
    
    if not klines:
        print("データが収集できませんでした。終了します。")
        return
    
    # データの整形処理
    df = collector.process_klines(klines, args.symbol)
    
    # 検証オプションが有効ならデータの検証を実施
    validation_results = None
    if args.validate:
        validation_results = collector.validate_data(df)
    
    # データの保存
    interval_suffix = f"{args.interval}m" if args.interval != 'D' else 'D'
    filename = f"{args.symbol}_{interval_suffix}.csv"
    filepath = os.path.join(args.output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"データを {filepath} に保存しました")
    
    # メタデータの保存
    metadata = {
        'symbol': args.symbol,
        'interval': args.interval,
        'days_collected': args.days,
        'total_records': len(df),
        'date_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat()
        },
        'columns': list(df.columns),
        'testnet': args.testnet,
        'collection_time': datetime.now().isoformat(),
        'validation_results': validation_results
    }
    
    metadata_filepath = os.path.join(args.output_dir, f"{args.symbol}_{interval_suffix}_metadata.json")
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"メタデータを {metadata_filepath} に保存しました")
    
    # サマリの表示
    print("\n" + "="*50)
    print("データ収集の概要")
    print("="*50)
    print(f"シンボル: {args.symbol}")
    print(f"足間隔: {args.interval} 分")
    print(f"件数: {len(df):,}")
    print(f"期間: {df['timestamp'].min()} から {df['timestamp'].max()} まで")
    print(f"ファイル: {filepath}")
    print(f"ファイルサイズ: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    if validation_results:
        print("\n検証結果:")
        issues = sum([
            validation_results['price_anomalies'],
            validation_results['volume_anomalies'],
            validation_results['gaps_in_data']
        ])
        if issues == 0:
            print("  ✅ 問題は見つかりませんでした")
        else:
            print(f"  ⚠️  {issues} 件の潜在的な問題が検出されました（メタデータを参照してください）")
    
    print("\nトレーニングの準備完了！ このデータを利用して以下のコマンドで学習を実施してください:")
    print(f"python train_patchtst.py --data-path {filepath}")


if __name__ == "__main__":
    main()