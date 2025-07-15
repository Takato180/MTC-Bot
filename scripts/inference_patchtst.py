#!/usr/bin/env python3
"""
PatchTST暗号通貨取引モデル推論スクリプト

このスクリプトは以下を処理します：
- 学習済みPatchTSTモデルを使用したリアルタイム予測
- ライブ取引システムとの統合
- 過去データでのバッチ予測
- パフォーマンス分析と可視化
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')

# srcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from strategy_service.patchtst.trainer import PatchTSTInference
from strategy_service.patchtst.data_loader import CryptoDataLoader
from strategy_service.patchtst.hybrid_strategy import HybridTradingStrategy


def parse_arguments():
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(description='暗号通貨取引用PatchTSTモデルの推論')
    
    # モデルに関する引数
    parser.add_argument('--model-path', type=str, required=True,
                        help='学習済みモデルのチェックポイントのパス')
    parser.add_argument('--preprocessor-path', type=str, required=True,
                        help='前処理器(pickleファイル)のパス')
    
    # データに関する引数
    parser.add_argument('--data-path', type=str,
                        help='バッチ予測用のOHLCVデータが保存されたCSVファイルのパス')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='取引対象のシンボル')
    
    # 推論モードに関する引数
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'batch', 'hybrid'],
                        help='推論モードの選択')
    
    # ハイブリッド戦略用の引数
    parser.add_argument('--ml-weight', type=float, default=0.6,
                        help='ハイブリッド戦略におけるML予測の重み')
    parser.add_argument('--rule-weight', type=float, default=0.4,
                        help='ハイブリッド戦略におけるルールベース信号の重み')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='取引実施のための最低信頼度閾値')
    
    # 出力に関する引数
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='予測結果の出力先ディレクトリ')
    parser.add_argument('--save-results', action='store_true',
                        help='予測結果をファイルに保存する')
    
    # 可視化に関する引数
    parser.add_argument('--plot-predictions', action='store_true',
                        help='予測結果のプロットを行う')
    parser.add_argument('--plot-strategy', action='store_true',
                        help='戦略のパフォーマンスをプロットする')
    
    return parser.parse_args()


def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    """
    学習済みモデルと前処理器を読み込む関数。
    
    引数:
        model_path: モデルのチェックポイントのパス
        preprocessor_path: 前処理器のpickleファイルのパス
    
    戻り値:
        (inference_engine, preprocessor) のタプル
    """
    print(f"{model_path} からモデルを読み込み中...")
    print(f"{preprocessor_path} から前処理器を読み込み中...")
    
    # 前処理器の読み込み
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # モデルの読み込み
    inference_engine = PatchTSTInference(model_path, preprocessor)
    
    print("モデルと前処理器の読み込みに成功しました！")
    
    return inference_engine, preprocessor


def single_prediction(inference_engine: PatchTSTInference, data_path: str):
    """
    最新データに対して1回限りの予測を実施する関数。
    
    引数:
        inference_engine: 学習済み推論エンジン
        data_path: データファイルのパス
    
    戻り値:
        予測結果の辞書
    """
    print("シングル予測を実施中...")
    
    # データの読み込み
    data_loader = CryptoDataLoader()
    df = data_loader.load_from_csv(data_path)
    
    # 予測の実施
    prediction_result = inference_engine.predict_next_prices(df)
    
    print("予測が完了しました！")
    print(f"予測価格の形状: {prediction_result['prediction'].shape}")
    print(f"予測期間: {prediction_result['pred_len']} タイムステップ")
    
    return prediction_result


def batch_prediction(inference_engine: PatchTSTInference, data_path: str, 
                     output_dir: str, save_results: bool = False):
    """
    複数の時点に対して予測を実施する関数。
    
    引数:
        inference_engine: 学習済み推論エンジン
        data_path: データファイルのパス
        output_dir: 出力先ディレクトリ
        save_results: 結果を保存するかどうかのフラグ
    
    戻り値:
        バッチ予測結果の辞書
    """
    print("バッチ予測を実施中...")
    
    # データの読み込み
    data_loader = CryptoDataLoader()
    df = data_loader.load_from_csv(data_path)
    
    # パラメータ設定
    seq_len = inference_engine.config.seq_len
    pred_len = inference_engine.config.pred_len
    
    # 複数時点での予測を実施（予測はオーバーラップする）
    predictions = []
    timestamps = []
    
    for i in range(seq_len, len(df) - pred_len, pred_len // 2):  # 予測区間が重複
        # 現在までのデータを抽出
        current_data = df.iloc[:i+1]
        
        try:
            # 予測の実施
            result = inference_engine.predict_next_prices(current_data)
            
            predictions.append(result['prediction'])
            timestamps.append(df.index[i] if hasattr(df, 'index') else i)
            
        except Exception as e:
            print(f"インデックス {i} でエラー: {e}")
            continue
    
    # 結果の結合
    batch_results = {
        'predictions': np.array(predictions),
        'timestamps': timestamps,
        'feature_names': inference_engine.preprocessor.scaler.feature_names_in_ if hasattr(inference_engine.preprocessor.scaler, 'feature_names_in_') else None,
        'pred_len': pred_len
    }
    
    print(f"バッチ予測が完了しました！予測回数: {len(predictions)}")
    
    # 結果をファイルに保存
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # 予測結果の保存
        predictions_path = os.path.join(output_dir, 'batch_predictions.npz')
        np.savez(predictions_path, 
                 predictions=batch_results['predictions'],
                 timestamps=batch_results['timestamps'])
        
        # メタデータの保存
        metadata = {
            'pred_len': batch_results['pred_len'],
            'n_predictions': len(predictions),
            'feature_names': batch_results['feature_names']
        }
        
        metadata_path = os.path.join(output_dir, 'batch_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"結果を {output_dir} に保存しました")
    
    return batch_results


def hybrid_strategy_inference(model_path: str, preprocessor, data_path: str,
                              ml_weight: float, rule_weight: float,
                              confidence_threshold: float, output_dir: str,
                              save_results: bool = False):
    """
    MLとルールベースのシグナルを組み合わせたハイブリッド戦略の推論を実施する関数。
    
    引数:
        model_path: 学習済みモデルのパス
        preprocessor: 前処理器
        data_path: データファイルのパス
        ml_weight: ML予測の重み
        rule_weight: ルールベースシグナルの重み
        confidence_threshold: 信頼度の閾値
        output_dir: 結果出力先ディレクトリ
        save_results: 結果を保存するかのフラグ
    
    戻り値:
        戦略の結果の辞書
    """
    print("ハイブリッド戦略推論を実施中...")
    
    # ハイブリッド戦略の作成
    strategy = HybridTradingStrategy(
        model_path=model_path,
        preprocessor=preprocessor,
        ml_weight=ml_weight,
        rule_weight=rule_weight,
        confidence_threshold=confidence_threshold
    )
    
    # データの読み込み
    data_loader = CryptoDataLoader()
    df = data_loader.load_from_csv(data_path)
    
    # バックテストの実施
    print("戦略のバックテストを実施中...")
    backtest_results = strategy.backtest_strategy(df)
    
    print("バックテストが完了しました！")
    print(f"総リターン: {backtest_results['total_return']:.2%}")
    print(f"取引回数: {backtest_results['total_trades']}")
    print(f"勝率: {backtest_results['win_rate']:.2%}")
    print(f"シャープレシオ: {backtest_results['sharpe_ratio']:.3f}")
    print(f"最大ドローダウン: {backtest_results['max_drawdown']:.2%}")
    
    # 結果をファイルに保存
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # バックテスト結果の保存
        results_path = os.path.join(output_dir, 'hybrid_strategy_results.json')
        # JSONシリアライズのためにnumpyの型をPythonの型に変換
        json_results = {
            'total_return': float(backtest_results['total_return']),
            'total_trades': int(backtest_results['total_trades']),
            'winning_trades': int(backtest_results['winning_trades']),
            'win_rate': float(backtest_results['win_rate']),
            'final_portfolio_value': float(backtest_results['final_portfolio_value']),
            'max_drawdown': float(backtest_results['max_drawdown']),
            'sharpe_ratio': float(backtest_results['sharpe_ratio']),
            'strategy_config': {
                'ml_weight': ml_weight,
                'rule_weight': rule_weight,
                'confidence_threshold': confidence_threshold
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 取引履歴が存在する場合、CSVとして保存
        if backtest_results['trade_history']:
            trade_df = pd.DataFrame(backtest_results['trade_history'])
            trade_path = os.path.join(output_dir, 'trade_history.csv')
            trade_df.to_csv(trade_path, index=False)
        
        print(f"戦略結果を {output_dir} に保存しました")
    
    return backtest_results


def plot_predictions(predictions: np.ndarray, timestamps, actual_data: pd.DataFrame = None,
                     output_dir: str = None):
    """
    予測結果をプロットする関数。
    
    引数:
        predictions: 予測値の配列
        timestamps: 時間スタンプの配列
        actual_data: 実際のデータ（比較用）
        output_dir: プロット画像の出力先ディレクトリ
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8')
        
        # グラフ作成用のフィギュアとサブプロットの設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PatchTST 予測分析', fontsize=16)
        
        # プロット1: 時系列に沿った予測値の表示
        ax1 = axes[0, 0]
        if len(predictions.shape) == 3:  # (予測回数, 予測期間, 特徴量数)
            # 終値予測を表示（先頭の特徴量が終値と仮定）
            close_predictions = predictions[:, :, 0]
            
            for i in range(min(5, len(close_predictions))):
                ax1.plot(close_predictions[i], alpha=0.7, label=f'予測 {i+1}')
        
        ax1.set_title('時間軸に沿った価格予測')
        ax1.set_xlabel('タイムステップ')
        ax1.set_ylabel('予測価格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # プロット2: 予測値の分布
        ax2 = axes[0, 1]
        if len(predictions.shape) == 3:
            pred_flat = predictions[:, :, 0].flatten()
            ax2.hist(pred_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax2.set_title('予測値の分布')
        ax2.set_xlabel('予測価格')
        ax2.set_ylabel('頻度')
        ax2.grid(True, alpha=0.3)
        
        # プロット3: 予測の不確かさ（標準偏差）
        ax3 = axes[1, 0]
        if len(predictions.shape) == 3:
            pred_std = np.std(predictions[:, :, 0], axis=1)
            ax3.plot(pred_std, marker='o', markersize=3)
        
        ax3.set_title('予測の不確かさ')
        ax3.set_xlabel('予測インデックス')
        ax3.set_ylabel('標準偏差')
        ax3.grid(True, alpha=0.3)
        
        # プロット4: 特徴量の重要度（複数特徴量がある場合）
        ax4 = axes[1, 1]
        if len(predictions.shape) == 3 and predictions.shape[2] > 1:
            feature_importance = np.mean(np.abs(predictions), axis=(0, 1))
            ax4.bar(range(len(feature_importance)), feature_importance)
            ax4.set_title('特徴量の重要度')
            ax4.set_xlabel('特徴量インデックス')
            ax4.set_ylabel('平均絶対予測値')
        else:
            ax4.text(0.5, 0.5, '単一特徴量の\n予測結果',
                     ha='center', va='center', transform=ax4.transAxes,
                     fontsize=12)
            ax4.set_title('特徴量解析')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'predictions_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"予測プロットを {plot_path} に保存しました")
        
        plt.show()
        
    except ImportError:
        print("Matplotlibが見つかりません。以下でインストールしてください: pip install matplotlib seaborn")
    except Exception as e:
        print(f"プロット作成時にエラーが発生しました: {e}")


def plot_strategy_performance(backtest_results: dict, output_dir: str = None):
    """
    戦略のパフォーマンス結果をプロットする関数。
    
    引数:
        backtest_results: バックテスト結果の辞書
        output_dir: プロット画像の出力先ディレクトリ
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8')
        
        if not backtest_results['trade_history']:
            print("プロット用の取引履歴がありません")
            return
        
        # 取引履歴からDataFrameを作成
        trade_df = pd.DataFrame(backtest_results['trade_history'])
        
        # フィギュアとサブプロットの設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ハイブリッド戦略のパフォーマンス分析', fontsize=16)
        
        # プロット1: 時系列でのポートフォリオの価値の推移
        ax1 = axes[0, 0]
        ax1.plot(trade_df['timestamp'], trade_df['portfolio_value'], linewidth=2)
        ax1.set_title('時系列でのポートフォリオ価値')
        ax1.set_xlabel('時間')
        ax1.set_ylabel('ポートフォリオ価値')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # プロット2: 時系列でのポジションサイズの推移
        ax2 = axes[0, 1]
        ax2.plot(trade_df['timestamp'], trade_df['position'], linewidth=2, color='orange')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('時系列でのポジションサイズ')
        ax2.set_xlabel('時間')
        ax2.set_ylabel('ポジションサイズ')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # プロット3: 取引サイズの分布
        ax3 = axes[1, 0]
        ax3.hist(trade_df['trade_size'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title('取引サイズの分布')
        ax3.set_xlabel('取引サイズ')
        ax3.set_ylabel('頻度')
        ax3.grid(True, alpha=0.3)
        
        # プロット4: 時系列でのシグナル信頼度の推移
        ax4 = axes[1, 1]
        ax4.plot(trade_df['timestamp'], trade_df['confidence'], 
                 marker='o', markersize=3, linewidth=1, color='red')
        ax4.set_title('時系列での信頼度')
        ax4.set_xlabel('時間')
        ax4.set_ylabel('信頼度')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'strategy_performance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"戦略パフォーマンスのプロットを {plot_path} に保存しました")
        
        plt.show()
        
    except ImportError:
        print("Matplotlibが見つかりません。以下でインストールしてください: pip install matplotlib seaborn")
    except Exception as e:
        print(f"戦略プロット作成時にエラーが発生しました: {e}")


def main():
    """メインの推論処理を実施する関数"""
    args = parse_arguments()
    
    # 出力先ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルと前処理器の読み込み
    inference_engine, preprocessor = load_model_and_preprocessor(
        args.model_path, args.preprocessor_path
    )
    
    # 指定されたモードに応じた推論の実施
    if args.mode == 'single':
        if not args.data_path:
            raise ValueError("シングル予測モードではデータパスが必要です")
        
        result = single_prediction(inference_engine, args.data_path)
        
        if args.save_results:
            result_path = os.path.join(args.output_dir, 'single_prediction.json')
            with open(result_path, 'w') as f:
                json.dump({
                    'prediction': result['prediction'].tolist(),
                    'feature_names': result['feature_names'],
                    'pred_len': result['pred_len']
                }, f, indent=2)
            print(f"シングル予測結果を {result_path} に保存しました")
    
    elif args.mode == 'batch':
        if not args.data_path:
            raise ValueError("バッチ予測モードではデータパスが必要です")
        
        results = batch_prediction(
            inference_engine, args.data_path, args.output_dir, args.save_results
        )
        
        if args.plot_predictions:
            plot_predictions(results['predictions'], results['timestamps'], 
                             output_dir=args.output_dir)
    
    elif args.mode == 'hybrid':
        if not args.data_path:
            raise ValueError("ハイブリッド戦略モードではデータパスが必要です")
        
        results = hybrid_strategy_inference(
            args.model_path, preprocessor, args.data_path,
            args.ml_weight, args.rule_weight, args.confidence_threshold,
            args.output_dir, args.save_results
        )
        
        if args.plot_strategy:
            plot_strategy_performance(results, args.output_dir)
    
    print("推論処理が正常に完了しました！")


if __name__ == "__main__":
    main()