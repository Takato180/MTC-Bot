#!/usr/bin/env python3
"""
正規PatchTST推論・バックテストスクリプト
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# パス追加
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from strategy_service.patchtst.official_model import OfficialPatchTST, OfficialPatchTSTConfig
from strategy_service.patchtst.official_trainer import OfficialPatchTSTTrainer
from strategy_service.patchtst.data_loader import (
    CryptoDataLoader, CryptoDataPreprocessor, create_data_loaders
)

def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(description='正規PatchTST推論・バックテスト')
    
    # モデル・データ設定
    parser.add_argument('--model-path', type=str, required=True,
                       help='モデルファイルパス')
    parser.add_argument('--data-path', type=str, required=True,
                       help='データファイルパス')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch', 'backtest'],
                       help='推論モード')
    
    # 推論設定
    parser.add_argument('--device', type=str, default='auto',
                       help='デバイス (auto/cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='バッチサイズ')
    parser.add_argument('--plot-predictions', action='store_true',
                       help='予測結果をプロット')
    parser.add_argument('--save-results', action='store_true',
                       help='結果を保存')
    
    # バックテスト設定
    parser.add_argument('--initial-capital', type=float, default=1000.0,
                       help='初期資本（USD）')
    parser.add_argument('--trade-fee', type=float, default=0.001,
                       help='取引手数料率')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='取引信頼度閾値')
    
    return parser.parse_args()

def load_model(model_path: str, device: torch.device):
    """モデル読み込み"""
    print(f"モデル読み込み: {model_path}")
    
    # チェックポイント読み込み
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # モデル作成
    model = OfficialPatchTST(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"モデル読み込み完了")
    print(f"設定: seq_len={config.seq_len}, pred_len={config.pred_len}")
    print(f"モデル次元: {config.d_model}, ヘッド数: {config.n_heads}")
    
    return model, config

def load_and_preprocess_data(data_path: str, config):
    """データ読み込み・前処理"""
    print(f"データ読み込み: {data_path}")
    
    # データローダー
    data_loader = CryptoDataLoader()
    df = data_loader.load_from_csv(data_path)
    
    print(f"データサイズ: {len(df):,}行")
    print(f"期間: {df.index[0]} ～ {df.index[-1]}")
    
    # 前処理
    preprocessor = CryptoDataPreprocessor(
        scaler_type='standard',
        feature_engineering=True
    )
    
    processed_data, feature_names = preprocessor.preprocess(df, fit=True)
    
    print(f"特徴量数: {len(feature_names)}")
    print(f"前処理後データ形状: {processed_data.shape}")
    
    return processed_data, feature_names, df, preprocessor

def single_prediction(model, data, config, device):
    """単一予測"""
    print("単一予測実行中...")
    
    # 最新データで予測
    seq_len = config.seq_len
    if len(data) < seq_len:
        raise ValueError(f"データが不足しています。最低{seq_len}行必要です。")
    
    # 最新seq_len分のデータを使用
    input_data = data[-seq_len:].reshape(1, seq_len, -1)
    input_tensor = torch.FloatTensor(input_data).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = outputs['output'].cpu().numpy()
        price_pred = outputs['price'].cpu().numpy()
        direction_pred = outputs['direction'].cpu().numpy()
    
    print("予測完了")
    print(f"予測形状: {prediction.shape}")
    print(f"価格予測: {price_pred[0, 0]:.4f}")
    
    # 方向予測の解釈
    direction_labels = ['下降', '横ばい', '上昇']
    direction_idx = np.argmax(direction_pred[0])
    direction_confidence = np.max(direction_pred[0])
    
    print(f"方向予測: {direction_labels[direction_idx]} (信頼度: {direction_confidence:.4f})")
    
    return {
        'prediction': prediction[0],
        'price_prediction': price_pred[0, 0],
        'direction_prediction': direction_labels[direction_idx],
        'direction_confidence': direction_confidence
    }

def batch_prediction(model, data, config, device, batch_size=32):
    """バッチ予測"""
    print("バッチ予測実行中...")
    
    seq_len = config.seq_len
    pred_len = config.pred_len
    
    # 予測可能な位置を計算
    total_samples = len(data) - seq_len - pred_len + 1
    if total_samples <= 0:
        raise ValueError("予測に十分なデータがありません")
    
    predictions = []
    price_predictions = []
    direction_predictions = []
    
    # バッチ処理
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch_inputs = []
        
        for j in range(i, batch_end):
            input_seq = data[j:j+seq_len]
            batch_inputs.append(input_seq)
        
        batch_tensor = torch.FloatTensor(np.array(batch_inputs)).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            batch_pred = outputs['output'].cpu().numpy()
            batch_price = outputs['price'].cpu().numpy()
            batch_direction = outputs['direction'].cpu().numpy()
            
            predictions.extend(batch_pred)
            price_predictions.extend(batch_price)
            direction_predictions.extend(batch_direction)
    
    print(f"バッチ予測完了: {len(predictions)}件")
    
    return {
        'predictions': np.array(predictions),
        'price_predictions': np.array(price_predictions),
        'direction_predictions': np.array(direction_predictions)
    }

def simple_backtest(predictions, original_data, config, initial_capital=1000.0, trade_fee=0.001, confidence_threshold=0.6):
    """簡単なバックテスト"""
    print("バックテスト実行中...")
    
    seq_len = config.seq_len
    pred_len = config.pred_len
    
    # 価格データ抽出（close価格を仮定）
    prices = original_data['close'].values
    
    # 取引シミュレーション
    capital = initial_capital
    position = 0  # 0: ノーポジション, 1: ロング, -1: ショート
    trades = []
    portfolio_values = []
    
    direction_predictions = predictions['direction_predictions']
    
    for i in range(len(direction_predictions)):
        current_idx = seq_len + i
        if current_idx >= len(prices):
            break
            
        current_price = prices[current_idx]
        direction_prob = np.max(direction_predictions[i])
        predicted_direction = np.argmax(direction_predictions[i])
        
        # 信頼度チェック
        if direction_prob < confidence_threshold:
            continue
        
        # 取引判定
        if predicted_direction == 2 and position != 1:  # 上昇予測、ロング
            if position == -1:  # ショートポジション決済
                profit = (prices[current_idx-1] - current_price) * abs(position)
                capital += profit * (1 - trade_fee)
            
            # ロングポジション開設
            position = 1
            trades.append({
                'timestamp': current_idx,
                'action': 'BUY',
                'price': current_price,
                'confidence': direction_prob
            })
            
        elif predicted_direction == 0 and position != -1:  # 下降予測、ショート
            if position == 1:  # ロングポジション決済
                profit = (current_price - prices[current_idx-1]) * abs(position)
                capital += profit * (1 - trade_fee)
            
            # ショートポジション開設
            position = -1
            trades.append({
                'timestamp': current_idx,
                'action': 'SELL',
                'price': current_price,
                'confidence': direction_prob
            })
        
        # ポートフォリオ価値計算
        if position != 0:
            unrealized_pnl = 0
            if position == 1:
                unrealized_pnl = (current_price - prices[current_idx-1]) * abs(position)
            else:
                unrealized_pnl = (prices[current_idx-1] - current_price) * abs(position)
            portfolio_value = capital + unrealized_pnl
        else:
            portfolio_value = capital
        
        portfolio_values.append(portfolio_value)
    
    # 最終ポジション決済
    if position != 0:
        final_price = prices[-1]
        if position == 1:
            profit = (final_price - prices[-2]) * abs(position)
        else:
            profit = (prices[-2] - final_price) * abs(position)
        capital += profit * (1 - trade_fee)
    
    # 結果計算
    total_return = (capital - initial_capital) / initial_capital * 100
    total_trades = len(trades)
    
    print(f"バックテスト完了")
    print(f"初期資本: ${initial_capital:.2f}")
    print(f"最終資本: ${capital:.2f}")
    print(f"総収益率: {total_return:.2f}%")
    print(f"総取引数: {total_trades}")
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def plot_predictions(predictions, original_data, config, save_path=None):
    """予測結果をプロット"""
    print("予測結果プロット中...")
    
    seq_len = config.seq_len
    pred_len = config.pred_len
    
    # 実際の価格データ
    actual_prices = original_data['close'].values
    
    if 'predictions' in predictions:
        # バッチ予測の場合
        pred_data = predictions['predictions']
        price_preds = predictions['price_predictions']
        
        # プロット作成
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 価格推移と予測
        axes[0].plot(actual_prices, label='実際の価格', alpha=0.7)
        
        # 予測価格をプロット
        for i in range(0, len(price_preds), 50):  # 50件おきにプロット
            pred_idx = seq_len + i
            if pred_idx < len(actual_prices):
                axes[0].axvline(x=pred_idx, color='red', alpha=0.3, linestyle='--')
                axes[0].scatter(pred_idx, price_preds[i], color='red', s=20, alpha=0.7)
        
        axes[0].set_title('実際の価格 vs 予測価格')
        axes[0].set_xlabel('時間')
        axes[0].set_ylabel('価格')
        axes[0].legend()
        axes[0].grid(True)
        
        # 方向予測の精度
        direction_preds = predictions['direction_predictions']
        direction_accuracy = []
        
        for i in range(len(direction_preds)):
            pred_idx = seq_len + i
            if pred_idx + pred_len < len(actual_prices):
                actual_change = actual_prices[pred_idx + pred_len] - actual_prices[pred_idx]
                predicted_direction = np.argmax(direction_preds[i])
                
                if actual_change > 0 and predicted_direction == 2:  # 上昇予測が正解
                    direction_accuracy.append(1)
                elif actual_change < 0 and predicted_direction == 0:  # 下降予測が正解
                    direction_accuracy.append(1)
                elif abs(actual_change) < 0.01 and predicted_direction == 1:  # 横ばい予測が正解
                    direction_accuracy.append(1)
                else:
                    direction_accuracy.append(0)
        
        # 移動平均精度
        window_size = 50
        accuracy_ma = pd.Series(direction_accuracy).rolling(window=window_size).mean()
        
        axes[1].plot(accuracy_ma, label=f'方向予測精度 (MA{window_size})')
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='ランダム予測')
        axes[1].set_title('方向予測精度の推移')
        axes[1].set_xlabel('予測数')
        axes[1].set_ylabel('精度')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"プロット保存: {save_path}")
        
        plt.show()
    
    else:
        # 単一予測の場合
        print("単一予測のため、詳細プロットは省略")

def main():
    """メイン処理"""
    args = parse_arguments()
    
    print("="*70)
    print("正規PatchTST推論・バックテストシステム")
    print("="*70)
    print(f"モデル: {args.model_path}")
    print(f"データ: {args.data_path}")
    print(f"モード: {args.mode}")
    print()
    
    # デバイス設定
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用デバイス: {device}")
    
    # モデル読み込み
    model, config = load_model(args.model_path, device)
    
    # データ読み込み・前処理
    processed_data, feature_names, original_data, preprocessor = load_and_preprocess_data(args.data_path, config)
    
    # 推論実行
    print("="*70)
    print("推論実行")
    print("="*70)
    
    if args.mode == 'single':
        # 単一予測
        results = single_prediction(model, processed_data, config, device)
        
        print("\n予測結果:")
        print(f"価格予測: {results['price_prediction']:.4f}")
        print(f"方向予測: {results['direction_prediction']}")
        print(f"信頼度: {results['direction_confidence']:.4f}")
        
    elif args.mode == 'batch':
        # バッチ予測
        results = batch_prediction(model, processed_data, config, device, args.batch_size)
        
        print("\nバッチ予測結果:")
        print(f"予測数: {len(results['predictions'])}")
        print(f"平均価格予測: {np.mean(results['price_predictions']):.4f}")
        
        # 方向予測の分布
        direction_counts = np.bincount(np.argmax(results['direction_predictions'], axis=1))
        direction_labels = ['下降', '横ばい', '上昇']
        
        print("\n方向予測の分布:")
        for i, count in enumerate(direction_counts):
            if i < len(direction_labels):
                print(f"{direction_labels[i]}: {count}件 ({count/len(results['predictions'])*100:.1f}%)")
        
        # プロット
        if args.plot_predictions:
            plot_predictions(results, original_data, config)
        
        # バックテスト
        if args.mode == 'batch':
            print("\n" + "="*70)
            print("バックテスト実行")
            print("="*70)
            
            backtest_results = simple_backtest(
                results, original_data, config,
                args.initial_capital, args.trade_fee, args.confidence_threshold
            )
            
            # 結果保存
            if args.save_results:
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 予測結果保存
                pred_file = results_dir / f"predictions_{timestamp}.npz"
                np.savez(pred_file, **results)
                
                # バックテスト結果保存
                backtest_file = results_dir / f"backtest_{timestamp}.json"
                import json
                with open(backtest_file, 'w') as f:
                    json.dump(backtest_results, f, indent=2, default=str)
                
                print(f"結果保存: {pred_file}, {backtest_file}")
    
    print("\n" + "="*70)
    print("推論完了")
    print("="*70)

if __name__ == "__main__":
    main()