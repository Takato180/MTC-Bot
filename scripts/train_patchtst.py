#!/usr/bin/env python3
"""
PatchTST暗号通貨取引モデル学習スクリプト

このスクリプトは以下を処理します：
- データ読み込みと前処理
- GPU加速によるモデル学習
- モデル評価と指標計算
- モデルチェックポイント保存
- ハイパーパラメータ最適化
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from strategy_service.patchtst.model import PatchTST, PatchTSTConfig
from strategy_service.patchtst.trainer import PatchTSTTrainer
from strategy_service.patchtst.data_loader import (
    CryptoDataLoader, CryptoDataPreprocessor, create_data_loaders, CRYPTO_DATA_CONFIG
)


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='暗号通貨取引用PatchTSTモデルの学習')
    
    # データ関連引数
    parser.add_argument('--data-path', type=str, required=True,
                       help='OHLCVデータを含むCSVファイルのパス')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='取引シンボル')
    
    # モデル関連引数
    parser.add_argument('--seq-len', type=int, default=336,
                       help='入力系列長 (デフォルト: 336 - 14日分の時間足)')
    parser.add_argument('--pred-len', type=int, default=96,
                       help='予測長 (デフォルト: 96 - 4日分の時間足)')
    parser.add_argument('--patch-len', type=int, default=16,
                       help='パッチ長')
    parser.add_argument('--stride', type=int, default=8,
                       help='パッチストライド')
    parser.add_argument('--d-model', type=int, default=256,
                       help='モデル次元 (RTX 4060 Ti最適化: 256推奨)')
    parser.add_argument('--n-heads', type=int, default=16,
                       help='アテンションヘッド数 (16コア最適化)')
    parser.add_argument('--n-layers', type=int, default=8,
                       help='Transformer層数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='ドロップアウト率')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='バッチサイズ（RTX 4060 Ti最適化: 128推奨）')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Data processing arguments
    parser.add_argument('--scaler-type', type=str, default='standard',
                       choices=['standard', 'minmax'],
                       help='Type of data scaler')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set size')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models and logs')
    parser.add_argument('--experiment-name', type=str, default='patchtst_btc',
                       help='Experiment name')
    
    # GPU arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=16,
                       help='データローダーのワーカー数（推奨: CPUスレッド数）')
    
    # Optimization arguments
    parser.add_argument('--optimize-hyperparams', action='store_true',
                       help='Perform hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials')
    
    return parser.parse_args()


def load_and_preprocess_data(data_path: str, 
                           scaler_type: str = 'standard',
                           feature_engineering: bool = True) -> tuple:
    """
    暗号通貨データを読み込んで前処理します。
    
    Args:
        data_path: データファイルのパス
        scaler_type: 使用するスケーラーの種類
        feature_engineering: 追加特徴量を作成するかどうか
    
    Returns:
        (前処理済みデータ, 特徴量名, 前処理器)のタプル
    """
    print(f"{data_path}からデータを読み込み中...")
    
    # データ読み込み
    data_loader = CryptoDataLoader()
    df = data_loader.load_from_csv(data_path)
    
    print(f"{len(df)}サンプルを読み込みました")
    print(f"日付範囲: {df.index[0]} ～ {df.index[-1]}")
    
    # 前処理器の初期化
    preprocessor = CryptoDataPreprocessor(
        scaler_type=scaler_type,
        feature_engineering=feature_engineering
    )
    
    # データ前処理
    processed_data, feature_names = preprocessor.preprocess(df, fit=True)
    
    print(f"{len(feature_names)}個の特徴量を作成しました")
    print(f"前処理済みデータ形状: {processed_data.shape}")
    
    return processed_data, feature_names, preprocessor


def create_model_config(args, n_vars: int) -> PatchTSTConfig:
    """
    引数からモデル設定を作成
    
    Args:
        args: コマンドライン引数
        n_vars: 変数の数
    
    Returns:
        PatchTSTConfig インスタンス
    """
    return PatchTSTConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_vars=n_vars,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )


def train_model(args, data: np.ndarray, config: PatchTSTConfig, 
                output_dir: str, preprocessor: CryptoDataPreprocessor):
    """
    PatchTSTモデルの学習を実行
    
    Args:
        args: コマンドライン引数
        data: 処理済みデータ
        config: モデル設定
        output_dir: 出力ディレクトリ
        preprocessor: データ前処理器
    """
    print("データローダーを作成中...")
    
    # データローダーの作成
    train_loader, val_loader, test_loader = create_data_loaders(
        data=data,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=args.batch_size,
        test_size=args.test_size,
        validation_size=args.val_size,
        num_workers=args.num_workers
    )
    
    print(f"学習バッチ数: {len(train_loader)}")
    print(f"検証バッチ数: {len(val_loader)}")
    print(f"テストバッチ数: {len(test_loader)}")
    
    # モデルの作成
    print("モデルを作成中...")
    model = PatchTST(
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        n_vars=config.n_vars,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout
    )
    
    # パラメータ数のカウント
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    
    # トレーナーの作成
    trainer = PatchTSTTrainer(
        model=model,
        config=config,
        device=args.device,
        log_dir=os.path.join(output_dir, 'logs'),
        checkpoint_dir=os.path.join(output_dir, 'checkpoints')
    )
    
    # モデルの学習
    print("学習を開始します...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_best=True
    )
    
    # テストセットでの評価
    print("テストセットで評価中...")
    predictions, targets = trainer.predict(test_loader)
    test_metrics = trainer.calculate_metrics(targets, predictions)
    
    print("テスト結果:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")
    
    # 結果の保存
    results = {
        'config': config.to_dict(),
        'train_history': history,
        'test_metrics': test_metrics,
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    }
    
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 前処理器の保存
    import pickle
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"学習が完了しました。結果は {output_dir} に保存されました")


def optimize_hyperparameters(args, data: np.ndarray, n_vars: int, output_dir: str):
    """
    Optunaを使用したハイパーパラメータ最適化
    
    Args:
        args: コマンドライン引数
        data: 処理済みデータ
        n_vars: 変数の数
        output_dir: 出力ディレクトリ
    """
    import optuna
    
    def objective(trial):
        # ハイパーパラメータの推奨
        config = PatchTSTConfig(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            n_vars=n_vars,
            patch_len=trial.suggest_int('patch_len', 8, 32),
            stride=trial.suggest_int('stride', 4, 16),
            d_model=trial.suggest_categorical('d_model', [64, 128, 256]),
            n_heads=trial.suggest_categorical('n_heads', [4, 8, 16]),
            n_layers=trial.suggest_int('n_layers', 3, 8),
            dropout=trial.suggest_float('dropout', 0.1, 0.3)
        )
        
        # データローダーの作成
        train_loader, val_loader, _ = create_data_loaders(
            data=data,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            batch_size=args.batch_size,
            test_size=args.test_size,
            validation_size=args.val_size,
            num_workers=args.num_workers
        )
        
        # モデルの作成
        model = PatchTST(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            n_vars=config.n_vars,
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout
        )
        
        # トレーナーの作成
        trainer = PatchTSTTrainer(
            model=model,
            config=config,
            device=args.device,
            log_dir=os.path.join(output_dir, 'optuna_logs', f'trial_{trial.number}'),
            checkpoint_dir=os.path.join(output_dir, 'optuna_checkpoints', f'trial_{trial.number}')
        )
        
        try:
            # 最適化のためにエポック数を少なくして学習
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=min(args.epochs, 30),  # 最適化のためにエポック数を制限
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                patience=10,
                save_best=False
            )
            
            # 検証損失を返す
            return min(history['val_loss'])
        
        except Exception as e:
            print(f"試行 {trial.number} が失敗しました: {e}")
            return float('inf')
    
    # 最適化の実行
    print(f"{args.n_trials}回の試行でハイパーパラメータ最適化を開始します...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    
    print("最適化が完了しました！")
    print(f"最適試行: {study.best_trial.number}")
    print(f"最適検証損失: {study.best_value:.6f}")
    print("最適パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 最適化結果の保存
    optuna_results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': args.n_trials
    }
    
    optuna_path = os.path.join(output_dir, 'optuna_results.json')
    with open(optuna_path, 'w') as f:
        json.dump(optuna_results, f, indent=2)
    
    return study.best_params


def main():
    """メインの学習関数"""
    args = parse_arguments()
    
    # 出力ディレクトリの作成
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 引数の保存
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # データの読み込みと前処理
    data, feature_names, preprocessor = load_and_preprocess_data(
        args.data_path, args.scaler_type, feature_engineering=True
    )
    
    n_vars = len(feature_names)
    
    # ハイパーパラメータ最適化
    if args.optimize_hyperparams:
        best_params = optimize_hyperparameters(args, data, n_vars, output_dir)
        
        # 最適パラメータで引数を更新
        for key, value in best_params.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # モデル設定の作成
    config = create_model_config(args, n_vars)
    
    # モデルの学習
    train_model(args, data, config, output_dir, preprocessor)
    
    print("学習パイプラインが正常に完了しました！")


if __name__ == "__main__":
    main()