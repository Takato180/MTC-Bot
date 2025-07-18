#!/usr/bin/env python3
"""
正規PatchTST学習スクリプト
GPU最適化・進捗監視・MLflow統合
"""

import sys
import os
import argparse
from pathlib import Path
import torch
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
    parser = argparse.ArgumentParser(description='正規PatchTST学習')
    
    # データ設定
    parser.add_argument('--data-path', type=str, default='data/BTCUSDT_60m_clean.csv',
                       help='データファイルパス')
    parser.add_argument('--experiment-name', type=str, default='official_patchtst',
                       help='実験名')
    
    # モデル設定
    parser.add_argument('--seq-len', type=int, default=336,
                       help='入力系列長')
    parser.add_argument('--pred-len', type=int, default=96,
                       help='予測長')
    parser.add_argument('--d-model', type=int, default=512,
                       help='モデル次元')
    parser.add_argument('--n-heads', type=int, default=8,
                       help='アテンションヘッド数')
    parser.add_argument('--e-layers', type=int, default=3,
                       help='エンコーダ層数')
    parser.add_argument('--patch-len', type=int, default=16,
                       help='パッチ長')
    parser.add_argument('--stride', type=int, default=8,
                       help='ストライド')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='ドロップアウト率')
    
    # 学習設定
    parser.add_argument('--epochs', type=int, default=100,
                       help='エポック数')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='バッチサイズ')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学習率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='重み減衰')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # GPU設定
    parser.add_argument('--device', type=str, default='auto',
                       help='デバイス (auto/cuda/cpu)')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='混合精度学習')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='データローダーワーカー数')
    
    return parser.parse_args()

def check_gpu_status():
    """GPU状況確認"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_cap = torch.cuda.get_device_capability(0)
        
        print(f"GPU検出: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f}GB")
        print(f"Compute Capability: sm_{compute_cap[0]}{compute_cap[1]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        
        # RTXシリーズ最適化
        if "RTX" in gpu_name:
            if "50" in gpu_name:
                batch_size = 32  # RTX 50シリーズ
                d_model = 256
                print("RTX 50シリーズ最適化設定")
            elif "40" in gpu_name:
                batch_size = 64  # RTX 40シリーズ
                d_model = 512
                print("RTX 40シリーズ最適化設定")
            else:
                batch_size = 32  # その他RTX
                d_model = 256
                print("RTX汎用最適化設定")
        else:
            batch_size = 32
            d_model = 256
            print("汎用GPU設定")
            
        return device, batch_size, d_model
    else:
        print("GPU未検出 - CPUで実行")
        return torch.device('cpu'), 16, 128

def load_and_preprocess_data(data_path: str, args):
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
    
    # データローダー作成
    train_loader, val_loader, test_loader = create_data_loaders(
        data=processed_data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        test_size=0.15,
        validation_size=0.15,
        num_workers=args.num_workers
    )
    
    print(f"学習バッチ数: {len(train_loader)}")
    print(f"検証バッチ数: {len(val_loader)}")
    print(f"テストバッチ数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, len(feature_names)

def create_model_and_trainer(args, n_vars: int, device: torch.device):
    """モデル・トレーナー作成"""
    print("正規PatchTSTモデル作成中...")
    
    # 設定
    config = OfficialPatchTSTConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=n_vars,
        dec_in=n_vars,
        c_out=n_vars,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        patch_len=args.patch_len,
        stride=args.stride,
        dropout=args.dropout
    )
    
    # モデル作成
    model = OfficialPatchTST(config)
    
    # パラメータ数表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    
    # トレーナー作成
    trainer = OfficialPatchTSTTrainer(
        model=model,
        config=config,
        device=args.device,
        mixed_precision=args.mixed_precision,
        experiment_name=args.experiment_name
    )
    
    return trainer

def main():
    """メイン処理"""
    args = parse_arguments()
    
    print("="*70)
    print("正規PatchTST学習システム")
    print("="*70)
    print(f"実験名: {args.experiment_name}")
    print(f"データ: {args.data_path}")
    print(f"エポック: {args.epochs}")
    print(f"バッチサイズ: {args.batch_size}")
    print()
    
    # GPU状況確認
    device, optimal_batch_size, optimal_d_model = check_gpu_status()
    
    # 最適化設定の上書き
    if args.device == 'auto':
        args.device = device
    if args.batch_size == 64:  # デフォルト値の場合
        args.batch_size = optimal_batch_size
    if args.d_model == 512:  # デフォルト値の場合
        args.d_model = optimal_d_model
    
    print(f"最適化後 - バッチサイズ: {args.batch_size}, モデル次元: {args.d_model}")
    print()
    
    # データ読み込み・前処理
    train_loader, val_loader, test_loader, n_vars = load_and_preprocess_data(args.data_path, args)
    
    # モデル・トレーナー作成
    trainer = create_model_and_trainer(args, n_vars, device)
    
    # 学習実行
    print("="*70)
    print("学習開始")
    print("="*70)
    
    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    print("="*70)
    print("学習完了")
    print("="*70)
    print("結果確認方法:")
    print(f"TensorBoard: tensorboard --logdir=models/{args.experiment_name}/logs --port=6006")
    print(f"MLflow UI: mlflow ui --port=5000")
    print(f"モデル: models/{args.experiment_name}/checkpoints/best_model.pth")
    
    return results

if __name__ == "__main__":
    main()