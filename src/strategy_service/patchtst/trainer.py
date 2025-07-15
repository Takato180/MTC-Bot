"""
PatchTST暗号通貨取引モデルの学習および推論パイプライン

このモジュールは以下を提供します：
- 早期停止と学習率スケジューリングを伴うモデル学習
- モデル評価と指標計算
- リアルタイム予測用の推論パイプライン
- モデルチェックポイントと再開
- GPU加速サポート
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
import json
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .model import PatchTST, PatchTSTConfig
from .data_loader import CryptoDataPreprocessor


class EarlyStopping:
    """
    過学習を防ぐための早期停止ユーティリティ
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: 改善がないまま学習を停止するまでのエポック数
            min_delta: 改善とみなすための監視量の最小変化量
            restore_best_weights: 最適エポックからモデル重みを復元するかどうか
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        学習を停止すべきかチェック
        
        Args:
            val_loss: 現在の検証損失
            model: 重みを保存する可能性のあるモデル
        
        Returns:
            学習を停止すべきならTrue、継続すべきならFalse
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class PatchTSTTrainer:
    """
    PatchTSTモデル用トレーナークラス
    """
    
    def __init__(self, 
                 model: PatchTST,
                 config: PatchTSTConfig,
                 device: str = 'auto',
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Args:
            model: PatchTSTモデルインスタンス
            config: モデル設定
            device: 使用するデバイス ('auto', 'cpu', 'cuda')
            log_dir: tensorboardログ用ディレクトリ
            checkpoint_dir: モデルチェックポイント用ディレクトリ
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # デバイスの設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # ディレクトリの作成
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # tensorboardライターの初期化
        self.writer = SummaryWriter(log_dir)
        
        # 学習履歴
        self.train_history = []
        self.val_history = []
        
        print(f"Training on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        評価指標を計算
        
        Args:
            y_true: 真の値
            y_pred: 予測値
        
        Returns:
            指標の辞書
        """
        # 指標計算のために配列を平坦化
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # NaN値を除去
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) == 0:
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train_epoch(self, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """
        1エポックの学習を実行
        
        Args:
            train_loader: 学習データローダー
            optimizer: 最適化器
            criterion: 損失関数
        
        Returns:
            平均学習損失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 順伝播
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 進行バーの更新
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        モデルの検証を実行
        
        Args:
            val_loader: 検証データローダー
            criterion: 損失関数
        
        Returns:
            (検証損失, 指標)のタプル
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 指標計算のために予測とターゲットを保存
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        # 指標の計算
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = self.calculate_metrics(all_targets, all_predictions)
        
        return total_loss / num_batches, metrics
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-5,
            patience: int = 15,
            save_best: bool = True) -> Dict[str, List[float]]:
        """
        モデルの学習を実行
        
        Args:
            train_loader: 学習データローダー
            val_loader: 検証データローダー
            epochs: 学習エポック数
            learning_rate: 学習率
            weight_decay: 正則化のための重み減衰
            patience: 早期停止の忍耐値
            save_best: 最適モデルを保存するかどうか
        
        Returns:
            学習履歴の辞書
        """
        # 最適化器と損失関数の初期化
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=learning_rate, 
                               weight_decay=weight_decay)
        
        # 学習率スケジューラー
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        criterion = nn.MSELoss()
        
        # 早期停止
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
        
        # 学習ループ
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 学習
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # 検証
            val_loss, val_metrics = self.validate(val_loader, criterion)
            
            # 学習率スケジューリング
            scheduler.step(val_loss)
            
            # 履歴の記録
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            
            # Tensorboardログ記録
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # エポック結果の表示
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val R2: {val_metrics['r2']:.4f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 50)
            
            # 最適モデルの保存
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
            
            # 早期停止
            if early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.writer.close()
        
        return {
            'train_loss': self.train_history,
            'val_loss': self.val_history
        }
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        データセットで予測を実行
        
        Args:
            data_loader: 予測用データローダー
        
        Returns:
            (予測, ターゲット)のタプル
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Predicting", leave=False)
            
            for data, target in progress_bar:
                data = data.to(self.device)
                
                output = self.model(data)
                
                predictions.append(output.cpu().numpy())
                targets.append(target.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        return predictions, targets
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False):
        """
        モデルチェックポイントを保存
        
        Args:
            epoch: 現在のエポック
            train_loss: 学習損失
            val_loss: 検証損失
            is_best: これがこれまでの最適モデルかどうか
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config.to_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"Best model saved at epoch {epoch+1}")
    
    def load_checkpoint(self, filepath: str):
        """
        モデルチェックポイントを読み込み
        
        Args:
            filepath: チェックポイントファイルのパス
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Validation Loss: {checkpoint['val_loss']:.6f}")


class PatchTSTInference:
    """
    PatchTSTモデル用推論パイプライン
    """
    
    def __init__(self, 
                 model_path: str,
                 preprocessor: CryptoDataPreprocessor,
                 device: str = 'auto'):
        """
        Args:
            model_path: 学習済みモデルのパス
            preprocessor: データ前処理器
            device: 使用するデバイス
        """
        self.preprocessor = preprocessor
        
        # デバイスの設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # モデルの読み込み
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        学習済みモデルを読み込み
        
        Args:
            model_path: モデルファイルのパス
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 設定からモデルを作成
        config = PatchTSTConfig(**checkpoint['config'])
        self.model = PatchTST(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            n_vars=config.n_vars,
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        
        # 重みの読み込み
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.config = config
        print(f"Model loaded from {model_path}")
    
    def predict_single(self, data: np.ndarray) -> np.ndarray:
        """
        単一系列で予測を実行
        
        Args:
            data: 形状(seq_len, n_vars)の入力系列
        
        Returns:
            形状(pred_len, n_vars)の予測
        """
        # バッチ次元を追加
        data = torch.FloatTensor(data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(data)
        
        return prediction.cpu().numpy().squeeze(0)
    
    def predict_next_prices(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        過去データから次の価格を予測
        
        Args:
            historical_data: 過去のOHLCVデータ
        
        Returns:
            予測とメタデータを含む辞書
        """
        # データの前処理
        processed_data, feature_names = self.preprocessor.preprocess(
            historical_data, fit=False
        )
        
        # 予測のために最後のseq_lenサンプルを取得
        if len(processed_data) < self.config.seq_len:
            raise ValueError(f"Need at least {self.config.seq_len} samples for prediction")
        
        input_sequence = processed_data[-self.config.seq_len:]
        
        # 予測の実行
        prediction = self.predict_single(input_sequence)
        
        # 予測の逆変換
        pred_reshaped = prediction.reshape(-1, len(feature_names))
        pred_original_scale = self.preprocessor.inverse_transform(pred_reshaped)
        
        return {
            'prediction': pred_original_scale,
            'feature_names': feature_names,
            'pred_len': self.config.pred_len,
            'timestamp': historical_data.index[-1] if hasattr(historical_data, 'index') else None
        }