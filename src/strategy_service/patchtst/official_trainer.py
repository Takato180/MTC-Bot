#!/usr/bin/env python3
"""
正規PatchTST専用トレーナー
GPU最適化、進捗監視、MLflow/TensorBoard統合
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import mlflow
import mlflow.pytorch

from .official_model import OfficialPatchTST, OfficialPatchTSTConfig

class OfficialPatchTSTTrainer:
    """正規PatchTST用トレーナー"""
    
    def __init__(self, 
                 model: OfficialPatchTST,
                 config: OfficialPatchTSTConfig,
                 device: str = 'auto',
                 mixed_precision: bool = True,
                 experiment_name: str = 'official_patchtst',
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        
        self.model = model
        self.config = config
        self.mixed_precision = mixed_precision
        self.experiment_name = experiment_name
        
        # デバイス設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # GPU最適化設定
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        # 混合精度設定
        self.scaler = GradScaler() if mixed_precision and self.device.type == 'cuda' else None
        
        # ディレクトリ作成
        self.log_dir = os.path.join('models', experiment_name, log_dir)
        self.checkpoint_dir = os.path.join('models', experiment_name, checkpoint_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard設定
        self.writer = SummaryWriter(self.log_dir)
        
        # MLflow設定
        mlflow.set_experiment(experiment_name)
        
        # 学習履歴
        self.train_history = {'loss': [], 'val_loss': []}
        
        print(f"OfficialPatchTSTTrainer初期化完了")
        print(f"デバイス: {self.device}")
        print(f"混合精度: {mixed_precision}")
        print(f"実験名: {experiment_name}")
        
    def train_epoch(self, train_loader, optimizer, criterion, epoch: int) -> float:
        """1エポック学習"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if self.mixed_precision and self.scaler:
                with autocast():
                    outputs = self.model(data)
                    loss = criterion(outputs['output'], target)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = criterion(outputs['output'], target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            # 進捗表示
            if batch_idx % 10 == 0:
                progress = batch_idx / num_batches * 100
                print(f"Epoch {epoch:3d} [{batch_idx:3d}/{num_batches:3d}] "
                      f"({progress:5.1f}%) Loss: {loss.item():.6f}")
            
            # GPU メモリ効率化
            if self.device.type == 'cuda' and batch_idx % 5 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader, criterion) -> float:
        """1エポック検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                if self.mixed_precision and self.scaler:
                    with autocast():
                        outputs = self.model(data)
                        loss = criterion(outputs['output'], target)
                else:
                    outputs = self.model(data)
                    loss = criterion(outputs['output'], target)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def fit(self, 
            train_loader,
            val_loader,
            epochs: int = 100,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-5,
            patience: int = 15,
            save_best: bool = True) -> Dict[str, Any]:
        """学習実行"""
        
        # MLflow run開始
        with mlflow.start_run():
            # パラメータログ
            mlflow.log_params({
                'model_type': 'OfficialPatchTST',
                'seq_len': self.config.seq_len,
                'pred_len': self.config.pred_len,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'e_layers': self.config.e_layers,
                'patch_len': self.config.patch_len,
                'stride': self.config.stride,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'epochs': epochs,
                'batch_size': train_loader.batch_size,
                'device': str(self.device),
                'mixed_precision': self.mixed_precision
            })
            
            # オプティマイザー設定
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # 学習率スケジューラー
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            
            # 損失関数
            criterion = nn.MSELoss()
            
            # 学習開始時刻
            start_time = time.time()
            
            # Early stopping設定
            best_val_loss = float('inf')
            patience_counter = 0
            
            print(f"\n正規PatchTST学習開始")
            print(f"エポック: {epochs}")
            print(f"学習率: {learning_rate}")
            print(f"Early stopping patience: {patience}")
            print("="*60)
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # 学習
                train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch + 1)
                
                # 検証
                val_loss = self.validate_epoch(val_loader, criterion)
                
                # 学習率更新
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                
                # 時間計算
                epoch_time = time.time() - epoch_start
                elapsed_time = time.time() - start_time
                
                # ログ記録
                self.train_history['loss'].append(train_loss)
                self.train_history['val_loss'].append(val_loss)
                
                # TensorBoard記録
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                # MLflow記録
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }, step=epoch)
                
                # 進捗表示
                print(f"Epoch {epoch+1:3d}/{epochs:3d} | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Total: {elapsed_time/60:.1f}min")
                
                # Early stopping & ベストモデル保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_best:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'config': self.config,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'best_val_loss': best_val_loss
                        }
                        
                        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                        torch.save(checkpoint, checkpoint_path)
                        
                        # MLflowにモデル保存
                        mlflow.pytorch.log_model(self.model, "model")
                        
                        print(f"  → ベストモデル保存 (Val Loss: {best_val_loss:.6f})")
                
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # GPU メモリクリア
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # 学習完了
            total_time = time.time() - start_time
            
            # 最終結果保存
            results = {
                'experiment_name': self.experiment_name,
                'total_epochs': epoch + 1,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'total_time': total_time,
                'config': self.config.__dict__,
                'train_history': self.train_history
            }
            
            results_path = os.path.join(os.path.dirname(self.checkpoint_dir), 'training_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # MLflow最終メトリクス
            mlflow.log_metrics({
                'best_val_loss': best_val_loss,
                'total_time': total_time,
                'total_epochs': epoch + 1
            })
            
            print("="*60)
            print(f"正規PatchTST学習完了")
            print(f"総エポック: {epoch + 1}")
            print(f"最良検証Loss: {best_val_loss:.6f}")
            print(f"総学習時間: {total_time/60:.1f}分")
            print(f"結果保存: {results_path}")
            print(f"チェックポイント: {checkpoint_path}")
            print("="*60)
            
            self.writer.close()
            
            return results
    
    def predict(self, data_loader) -> np.ndarray:
        """予測実行"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device, non_blocking=True)
                
                if self.mixed_precision and self.scaler:
                    with autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                
                predictions.append(outputs['output'].cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def save_model(self, path: str):
        """モデル保存"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history
        }
        torch.save(checkpoint, path)
        print(f"モデル保存: {path}")
    
    def load_model(self, path: str):
        """モデル読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        print(f"モデル読み込み: {path}")