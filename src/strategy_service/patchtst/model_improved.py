"""
改善版PatchTST: Channel IndependenceとInstance Normalizationを追加
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
import numpy as np
from einops import rearrange, repeat


class ImprovedPatchEmbedding(nn.Module):
    """
    改善版パッチ埋め込み層 - Instance Normalization付き
    """
    
    def __init__(self, 
                 seq_len: int, 
                 patch_len: int, 
                 stride: int, 
                 d_model: int,
                 dropout: float = 0.1,
                 instance_norm: bool = True):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.instance_norm = instance_norm
        
        # パッチ数の計算
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # Instance Normalization (各サンプル・チャネル独立で正規化)
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(1, affine=True)
        
        # パッチの線形投影
        self.projection = nn.Linear(patch_len, d_model)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 位置エンコーディング（学習可能）
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.1
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_vars)
        Returns:
            (batch_size, n_vars, n_patches, d_model)
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Instance Normalization適用（チャネル独立）
        if self.instance_norm:
            # (batch_size * n_vars, 1, seq_len)に変形してInstance Norm適用
            x_norm = x.transpose(1, 2).contiguous()  # (batch_size, n_vars, seq_len)
            x_norm = x_norm.view(-1, 1, seq_len)     # (batch_size * n_vars, 1, seq_len)
            x_norm = self.norm(x_norm)
            x = x_norm.view(batch_size, n_vars, seq_len).transpose(1, 2)  # 元の形に戻す
        
        # パッチ作成 - より効率的な実装
        # Unfoldを使用してパッチを一度に作成
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # (batch_size, n_patches, n_vars, patch_len)
        
        # Channel Independence: 各変数を独立して処理
        # (batch_size, n_vars, n_patches, patch_len)に変形
        patches = patches.transpose(1, 2)
        
        # 線形投影適用
        embeddings = self.projection(patches)  # (batch_size, n_vars, n_patches, d_model)
        
        # 位置エンコーディング追加
        embeddings = embeddings + self.positional_encoding.unsqueeze(0)
        
        # ドロップアウト適用
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ImprovedMultiHeadAttention(nn.Module):
    """
    改善版マルチヘッドアテンション - スケーリングと安定性を向上
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        # Query, Key, Value投影
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # 出力投影
        self.w_o = nn.Linear(d_model, d_model)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # パラメータ初期化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初期化でパラメータを初期化"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: アテンションマスク
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Query, Key, Value計算
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # マスク適用
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax + Dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Value適用
        attention_output = torch.matmul(attention_weights, v)
        
        # ヘッドを結合
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最終線形変換
        output = self.w_o(attention_output)
        
        return output


class ImprovedTransformerEncoder(nn.Module):
    """
    改善版Transformerエンコーダー
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.attention = ImprovedMultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # ReLUよりも滑らかな活性化関数
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Multi-Head Attention + Residual Connection
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed Forward + Residual Connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class ImprovedPatchTST(nn.Module):
    """
    改善版PatchTST - Channel Independence & Instance Normalization付き
    """
    
    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 n_vars: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 channel_independence: bool = True,
                 instance_norm: bool = True):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.channel_independence = channel_independence
        self.instance_norm = instance_norm
        
        # パッチ埋め込み
        self.patch_embedding = ImprovedPatchEmbedding(
            seq_len, patch_len, stride, d_model, dropout, instance_norm
        )
        
        # Transformerエンコーダー層
        self.encoder_layers = nn.ModuleList([
            ImprovedTransformerEncoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 出力投影層（Channel Independent）
        if channel_independence:
            # 各変数に対して独立した投影層
            self.projection_heads = nn.ModuleList([
                nn.Linear(d_model, pred_len) for _ in range(n_vars)
            ])
        else:
            # 共有投影層
            self.projection_head = nn.Linear(d_model * n_vars, pred_len * n_vars)
        
        # パラメータ初期化
        self._init_weights()
    
    def _init_weights(self):
        """パラメータ初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_vars)
        Returns:
            (batch_size, pred_len, n_vars) if channel_independence
            (batch_size, pred_len * n_vars) otherwise
        """
        batch_size = x.shape[0]
        
        # パッチ埋め込み
        embeddings = self.patch_embedding(x)  # (batch_size, n_vars, n_patches, d_model)
        
        if self.channel_independence:
            # Channel Independent処理
            outputs = []
            
            for i in range(self.n_vars):
                # 各変数を独立して処理
                var_embeddings = embeddings[:, i, :, :]  # (batch_size, n_patches, d_model)
                
                # Transformerエンコーダー通過
                for layer in self.encoder_layers:
                    var_embeddings = layer(var_embeddings)
                
                # Global Average Pooling + 投影
                pooled = var_embeddings.mean(dim=1)  # (batch_size, d_model)
                var_output = self.projection_heads[i](pooled)  # (batch_size, pred_len)
                outputs.append(var_output)
            
            # 結果を結合: (batch_size, pred_len, n_vars)
            output = torch.stack(outputs, dim=-1)
            
        else:
            # 従来の処理方法
            # 全変数をまとめて処理
            embeddings = embeddings.view(batch_size, -1, self.patch_embedding.d_model)
            
            for layer in self.encoder_layers:
                embeddings = layer(embeddings)
            
            # Global Average Pooling + 投影
            pooled = embeddings.mean(dim=1)
            output = self.projection_head(pooled)
            output = output.view(batch_size, self.pred_len, self.n_vars)
        
        return output


class ImprovedPatchTSTConfig:
    """改善版PatchTSTモデル用設定クラス"""
    
    def __init__(self,
                 seq_len: int = 336,
                 pred_len: int = 96,
                 n_vars: int = 1,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 channel_independence: bool = True,
                 instance_norm: bool = True):
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.channel_independence = channel_independence
        self.instance_norm = instance_norm
    
    def to_dict(self):
        """設定を辞書形式で返す"""
        return {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_vars': self.n_vars,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'channel_independence': self.channel_independence,
            'instance_norm': self.instance_norm
        }