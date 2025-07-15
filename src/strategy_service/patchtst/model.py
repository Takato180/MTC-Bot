"""
PatchTST: 暗号通貨価格予測のための時系列Transformerモデル

この実装は以下の論文に基づいています：
"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
by Nie et al. (2023)

主な特徴:
- パッチング: 時系列を部分系列レベルのパッチに分割
- チャネル独立性: 各変数が単変量時系列を含む
- 効率的な計算: メモリ使用量と計算時間を大幅に削減
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
import numpy as np
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """
    時系列データ用パッチ埋め込み層
    
    時系列データをパッチに変換し、埋め込み空間に投影します。
    """
    
    def __init__(self, 
                 seq_len: int, 
                 patch_len: int, 
                 stride: int, 
                 d_model: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # パッチ数の計算
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # パッチの線形投影
        self.projection = nn.Linear(patch_len, d_model)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 位置エンコーディング
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_vars)
        
        Returns:
            Tensor of shape (batch_size, n_vars, n_patches, d_model)
        """
        batch_size, seq_len, n_vars = x.shape
        
        # 各変数のパッチを作成
        patches = []
        for i in range(self.n_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, start_idx:end_idx, :]  # (batch_size, patch_len, n_vars)
            patches.append(patch)
        
        # パッチを積み重ね: (batch_size, n_patches, patch_len, n_vars)
        patches = torch.stack(patches, dim=1)
        
        # (batch_size, n_vars, n_patches, patch_len)に再配置
        patches = rearrange(patches, 'b n p v -> b v n p')
        
        # 線形投影を適用: (batch_size, n_vars, n_patches, d_model)
        embeddings = self.projection(patches)
        
        # 位置エンコーディングを追加
        embeddings = embeddings + self.positional_encoding.unsqueeze(0)
        
        # ドロップアウトを適用
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    """
    PatchTST用マルチヘッドアテンション機構
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = query.shape
        
        # 線形変換
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # アテンション計算のための転置
        Q = Q.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # スケールドドット積アテンション
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # バリューにアテンションを適用
        output = torch.matmul(attention, V)
        
        # ヘッドを連結
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 最終線形層
        output = self.w_o(output)
        
        return output


class FeedForward(nn.Module):
    """
    Transformer用フィードフォワードネットワーク
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    単一Transformerエンコーダ層
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 残差接続付きセルフアテンション
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 残差接続付きフィードフォワード
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class PatchTST(nn.Module):
    """
    時系列予測用PatchTSTモデル
    
    このモデルは暗号通貨価格予測のためのPatchTSTアーキテクチャを実装します。
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
                 dropout: float = 0.1):
        """
        Args:
            seq_len: 入力系列長
            pred_len: 予測長
            n_vars: 変数数（チャネル数）
            patch_len: 各パッチの長さ
            stride: パッチ抽出のストライド
            d_model: モデル次元
            n_heads: アテンションヘッド数
            n_layers: Transformer層数
            d_ff: フィードフォワード次元
            dropout: ドロップアウト率
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # パッチ埋め込み
        self.patch_embedding = PatchEmbedding(
            seq_len, patch_len, stride, d_model, dropout
        )
        
        # Transformerエンコーダ層
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 予測ヘッド
        self.prediction_head = nn.Linear(
            self.patch_embedding.n_patches * d_model, pred_len
        )
        
        # チャネル独立層正規化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # パラメータの初期化
        self._init_weights()
        
    def _init_weights(self):
        """モデル重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of PatchTST.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_vars)
        
        Returns:
            Tensor of shape (batch_size, pred_len, n_vars)
        """
        batch_size, seq_len, n_vars = x.shape
        
        # パッチ埋め込み: (batch_size, n_vars, n_patches, d_model)
        patch_embeddings = self.patch_embedding(x)
        
        # 各変数を独立して処理
        outputs = []
        for i in range(n_vars):
            # 変数iの埋め込みを取得: (batch_size, n_patches, d_model)
            var_embeddings = patch_embeddings[:, i, :, :]
            
            # Transformer層を適用
            for layer in self.transformer_layers:
                var_embeddings = layer(var_embeddings)
            
            # 層正規化を適用
            var_embeddings = self.layer_norm(var_embeddings)
            
            # 平坦化して予測: (batch_size, n_patches * d_model)
            var_embeddings = var_embeddings.view(batch_size, -1)
            
            # 予測: (batch_size, pred_len)
            prediction = self.prediction_head(var_embeddings)
            
            outputs.append(prediction)
        
        # 予測を積み重ね: (batch_size, pred_len, n_vars)
        outputs = torch.stack(outputs, dim=-1)
        
        return outputs


class PatchTSTConfig:
    """PatchTSTモデル用設定クラス"""
    
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
                 dropout: float = 0.1):
        
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
    
    def to_dict(self):
        """設定を辞書に変換"""
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
            'dropout': self.dropout
        }


def create_patchtst_model(config: PatchTSTConfig) -> PatchTST:
    """
    設定からPatchTSTモデルを作成
    
    Args:
        config: PatchTSTConfigインスタンス
    
    Returns:
        PatchTSTモデル
    """
    return PatchTST(
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