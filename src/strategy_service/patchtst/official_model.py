#!/usr/bin/env python3
"""
正規PatchTST実装（オリジナルコード使用）
暗号通貨取引用に最適化
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional

# オリジナルPatchTSTのlayersをインポート
from .layers.PatchTST_backbone import PatchTST_backbone
from .layers.PatchTST_layers import series_decomp

@dataclass
class OfficialPatchTSTConfig:
    """正規PatchTST設定"""
    seq_len: int = 336          # 入力系列長（14日）
    pred_len: int = 96          # 予測長（4日）
    enc_in: int = 7             # エンコーダ入力次元
    dec_in: int = 7             # デコーダ入力次元
    c_out: int = 7              # 出力次元
    d_model: int = 512          # モデル次元
    n_heads: int = 8            # アテンションヘッド数
    e_layers: int = 3           # エンコーダ層数
    d_layers: int = 1           # デコーダ層数
    d_ff: int = 2048           # FFN次元
    moving_avg: int = 25        # 移動平均ウィンドウ
    factor: int = 1             # アテンションファクター
    distil: bool = True         # 蒸留使用
    dropout: float = 0.1        # ドロップアウト
    embed: str = 'timeF'        # 埋め込み方式
    activation: str = 'gelu'    # 活性化関数
    output_attention: bool = False  # アテンション出力
    do_predict: bool = False    # 予測モード
    
    # PatchTST特有パラメータ
    patch_len: int = 16         # パッチ長
    stride: int = 8             # ストライド
    padding_patch: str = 'end'  # パッチング方式
    revin: bool = True          # RevIN使用
    affine: bool = True         # アフィン変換
    subtract_last: bool = False # 最終値減算
    decomposition: bool = False # 分解使用
    kernel_size: int = 25       # カーネルサイズ
    individual: bool = False    # 個別処理
    
    # 追加パラメータ
    fc_dropout: float = 0.1     # FC層ドロップアウト
    head_dropout: float = 0.0   # ヘッドドロップアウト

class OfficialPatchTST(nn.Module):
    """
    正規PatchTST実装
    オリジナルコードベース
    """
    
    def __init__(self, config: OfficialPatchTSTConfig):
        super().__init__()
        self.config = config
        
        # オリジナルPatchTSTモデル作成
        self.model = Model(config)
        
        # 暗号通貨取引用の追加レイヤー
        self.price_head = nn.Linear(config.c_out, 1)  # 価格予測用
        self.direction_head = nn.Linear(config.c_out, 3)  # 方向予測用（上昇/横ばい/下降）
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch_size, seq_len, n_vars]
            
        Returns:
            dict: 'output', 'price', 'direction'
        """
        # メイン予測
        output = self.model(x)  # [batch_size, pred_len, c_out]
        
        # 価格予測（最後の特徴量が価格と仮定）
        price = self.price_head(output[:, -1, :])  # [batch_size, pred_len]
        
        # 方向予測（上昇/横ばい/下降）
        direction = self.direction_head(output[:, -1, :])  # [batch_size, 3]
        
        return {
            'output': output,
            'price': price,
            'direction': direction
        }

class Model(nn.Module):
    """
    オリジナルPatchTSTモデル
    """
    
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, 
                 d_v: Optional[int] = None, norm: str = 'BatchNorm', attn_dropout: float = 0., 
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None, 
                 attn_mask: Optional[torch.Tensor] = None, res_attention: bool = True, 
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', 
                 learn_pe: bool = True, pretrain_head: bool = False, head_type: str = 'flatten', 
                 verbose: bool = False, **kwargs):
        
        super().__init__()
        
        # パラメータ設定
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
        
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # モデル構築
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window, 
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, 
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, 
                d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, 
                pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe, 
                fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window, 
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, 
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, 
                d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, 
                pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe, 
                fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window, 
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, 
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, 
                d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, 
                pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe, 
                fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [Batch, Input length, Channel]
            
        Returns:
            x: [Batch, Output length, Channel]
        """
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)
        return x

def create_official_patchtst(n_vars: int, seq_len: int = 336, pred_len: int = 96, 
                           d_model: int = 512, n_heads: int = 8, e_layers: int = 3,
                           patch_len: int = 16, stride: int = 8) -> OfficialPatchTST:
    """
    正規PatchTSTモデルを作成
    
    Args:
        n_vars: 変数数
        seq_len: 入力系列長
        pred_len: 予測長
        d_model: モデル次元
        n_heads: アテンションヘッド数
        e_layers: エンコーダ層数
        patch_len: パッチ長
        stride: ストライド
        
    Returns:
        OfficialPatchTST: 正規PatchTSTモデル
    """
    config = OfficialPatchTSTConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_vars,
        dec_in=n_vars,
        c_out=n_vars,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        patch_len=patch_len,
        stride=stride
    )
    
    return OfficialPatchTST(config)