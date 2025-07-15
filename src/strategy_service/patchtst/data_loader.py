"""
PatchTST暗号通貨取引用データ読み込みと処理パイプライン

このモジュールは以下を処理します：
- 様々なソースからの時系列データ読み込み
- データの前処理と正規化
- 暗号通貨取引用の特徴量エンジニアリング
- 学習と推論用のデータセット作成
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """
    PatchTSTと互換性のある時系列データ用Datasetクラス
    """
    
    def __init__(self, 
                 data: np.ndarray,
                 seq_len: int,
                 pred_len: int,
                 features: List[str] = None,
                 target: str = 'close',
                 stride: int = 1):
        """
        Args:
            data: 形状(n_samples, n_features)の時系列データ配列
            seq_len: 入力系列長
            pred_len: 予測長
            features: 特徴量名のリスト
            target: ターゲット変数名
            stride: 系列作成時のストライド
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features or [f'feature_{i}' for i in range(data.shape[1])]
        self.target = target
        self.stride = stride
        
        # 系列の作成
        self.sequences = []
        self.targets = []
        
        for i in range(0, len(data) - seq_len - pred_len + 1, stride):
            # 入力系列
            seq = data[i:i + seq_len]
            # ターゲット系列
            target = data[i + seq_len:i + seq_len + pred_len]
            
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        
        return sequence, target


class CryptoDataPreprocessor:
    """
    暗号通貨時系列データ用データ前処理器
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 handle_missing: str = 'forward_fill',
                 feature_engineering: bool = True):
        """
        Args:
            scaler_type: スケーラーの種類 ('standard', 'minmax', 'robust')
            handle_missing: 欠損値の処理方法 ('forward_fill', 'interpolate', 'drop')
            feature_engineering: 追加特徴量を作成するかどうか
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.feature_engineering = feature_engineering
        
        # スケーラーの初期化
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.fitted = False
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        暗号通貨取引用のテクニカル指標を作成
        
        Args:
            df: OHLCVデータを含むDataFrame
        
        Returns:
            追加のテクニカル特徴量を含むDataFrame
        """
        df = df.copy()
        
        # 価格ベースの特徴量
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # 移動平均
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # ボリンジャーバンド
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_ratio'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI（相対力指数）
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD（移動平均収束拡散法）
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 出来高指標
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ボラティリティ
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # サポートおよびレジスタンスレベル
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データセット内の欠損値を処理
        
        Args:
            df: 潜在的な欠損値を含むDataFrame
        
        Returns:
            欠損値が処理されたDataFrame
        """
        df = df.copy()
        
        if self.handle_missing == 'forward_fill':
            df = df.fillna(method='ffill')
        elif self.handle_missing == 'interpolate':
            df = df.interpolate()
        elif self.handle_missing == 'drop':
            df = df.dropna()
        
        # 残りのNaN値を0で埋める
        df = df.fillna(0)
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        PatchTSTモデル用にデータを前処理
        
        Args:
            df: OHLCVデータを含む生DataFrame
            fit: スケーラーをフィットするかどうか
        
        Returns:
            (処理済みデータ, 特徴量名)のタプル
        """
        # 有効な場合、テクニカル特徴量を作成
        if self.feature_engineering:
            df = self.create_technical_features(df)
        
        # 欠損値の処理
        df = self.handle_missing_values(df)
        
        # モデリング用の特徴量を選択
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol']]
        features_df = df[feature_cols]
        
        # データのスケーリング
        if fit:
            scaled_data = self.scaler.fit_transform(features_df.values)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call preprocess with fit=True first.")
            scaled_data = self.scaler.transform(features_df.values)
        
        return scaled_data, feature_cols
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        スケールされたデータを逆変換
        
        Args:
            data: スケールされたデータ
        
        Returns:
            元のスケールのデータ
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted.")
        
        return self.scaler.inverse_transform(data)


class CryptoDataLoader:
    """
    様々なソースからの暗号通貨データ用データローダー
    """
    
    def __init__(self):
        self.data_cache = {}
    
    def load_from_csv(self, file_path: str, 
                     timestamp_col: str = 'timestamp',
                     symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        CSVファイルからデータを読み込み
        
        Args:
            file_path: CSVファイルのパス
            timestamp_col: タイムスタンプ列の名前
            symbol_col: シンボル列の名前
        
        Returns:
            読み込まれたデータを含むDataFrame
        """
        df = pd.read_csv(file_path)
        
        # タイムスタンプをdatetimeに変換
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)
        
        return df
    
    def load_from_bybit_data(self, data: List[Dict]) -> pd.DataFrame:
        """
        Bybit API形式からデータを読み込み
        
        Args:
            data: OHLCVデータを含む辞書のリスト
        
        Returns:
            処理されたデータを含むDataFrame
        """
        df = pd.DataFrame(data)
        
        # タイムスタンプをミリ秒からdatetimeに変換
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # 数値列の確保
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def create_train_test_split(self, 
                               data: np.ndarray,
                               test_size: float = 0.2,
                               validation_size: float = 0.1,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        時系列データの訓練/検証/テスト分割を作成
        
        Args:
            data: 時系列データ
            test_size: テストデータの割合
            validation_size: 検証データの割合
            random_state: ランダムシード
        
        Returns:
            (訓練データ, 検証データ, テストデータ)のタプル
        """
        # 時系列では時間的分割を使用
        n_samples = len(data)
        
        # 分割インデックスの計算
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - validation_size))
        
        # データの分割
        train_data = data[:val_start]
        val_data = data[val_start:test_start]
        test_data = data[test_start:]
        
        return train_data, val_data, test_data


def create_data_loaders(data: np.ndarray,
                       seq_len: int,
                       pred_len: int,
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       validation_size: float = 0.1,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    訓練、検証、テスト用のデータローダーを作成
    
    Args:
        data: 処理済み時系列データ
        seq_len: 入力系列長
        pred_len: 予測長
        batch_size: バッチサイズ
        test_size: テストセットサイズ
        validation_size: 検証セットサイズ
        num_workers: データ読み込み用のワーカー数
    
    Returns:
        (訓練ローダー, 検証ローダー, テストローダー)のタプル
    """
    # データローダーインスタンスの作成
    data_loader = CryptoDataLoader()
    
    # データの分割
    train_data, val_data, test_data = data_loader.create_train_test_split(
        data, test_size, validation_size
    )
    
    # データセットの作成
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


# 使用例と設定
CRYPTO_DATA_CONFIG = {
    'seq_len': 336,  # 14日 × 24時間（時間足データ）
    'pred_len': 96,   # 4日間の予測
    'batch_size': 32,
    'test_size': 0.2,
    'validation_size': 0.1,
    'scaler_type': 'standard',
    'handle_missing': 'forward_fill',
    'feature_engineering': True
}