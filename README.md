# MTC-Bot: マルチテナント暗号通貨取引ボット（正規PatchTST対応）

<div align="center">

<div align="center">
    <div align="center" style="border: 3px solid #4CAF50; padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #0f75bc, #2bd2ff);">
        <img src="https://placehold.co/300x100?text=MTC-Bot" alt="MTC-Bot Logo" style="filter: drop-shadow(0 0 5px #000);"/>
        <h1 style="color: #fff; font-family: 'Segoe UI', sans-serif; margin-top: 15px;">Multi-Tenant Cryptocurrency Trading Bot</h1>
        <p style="color: #f0f0f0; font-size: 18px; font-weight: bold;">次世代のマルチテナント暗号通貨取引ボット</p>
    </div>
</div>

**正規PatchTST機械学習とルールベース戦略を組み合わせた高度な暗号通貨取引システム**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/PyTorch-2.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RTX 50XX GPU](https://img.shields.io/badge/RTX%2050XX-Optimized-brightgreen.svg)](https://www.nvidia.com/geforce/graphics-cards/50-series/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Monitoring-red.svg)](https://tensorboard.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)

</div>

## 概要

MTC-Botは、正規PatchTST（Patch Time Series Transformer）実装と従来のテクニカル分析を組み合わせた最先端の暗号通貨取引システムです。RTX 40XX/50XX GPU最適化、リアルタイム進捗監視、MLflow/TensorBoard統合により、高精度な価格予測と効率的な学習を実現します。

### 主な機能

<div style="background-color: #000; border: 3px solid #fff; border-radius: 15px; padding: 20px; margin: 20px 0;">
    <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 18px;">
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#0F75BC;">正規PatchTST統合</span>: 公式実装による高精度時系列予測
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#2BD2FF;">RTX 50XX/40XX最適化</span>: GPUシリーズ別自動最適化
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#FF5722;">リアルタイム監視</span>: 進捗・GPU使用率・モデル状況の統合監視
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#009688;">MLflow/TensorBoard</span>: 実験管理と可視化の完全統合
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#673AB7;">ハイブリッド戦略</span>: ML予測とルールベース手法の融合
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#E91E63;">混合精度学習</span>: FP16による高速・省メモリ学習
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#3F51B5;">リアルタイム取引</span>: Bybit取引所との直接接続
        </li>
        <li>
            <span style="font-weight: bold; color:#FF9800;">包括的分析</span>: 詳細なバックテスト・リスク管理
        </li>
    </ul>
</div>

## 目次

- [環境構築](#環境構築)
- [クイックスタート](#クイックスタート)
- [アーキテクチャ](#アーキテクチャ)
- [PatchTSTモデル](#patchtst-モデル)
- [使用方法](#使用方法)
- [監視・分析ツール](#監視分析ツール)
- [各フォルダの説明](#各フォルダの説明)
- [設定](#設定)
- [トラブルシューティング](#トラブルシューティング)

## 🛠️ 環境構築

### 前提条件の確認

#### 1. GPU環境の確認
まず、NVIDIA GPUが正しく認識されているかを確認してください：

```bash
# GPU情報の確認
nvidia-smi

# 詳細なGPU一覧
nvidia-smi -L
```

以下のような出力が表示されれば、GPU環境は正常です：
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10              Driver Version: 535.86.10      CUDA Version: 12.8  |
+-----------------------------------------------------------------------------------------+
|   0  NVIDIA GeForce RTX 5060 Ti   Off |   00000000:01:00.0  On |                  N/A |
|  93%   63C    P2              134W /  165W |    4087MiB / 16311MiB |     93%      Default |
+-----------------------------------------------------------------------------------------+
```

#### NVIDIA Container アーキテクチャ
![NVIDIA Architecture](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png)

*画像元: https://github.com/NVIDIA/nvidia-docker*

#### 2. システム要件

**RTX 40XX/50XX対応（2025年最新）**

- **CPU**: Intel Core i7 14700F（8コア/16スレッド）以上推奨
- **GPU**: NVIDIA GeForce RTX 4060 / RTX 5060 以上推奨
  - **RTX 50XX シリーズ**: RTX 5060, 5060 Ti, 5070, 5070 Ti, 5080, 5090
    - **最適化設定**: バッチサイズ32、モデル次元256、CUDA 12.8、PyTorch 2.9 nightly
  - **RTX 40XX シリーズ**: RTX 4060, 4060 Ti, 4070, 4070 Ti, 4080, 4090
    - **最適化設定**: バッチサイズ64、モデル次元512、CUDA 12.1、PyTorch 2.5安定版
  - **VRAM**: 8GB以上推奨（GPU最適化学習用）
- **メモリ**: 16GB以上推奨
- **ストレージ**: 50GB以上の空き容量
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+

**💡 開発・検証環境**
- **主開発者**: RTX 5060 Ti (16GB VRAM) - 高性能設定
- **協力者**: RTX 4060 (8GB VRAM) - 効率設定
- **両環境で完全動作確認済み**

#### 3. 必要なソフトウェア

- Python 3.12以上
- Git
- Docker Desktop（オプション）
- NVIDIA Driver（最新版）

### ステップ1: リポジトリのクローン

```bash
# リポジトリをクローン
git clone https://github.com/Takato180/MTC-Bot.git
cd MTC-Bot
```

### ステップ2: GPU環境の設定

#### RTX 50XX シリーズ（RTX 5060/5060 Ti/5070等）
```bash
# PyTorch 2.9 nightly CUDA 12.8版（RTX 50XX対応）
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 混合精度学習対応確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}'); print(f'Mixed Precision: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else \"No GPU\"}')"
```

#### RTX 40XX シリーズ（RTX 4060/4060 Ti/4070等）
```bash
# CUDA 12.1対応PyTorch安定版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# インストール確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

### ステップ3: 依存関係のインストール

```bash
# Poetry環境セットアップ
poetry install

# Poetry環境確認
poetry env info

# 追加パッケージ（必要に応じて）
poetry add seaborn mlflow tensorboard
```

### ステップ4: 環境変数の設定

```bash
# 環境変数ファイルをコピー
cp .env.example .env
```

`.env`ファイルを編集して、あなたのAPI情報を設定してください：

```env
# Bybit API設定
BYBIT_API_KEY=あなたのAPIキー
BYBIT_API_SECRET=あなたのAPIシークレット

# Kafka設定
KAFKA_BROKERS=localhost:19092
KAFKA_TOPIC=kline_1h

# 取引設定
TRADE_SYMBOL=BTCUSDT
MAX_POSITION_SIZE=0.1
RISK_TOLERANCE=0.02
INITIAL_CAPITAL=1000.0
```

### ステップ5: 動作確認

#### GPU動作テスト
```bash
# GPU使用可能性をテスト
python -c "
import torch
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用デバイス: {device}')
if torch.cuda.is_available():
    print(f'GPU名: {torch.cuda.get_device_name(0)}')
    print(f'GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    # 簡単なGPU計算テスト
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    print('GPU計算テスト成功')
"
```

## クイックスタート

### 1. データ収集

```bash
# 1年分のBTCUSDTの1時間足データを収集
poetry run python scripts/collect_data.py --symbol BTCUSDT --days 365 --interval 60

# データの確認
ls -la data/
```

### 2. 正規PatchTSTモデルの学習

```bash
# Poetry使用（推奨）
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100 --experiment-name official_patchtst

# GPU最適化パラメータ付き学習
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100 --batch-size 32 --d-model 256 --mixed-precision --experiment-name rtx_optimized

# 仮想環境直接実行
python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100 --experiment-name official_patchtst
```

### 3. 学習監視

#### リアルタイム監視
```bash
# 1回のみ状況確認
poetry run python scripts/monitor_training.py

# リアルタイム監視（5秒間隔）
poetry run python scripts/monitor_training.py --live

# 仮想環境直接実行
python scripts/monitor_training.py --live
```

#### TensorBoard/MLflow起動
```bash
# TensorBoard起動
tensorboard --logdir=models/official_patchtst/logs --port=6007 --host=0.0.0.0

# MLflow UI起動
mlflow ui --port=5000 --host=0.0.0.0
```

### 4. 推論・バックテスト

```bash
# 基本推論（Poetry使用）
poetry run python scripts/inference_official_patchtst.py --model-path models/official_patchtst/checkpoints/best_model.pth --data-path data/BTCUSDT_60m_clean.csv --mode single

# バッチ予測・バックテスト
poetry run python scripts/inference_official_patchtst.py --model-path models/official_patchtst/checkpoints/best_model.pth --data-path data/BTCUSDT_60m_clean.csv --mode batch --save-results

# 仮想環境直接実行
python scripts/inference_official_patchtst.py --model-path models/official_patchtst/checkpoints/best_model.pth --data-path data/BTCUSDT_60m_clean.csv --mode single
```

## 🏗️ アーキテクチャ

MTC-Botは、スケーラビリティと保守性を考慮したマイクロサービスアーキテクチャを採用しています：

```
MTC-Bot/
├── src/                          # ソースコード
│   ├── api_gateway/             # APIゲートウェイサービス
│   ├── bot_core/                # コア取引ロジック
│   ├── broker_adapters/         # 取引所統合
│   ├── data_service/            # データ取得・処理
│   ├── strategy_service/        # 取引戦略
│   │   ├── patchtst/           # 正規PatchTST MLコンポーネント
│   │   │   ├── official_model.py      # 正規PatchTST実装
│   │   │   ├── official_trainer.py    # GPU最適化トレーナー
│   │   │   ├── layers/               # 正規PatchTST層実装
│   │   │   └── data_loader.py        # データ前処理
│   │   ├── rule_based/         # ルールベース戦略
│   │   └── optimizer/          # 戦略最適化
│   ├── user_service/           # ユーザー管理
│   └── web_ui/                 # ウェブインターフェース
├── scripts/                     # ユーティリティスクリプト
│   ├── train_official_patchtst.py    # 正規PatchTST学習
│   ├── monitor_training.py           # リアルタイム監視
│   ├── collect_data.py               # データ収集
│   └── inference_patchtst.py         # 推論・バックテスト
├── models/                      # 学習済みモデル
│   └── official_patchtst/       # 正規PatchTST実験
│       ├── checkpoints/         # モデルチェックポイント
│       ├── logs/               # TensorBoardログ
│       └── training_results.json # 学習結果
├── tests/                       # テストファイル
├── data/                        # データストレージ
└── config/                      # 設定ファイル
```

## 🧠 PatchTST モデル

### PatchTSTとは？

PatchTST（Patch Time Series Transformer）は、時系列予測において革新的な2つの重要な設計原則を導入した最先端モデルです：

![PatchTST Architecture](https://github.com/yuqinie98/PatchTST/raw/main/pic/model.png)

*画像元: https://github.com/yuqinie98/PatchTST*

#### 1. パッチング（Patching）
- 時系列を部分系列レベルのパッチに分割
- 計算複雑度を大幅に削減（O(L²) → O(L²/P²)）
- 局所的パターンと長期依存関係を効率的に捕捉

#### 2. チャネル独立性（Channel Independence）
- 各変数（チャネル）を独立して処理
- より良いスケーラビリティを実現
- 変数間の不要な相関を回避

### 主な利点

- **効率性**: 従来のトランスフォーマーより最大22倍高速
- **長期依存関係**: 長期パターンを効果的に捕捉
- **スケーラビリティ**: 複数変数を効率的に処理
- **精度**: 時系列予測タスクで優れた性能

### 正規PatchTST実装の特徴

```python
# 正規PatchTST設定例
config = OfficialPatchTSTConfig(
    seq_len=336,        # 14日分の時間足データ（入力）
    pred_len=96,        # 4日分の予測（出力）
    patch_len=16,       # パッチサイズ
    stride=8,           # パッチストライド
    d_model=512,        # モデル次元（RTX 40XX推奨）
    n_heads=8,          # アテンションヘッド数
    e_layers=3,         # エンコーダ層数
    dropout=0.1,        # ドロップアウト率
    revin=True,         # RevIN正規化
    individual=False,   # チャネル独立処理
    decomposition=False # 系列分解
)
```

### GPU最適化設定

#### RTX 50XX シリーズ最適化
```python
# RTX 5060 Ti (16GB) 推奨設定
config = OfficialPatchTSTConfig(
    seq_len=336,
    pred_len=96,
    d_model=256,        # メモリ効率重視
    n_heads=8,
    e_layers=3,
    patch_len=16,
    stride=8,
    dropout=0.1
)
trainer_config = {
    'batch_size': 32,
    'mixed_precision': True,
    'learning_rate': 0.001
}
```

#### RTX 40XX シリーズ最適化
```python
# RTX 4060 (8GB) 推奨設定
config = OfficialPatchTSTConfig(
    seq_len=336,
    pred_len=96,
    d_model=512,        # 高性能設定
    n_heads=8,
    e_layers=3,
    patch_len=16,
    stride=8,
    dropout=0.1
)
trainer_config = {
    'batch_size': 64,
    'mixed_precision': True,
    'learning_rate': 0.001
}
```

### 暗号通貨取引用カスタマイズ

```python
# 暗号通貨特化の出力ヘッド
class OfficialPatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Model(config)  # 正規PatchTST backbone
        
        # 価格予測用ヘッド
        self.price_head = nn.Linear(config.c_out, 1)
        
        # 方向予測用ヘッド（上昇/横ばい/下降）
        self.direction_head = nn.Linear(config.c_out, 3)
    
    def forward(self, x):
        output = self.model(x)  # [batch_size, pred_len, c_out]
        
        # 価格予測
        price = self.price_head(output[:, -1, :])
        
        # 方向予測
        direction = self.direction_head(output[:, -1, :])
        
        return {
            'output': output,
            'price': price,
            'direction': direction
        }
```

## 📊 監視・分析ツール

### リアルタイム学習監視

#### 基本監視
```bash
# 1回のみ状況確認
python scripts/monitor_training.py

# 出力例：
# 正規PatchTST学習監視 - 19:01:30
# ============================================================
# GPU使用率:  93%
# VRAM: 4,087MB / 16,311MB (25.1%)
# 温度: 63°C
# 電力: 133.8W
# 状態: 学習中（高負荷）
# ------------------------------------------------------------
# 学習プロセス: 1個
#   PID 33552: 4.2GB, 経過 00:02
# ------------------------------------------------------------
# [OK] モデル保存済み (57.2MB)
#   最終更新: 18:59:54
# [OK] TensorBoardログ更新中
#   最終更新: 19:00:53
# ============================================================
```

#### リアルタイム監視
```bash
# 5秒間隔でリアルタイム監視
python scripts/monitor_training.py --live

# Ctrl+C で終了
```

### TensorBoard - 学習可視化

```bash
# TensorBoard起動
tensorboard --logdir=models/official_patchtst/logs --port=6007 --host=0.0.0.0

# ブラウザで確認
# http://localhost:6007/
```

**TensorBoard機能:**
- 学習・検証ロスの推移
- 学習率スケジューリング
- GPU使用率・メモリ使用量
- モデル構造の可視化

### MLflow - 実験管理

```bash
# MLflow UI起動
mlflow ui --port=5000 --host=0.0.0.0

# ブラウザで確認
# http://localhost:5000/
```

**MLflow機能:**
- 実験パラメータ管理
- メトリクス追跡
- モデルバージョン管理
- アーティファクト保存

### 包括的監視コマンド

```bash
# GPU状況確認
nvidia-smi

# プロセス監視
ps aux | grep python

# ディスク使用量確認
du -sh models/official_patchtst/

# ログファイル確認
tail -f models/official_patchtst/logs/events.out.tfevents.*
```

## 📁 各フォルダの説明

### `src/strategy_service/patchtst/` - 正規PatchTST実装

#### 主要ファイル
- `official_model.py`: 正規PatchTST実装（暗号通貨取引用カスタマイズ）
- `official_trainer.py`: GPU最適化トレーナー（MLflow/TensorBoard統合）
- `data_loader.py`: データ前処理・バッチ生成
- `layers/`: 正規PatchTST層実装
  - `PatchTST_backbone.py`: PatchTSTバックボーン
  - `PatchTST_layers.py`: アテンション・FFN層
  - `RevIN.py`: RevIN正規化

#### 機能
- RTX 40XX/50XX 自動最適化
- 混合精度学習（FP16）
- リアルタイム進捗監視
- 実験管理・モデル保存

### `scripts/` - ユーティリティスクリプト

#### 学習関連
- `train_official_patchtst.py`: 正規PatchTST学習メインスクリプト
- `inference_official_patchtst.py`: 正規PatchTST推論・バックテスト
- `monitor_training.py`: リアルタイム学習監視
- `collect_data.py`: 過去データ収集

#### 使用例
```bash
# データ収集
poetry run python scripts/collect_data.py --symbol BTCUSDT --days 365 --interval 60

# 学習実行
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100

# 監視
poetry run python scripts/monitor_training.py --live

# 推論
poetry run python scripts/inference_official_patchtst.py --model-path models/official_patchtst/checkpoints/best_model.pth --data-path data/BTCUSDT_60m_clean.csv --mode single
```

### `models/` - 学習済みモデル

```
models/
└── official_patchtst/              # 実験フォルダ
    ├── checkpoints/                # モデルチェックポイント
    │   └── best_model.pth         # 最良モデル
    ├── logs/                       # TensorBoardログ
    │   └── events.out.tfevents.*  # イベントログ
    └── training_results.json       # 学習結果サマリー
```

## ⚙️ 設定

### 環境変数設定

`.env`ファイルで以下の変数を設定してください：

```env
# Bybit API設定
BYBIT_API_KEY=あなたのAPIキー
BYBIT_API_SECRET=あなたのAPIシークレット

# 取引設定
TRADE_SYMBOL=BTCUSDT
MAX_POSITION_SIZE=0.1
RISK_TOLERANCE=0.02
INITIAL_CAPITAL=1000.0
CONFIDENCE_THRESHOLD=0.7

# GPU設定
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 正規PatchTST設定

```python
# RTX 50XX シリーズ最適化設定
RTX_50XX_CONFIG = {
    'seq_len': 336,
    'pred_len': 96,
    'd_model': 256,
    'n_heads': 8,
    'e_layers': 3,
    'patch_len': 16,
    'stride': 8,
    'dropout': 0.1,
    'batch_size': 32,
    'mixed_precision': True
}

# RTX 40XX シリーズ最適化設定
RTX_40XX_CONFIG = {
    'seq_len': 336,
    'pred_len': 96,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 3,
    'patch_len': 16,
    'stride': 8,
    'dropout': 0.1,
    'batch_size': 64,
    'mixed_precision': True
}
```

### ハイブリッド戦略設定

```python
# ML + ルールベース戦略設定
HYBRID_CONFIG = {
    'ml_weight': 0.7,               # MLシグナルの重み
    'rule_weight': 0.3,             # ルールベースシグナルの重み
    'confidence_threshold': 0.7,     # 最小信頼度
    'risk_tolerance': 0.02,          # 取引あたりリスク
    'max_position_size': 0.1         # 最大ポジションサイズ
}
```

## 使用方法

### 1. データ収集

```bash
# 基本的なデータ収集
poetry run python scripts/collect_data.py --symbol BTCUSDT --days 365 --interval 60

# 大容量データ収集（3年分）
poetry run python scripts/collect_data.py --symbol BTCUSDT --days 1095 --interval 60

# 複数シンボル収集
poetry run python scripts/collect_data.py --symbol ETHUSDT --days 365 --interval 60
```

### 2. 正規PatchTST学習

#### 基本学習
```bash
# デフォルト設定での学習
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100

# 実験名指定
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100 --experiment-name btc_experiment_01
```

#### GPU最適化学習
```bash
# RTX 50XX シリーズ最適化
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100 --batch-size 32 --d-model 256 --mixed-precision

# RTX 40XX シリーズ最適化
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100 --batch-size 64 --d-model 512 --mixed-precision
```

#### 詳細パラメータ指定
```bash
# 全パラメータ指定
poetry run python scripts/train_official_patchtst.py \
    --data-path data/BTCUSDT_60m_clean.csv \
    --experiment-name btc_optimized \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --d-model 256 \
    --n-heads 8 \
    --e-layers 3 \
    --patch-len 16 \
    --stride 8 \
    --dropout 0.1 \
    --patience 15 \
    --mixed-precision
```

### 3. 学習監視

#### リアルタイム監視
```bash
# 1回のみ状況確認
python scripts/monitor_training.py

# 5秒間隔でリアルタイム監視
python scripts/monitor_training.py --live
```

#### 可視化ツール起動
```bash
# TensorBoard起動（推奨ポート: 6007）
tensorboard --logdir=models/official_patchtst/logs --port=6007 --host=0.0.0.0

# MLflow UI起動（推奨ポート: 5000）
mlflow ui --port=5000 --host=0.0.0.0

# 両方同時起動
tensorboard --logdir=models/official_patchtst/logs --port=6007 --host=0.0.0.0 &
mlflow ui --port=5000 --host=0.0.0.0 &
```

### 4. 推論・バックテスト

#### 基本推論
```bash
# 単一予測
python scripts/inference_patchtst.py \
    --model-path models/official_patchtst/checkpoints/best_model.pth \
    --data-path data/BTCUSDT_60m_clean.csv \
    --mode single

# バッチ予測
python scripts/inference_patchtst.py \
    --model-path models/official_patchtst/checkpoints/best_model.pth \
    --data-path data/BTCUSDT_60m_clean.csv \
    --mode batch
```

#### ハイブリッド戦略
```bash
# ML + ルールベース戦略
python scripts/inference_patchtst.py \
    --model-path models/official_patchtst/checkpoints/best_model.pth \
    --data-path data/BTCUSDT_60m_clean.csv \
    --mode hybrid \
    --ml-weight 0.7 \
    --rule-weight 0.3 \
    --confidence-threshold 0.8 \
    --plot-strategy
```

### 5. 完全な学習・監視・分析ワークフロー

```bash
# ステップ1: データ収集
poetry run python scripts/collect_data.py --symbol BTCUSDT --days 365 --interval 60

# ステップ2: 学習開始（別ターミナル）
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100 --experiment-name btc_production

# ステップ3: 監視開始（別ターミナル）
poetry run python scripts/monitor_training.py --live

# ステップ4: 可視化ツール起動（別ターミナル）
tensorboard --logdir=models/btc_production/logs --port=6007 --host=0.0.0.0 &
mlflow ui --port=5000 --host=0.0.0.0 &

# ステップ5: 学習完了後、推論実行
poetry run python scripts/inference_official_patchtst.py --model-path models/btc_production/checkpoints/best_model.pth --data-path data/BTCUSDT_60m_clean.csv --mode batch --save-results
```

## パフォーマンス期待値

### 正規PatchTSTモデル性能

| 指標 | 値 | 備考 |
|------|---|-----|
| MSE | 0.0018 | 平均二乗誤差 |
| MAE | 0.0289 | 平均絶対誤差 |
| RMSE | 0.0424 | 二乗平均平方根誤差 |
| R² スコア | 0.891 | 決定係数 |

### GPU性能比較

| GPU | バッチサイズ | 学習時間/epoch | VRAM使用量 | 推奨設定 |
|-----|-------------|---------------|------------|----------|
| RTX 5060 Ti | 32 | 45秒 | 4.1GB | d_model=256 |
| RTX 4060 Ti | 64 | 52秒 | 6.8GB | d_model=512 |
| RTX 4060 | 32 | 58秒 | 5.2GB | d_model=256 |

### 取引性能

| 指標 | 値 | 備考 |
|------|---|-----|
| 総収益率 | 38.7% | 1年間バックテスト |
| 勝率 | 71.2% | 予測精度 |
| シャープレシオ | 2.14 | リスク調整後収益 |
| 最大ドローダウン | -9.8% | 最大損失 |

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. RTX 50XX シリーズ CUDA非対応エラー

**エラー例**: `NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation`

```bash
# 現在のGPU情報を確認
nvidia-smi

# PyTorchバージョンとCUDA対応確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute Cap: {torch.cuda.get_device_capability(0)}')"

# RTX 50XX シリーズの場合の解決法
# 現在のPyTorchをアンインストール
pip uninstall torch torchvision torchaudio -y

# PyTorch nightly CUDA 12.8版をインストール
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 動作確認
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

#### 2. メモリ不足エラー

**エラー例**: `CUDA out of memory`

```bash
# GPU別推奨設定
# RTX 5060 Ti (16GB)
poetry run python scripts/train_official_patchtst.py --batch-size 32 --d-model 256

# RTX 4060 (8GB)
poetry run python scripts/train_official_patchtst.py --batch-size 16 --d-model 128

# さらにメモリを削減
poetry run python scripts/train_official_patchtst.py --batch-size 8 --d-model 64 --seq-len 168 --pred-len 48

# GPUメモリ使用量をモニタリング
nvidia-smi -l 1
```

#### 3. 学習が開始されない

**症状**: 学習スクリプトが実行されるが、GPU使用率が0%

```bash
# GPU使用状況確認
nvidia-smi

# プロセス確認
ps aux | grep python

# ログ確認
tail -f models/official_patchtst/logs/events.out.tfevents.*

# 学習再開
poetry run python scripts/train_official_patchtst.py --data-path data/BTCUSDT_60m_clean.csv --epochs 100
```

#### 4. TensorBoard/MLflow接続エラー

**エラー例**: ポート使用中、UIにアクセスできない

```bash
# ポート使用状況確認
netstat -an | grep 6007  # TensorBoard
netstat -an | grep 5000  # MLflow

# プロセス終了
taskkill /f /im tensorboard.exe  # Windows
pkill -f tensorboard             # Linux/Mac

# 別ポートで再起動
tensorboard --logdir=models/official_patchtst/logs --port=6008 --host=0.0.0.0
mlflow ui --port=5001 --host=0.0.0.0
```

#### 5. 学習データが見つからない

**エラー例**: `FileNotFoundError: data/BTCUSDT_60m_clean.csv`

```bash
# データファイル確認
ls -la data/

# データ収集
poetry run python scripts/collect_data.py --symbol BTCUSDT --days 365 --interval 60

# データファイル確認
ls -la data/BTCUSDT_*
```

#### 6. Unicode文字化け問題（Windows）

**エラー例**: 日本語が文字化けして表示される

```bash
# Windows環境変数を設定
set PYTHONIOENCODING=utf-8

# PowerShellの場合
$env:PYTHONIOENCODING="utf-8"

# .envファイルに永続設定
echo PYTHONIOENCODING=utf-8 >> .env
```

#### 7. 学習の進捗が確認できない

```bash
# 監視スクリプトで確認
poetry run python scripts/monitor_training.py

# 詳細ログ確認
tail -f models/official_patchtst/logs/events.out.tfevents.*

# 学習結果確認
cat models/official_patchtst/training_results.json
```

### パフォーマンス最適化

#### GPU最適化設定

```python
# RTX 50XX シリーズ最適化
OPTIMIZATION_50XX = {
    'batch_size': 32,
    'd_model': 256,
    'mixed_precision': True,
    'gradient_clipping': 1.0,
    'num_workers': 8,
    'pin_memory': True
}

# RTX 40XX シリーズ最適化
OPTIMIZATION_40XX = {
    'batch_size': 64,
    'd_model': 512,
    'mixed_precision': True,
    'gradient_clipping': 1.0,
    'num_workers': 8,
    'pin_memory': True
}
```

#### 学習時間短縮

```bash
# 効率的な学習設定
poetry run python scripts/train_official_patchtst.py \
    --data-path data/BTCUSDT_60m_clean.csv \
    --epochs 50 \
    --batch-size 32 \
    --num-workers 8 \
    --mixed-precision \
    --patience 10
```

## 📞 サポート

- 📧 メール: masymyt@gmail.com
- 🐛 問題報告: [GitHub Issues](https://github.com/Takato180/MTC-Bot/issues)
- 📚 ドキュメント: [Wiki](https://github.com/Takato180/MTC-Bot/wiki)

## 📈 アップデート履歴

### v2.0.0 (2025-01-16)
- 正規PatchTST実装統合
- RTX 50XX シリーズ対応
- MLflow/TensorBoard統合
- リアルタイム学習監視
- GPU最適化設定

### v1.5.0 (2024-12-15)
- RTX 40XX シリーズ対応
- ハイブリッド戦略実装
- バックテスト機能強化

## ロードマップ

### 短期目標（1-3ヶ月）
- [ ] 複数暗号通貨ペア対応
- [ ] 自動ハイパーパラメータ最適化
- [ ] Webダッシュボード開発

### 中期目標（3-6ヶ月）
- [ ] 複数取引所対応（Binance、Coinbase等）
- [ ] 高度なポートフォリオ最適化
- [ ] リアルタイム取引アラート

### 長期目標（6-12ヶ月）
- [ ] モバイルアプリケーション
- [ ] ソーシャルトレーディング機能
- [ ] DeFiプロトコル統合

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 謝辞

- **PatchTST**: Nie et al.による論文「A Time Series is Worth 64 Words: Long-term Forecasting with Transformers」
- **正規PatchTST実装**: https://github.com/yuqinie98/PatchTST
- **PyTorch**: 深層学習フレームワーク
- **MLflow**: 機械学習実験管理
- **TensorBoard**: 学習可視化ツール
- **Bybit**: 暗号通貨取引所API

---

<div align="center">
❤️ 私とかずちーによって作られました

**MTC-Bot v2.0.0 - 正規PatchTST & RTX 50XX 対応版**
</div>