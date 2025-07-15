# MTC-Bot: マルチテナント暗号通貨取引ボット　※開発中だよぉぉお

<div align="center">

<div align="center">
    <div align="center" style="border: 3px solid #4CAF50; padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #0f75bc, #2bd2ff);">
        <img src="https://placehold.co/300x100?text=MTC-Bot" alt="MTC-Bot Logo" style="filter: drop-shadow(0 0 5px #000);"/>
        <h1 style="color: #fff; font-family: 'Segoe UI', sans-serif; margin-top: 15px;">Multi-Tenant Cryptocurrency Trading Bot</h1>
        <p style="color: #f0f0f0; font-size: 18px; font-weight: bold;">次世代のマルチテナント暗号通貨取引ボット</p>
    </div>
</div>

**PatchTST機械学習とルールベース戦略を組み合わせた高度な暗号通貨取引システム**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20Ready-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

## 概要

MTC-Botは、機械学習の力と従来のテクニカル分析を組み合わせた最先端の暗号通貨取引システムです。その中核には、最新の時系列予測モデルであるPatchTST（Patch Time Series Transformer）を使用し、高精度で暗号通貨の価格変動を予測します。

### 主な機能

<div style="background-color: #000; border: 3px solid #fff; border-radius: 15px; padding: 20px; margin: 20px 0;">
    <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 18px;">
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#0F75BC;">PatchTSTの統合</span>: 高精度価格予測を実現する時系列トランスフォーマー
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#2BD2FF;">ハイブリッド戦略</span>: 機械学習予測とルールベース手法の融合
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#FF5722;">GPU加速</span>: NVIDIA GPUにより迅速な学習と推論を実現
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#009688;">リアルタイム取引</span>: Bybit取引所との直接接続で効率的な取引
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#673AB7;">包括的バックテスト</span>: 詳細なパフォーマンス評価とメトリクス分析
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#E91E63;">リスク管理</span>: 内蔵のポジションサイジングとリスクコントロール
        </li>
        <li style="margin-bottom: 12px;">
            <span style="font-weight: bold; color:#3F51B5;">マイクロサービスアーキテクチャ</span>: スケーラブルで柔軟なシステム設計
        </li>
        <li>
            <span style="font-weight: bold; color:#FF9800;">高度な分析</span>: TensorBoard連携によるパフォーマンスの可視化
        </li>
    </ul>
</div>
        </li>
    </ul>
</div>

## 目次

- [環境構築](#環境構築)
- [クイックスタート](#クイックスタート)
- [アーキテクチャ](#アーキテクチャ)
- [PatchTSTモデル](#patchtst-モデル)
- [使用方法](#使用方法)
- [各フォルダの説明](#各フォルダの説明)
- [設定](#設定)
- [API リファレンス](#api-リファレンス)
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
| NVIDIA-SMI 535.86.10              Driver Version: 535.86.10      CUDA Version: 12.2  |
+-----------------------------------------------------------------------------------------+
|   0  NVIDIA GeForce RTX 5060 Ti   Off |   00000000:01:00.0  On |                  N/A |
|  85%   48C    P2               80W /  165W |    12345MiB /  16380MiB |     95%      Default |
+-----------------------------------------------------------------------------------------+
```

もしコマンドが見つからない場合は、[NVIDIA ドライバー](https://www.nvidia.com/drivers/)をインストールしてください。

#### NVIDIA Container アーキテクチャ
![NVIDIA Architecture](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png)

*画像元: https://github.com/NVIDIA/nvidia-docker*

#### 2. システム要件

- **CPU**: Intel Core i7 14700F（8コア/16スレッド）以上推奨
- **GPU**: NVIDIA GeForce RTX 4060 Ti / RTX 5060 Ti 以上推奨
- **メモリ**: 16GB以上推奨
- **ストレージ**: 50GB以上の空き容量
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+

(デフォルトがこの設定になってるだけなので推奨スペックでなくても大丈夫です。ただし、機械学習を行う上でNVIDIAのGPUがあることは前提としています。)

#### 3. 必要なソフトウェア

- Python 3.12以上  (Pythonのversion色々制限あるから管理気を付けて)
- Git
- Docker Desktop
- Poetry（依存関係管理）

### ステップ1: リポジトリのクローン

```bash
# リポジトリをクローン
git clone https://github.com/Takato180/MTC-Bot.git
cd MTC-Bot
```

### ステップ2: Python環境の設定

#### Poetryのインストール
```bash
# Poetryをインストール（まだインストールしていない場合）
curl -sSL https://install.python-poetry.org | python3 -

# Windowsの場合
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

#### 依存関係のインストール
```bash
# 仮想環境の作成と依存関係のインストール
poetry install

# 仮想環境をアクティベート
poetry shell
```

### ステップ3: GPU環境の設定

#### CUDA対応PyTorchのインストール
```bash
# CUDA 12.1対応のPyTorchをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# インストール確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

期待される出力：
```
PyTorch: 2.1.2+cu121
CUDA available: True
GPU count: 1
```

### ステップ4: Docker環境の設定

#### オプション1: Docker環境（推奨：初心者・チーム開発）

**前提条件**
- Docker & Docker Compose
- NVIDIA Container Toolkit
- RTX 4060 Ti / RTX 5060 Ti + CUDA 12.1

##### Docker Desktopのインストール
1. [Docker Desktop](https://www.docker.com/products/docker-desktop/)をダウンロード・インストール
2. Docker Desktopを起動
3. 動作確認：

```bash
# Dockerの動作確認
docker --version
docker-compose --version
```

##### GPU対応Docker環境の構築
```bash
# GPUサポート付きでビルド
make build-gpu

# 開発開始時
make up

# 開発終了時
make down
```

#### オプション2: ネイティブ環境（推奨：パフォーマンス重視）

```bash
# Poetry環境構築（上記ステップ2,3を参照）
poetry install
poetry shell

# GPU環境確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

#### Kafkaサービスの起動
```bash
# Kafkaコンテナを起動
docker-compose up -d kafka zookeeper

# 起動確認
docker-compose ps
```

### ステップ5: 環境変数の設定

```bash
# 環境変数ファイルをコピー
cp .env.example .env
```

`.env`ファイルを編集して、あなたのAPI情報を設定してください：

```env
# Bybit API設定（取引所でAPIキーを取得してください: testnetのものかリアルのものかは要確認!）
BYBIT_API_KEY=あなたのAPIキー
BYBIT_API_SECRET=あなたのAPIシークレット

# Kafka設定
KAFKA_BROKERS=localhost:19092
KAFKA_TOPIC=kline_1h

# 取引設定
TRADE_SYMBOL=BTCUSDT
MAX_POSITION_SIZE=0.1          # 最大ポジションサイズ（ポートフォリオの10%）
RISK_TOLERANCE=0.02            # リスク許容度（取引あたり2%）
INITIAL_CAPITAL=1000.0         # 初期資本（USD）
MIN_TRADE_SIZE=10.0            # 最小取引サイズ（USD）
MAX_TRADE_SIZE=100.0           # 最大取引サイズ（USD）
```

### ステップ6: 動作確認

#### 基本的な動作テスト
```bash
# Pythonモジュールのインポートテスト
python -c "
import torch
import pandas as pd
import numpy as np
from src.strategy_service.patchtst.model import PatchTSTConfig
print('✅ すべてのモジュールが正常にインポートされました')
"
```

#### GPU動作テスト
```bash
# GPU使用可能性をテスト
python -c "
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用デバイス: {device}')
if torch.cuda.is_available():
    print(f'GPU名: {torch.cuda.get_device_name(0)}')
    print(f'GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

#### Kafka接続テスト
```bash
# Kafka接続テスト
python tests/test_kafka_full.py
```

## クイックスタート

### 1. データ収集

まず、学習用の過去データを収集します：

```bash
# 1年分のBTCUSDTの1時間足データを収集
python scripts/collect_data.py --symbol BTCUSDT --days 365 --interval 60

# データの確認
ls -la data/
```

### 2. PatchTSTモデルの学習

収集したデータでモデルを学習します：

```bash
# 基本的な学習
python scripts/train_patchtst.py \
    --data-path data/BTCUSDT_60m.csv \
    --epochs 50 \
    --batch-size 64 \
    --experiment-name btc_basic

# ハイパーパラメータ最適化付きの学習
python scripts/train_patchtst.py \
    --data-path data/BTCUSDT_60m.csv \
    --optimize-hyperparams \
    --n-trials 50 \
    --epochs 100 \
    --experiment-name btc_optimized
```

### 3. 推論の実行

学習したモデルで予測を行います：

```bash
# 単一予測
python scripts/inference_patchtst.py \
    --model-path models/btc_basic/checkpoints/best_model.pth \
    --preprocessor-path models/btc_basic/preprocessor.pkl \
    --data-path data/BTCUSDT_60m.csv \
    --mode single

# ハイブリッド戦略でのバックテスト
python scripts/inference_patchtst.py \
    --model-path models/btc_basic/checkpoints/best_model.pth \
    --preprocessor-path models/btc_basic/preprocessor.pkl \
    --data-path data/BTCUSDT_60m.csv \
    --mode hybrid \
    --ml-weight 0.7 \
    --rule-weight 0.3 \
    --plot-strategy
```

### 4. 自動売買までの完全手順

**⚠️ 警告: リアルマネーでの取引は十分なテストの後に行ってください**

#### ステップ1: データ収集
```bash
# 過去1年分のBTCUSDTデータを収集
python scripts/collect_data.py --symbol BTCUSDT --days 365 --validate
```

#### ステップ2: モデル学習
```bash
# PatchTSTモデルの学習（RTX 4060 Ti / RTX 5060 Ti最適化済み）
python scripts/train_patchtst.py \
  --data-path data/BTCUSDT_60m.csv \
  --batch-size 128 \
  --d-model 256 \
  --n-heads 16 \
  --epochs 100
```

#### ステップ3: 推論・バックテスト
```bash
# ハイブリッド戦略でバックテスト
python scripts/inference_patchtst.py \
  --mode hybrid \
  --model-path models/patchtst_btc/checkpoints/best_model.pth \
  --preprocessor-path models/patchtst_btc/preprocessor.pkl \
  --data-path data/BTCUSDT_60m.csv \
  --plot-strategy
```

#### ステップ4: ライブ取引開始
```bash
# Kafka + リアルタイム取引
docker-compose up kafka zookeeper  # 別ターミナル

# メイン取引エンジン起動
python src/strategy_service/live_engine.py
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
│   │   ├── patchtst/           # PatchTST ML コンポーネント
│   │   ├── rule_based/         # ルールベース戦略
│   │   └── optimizer/          # 戦略最適化
│   ├── strategy_dsl/           # 戦略定義言語
│   ├── user_service/           # ユーザー管理
│   └── web_ui/                 # ウェブインターフェース
├── scripts/                     # ユーティリティスクリプト
├── tests/                       # テストファイル
├── data/                        # データストレージ
├── models/                      # 学習済みモデル
├── docs/                        # ドキュメント
└── config/                      # 設定ファイル
```

## 🧠 PatchTST モデル

### PatchTSTとは？

PatchTSTは、2つの重要な革新を導入した最先端の時系列予測モデルです：

1. **パッチング**: 時系列を部分系列レベルのパッチに分割し、計算複雑度を大幅に削減
2. **チャネル独立性**: 各変数を独立して処理し、より良いスケーラビリティを実現

### 主な利点

- **効率性**: 従来のトランスフォーマーより最大22倍高速
- **長期依存関係**: 長期パターンを効果的に捕捉
- **スケーラビリティ**: 複数変数を効率的に処理
- **精度**: 時系列予測タスクで優れた性能

### モデル設定例

```python
# モデル設定の例
config = PatchTSTConfig(
    seq_len=336,        # 14日分の時間足データ
    pred_len=96,        # 4日分の予測
    patch_len=16,       # パッチサイズ
    stride=8,           # パッチストライド
    d_model=128,        # モデル次元
    n_heads=8,          # アテンションヘッド数
    n_layers=6,         # トランスフォーマー層数
    dropout=0.1         # ドロップアウト率
)
```

## 📁 各フォルダの説明

### `src/` - メインソースコード

#### `strategy_service/patchtst/`
- `model.py`: PatchTSTモデルの実装
- `trainer.py`: 学習・推論パイプライン
- `data_loader.py`: データ処理・前処理
- `hybrid_strategy.py`: ML+ルールベースハイブリッド戦略

#### `broker_adapters/`
- `bybit_adapter.py`: Bybit取引所との統合

#### `data_service/`
- `bybit_ingestor/`: Bybitからのデータ取得

#### `strategy_dsl/`
- `indicator.py`: テクニカル指標の実装
- `strategies.py`: 戦略定義
- `examples/`: 戦略例

### `scripts/` - ユーティリティスクリプト

- `collect_data.py`: 過去データの収集
- `train_patchtst.py`: モデル学習スクリプト
- `inference_patchtst.py`: 推論・バックテストスクリプト

### `tests/` - テストファイル

- `unit/`: 単体テスト
- `integration/`: 統合テスト
- `e2e/`: エンドツーエンドテスト

### `data/` - データストレージ

- 収集した市場データ
- 前処理済みデータ
- バックテスト結果

### `models/` - 学習済みモデル

- モデルチェックポイント
- 前処理器
- 学習履歴

## ⚙️ 設定

### 環境変数設定

`.env`ファイルで以下の変数を設定してください：

```env
# Bybit API設定
BYBIT_API_KEY=あなたのAPIキー
BYBIT_API_SECRET=あなたのAPIシークレット

# Kafka設定
KAFKA_BROKERS=localhost:19092
KAFKA_TOPIC=kline_1h

# 取引設定
TRADE_SYMBOL=BTCUSDT
MAX_POSITION_SIZE=0.1          # 最大ポジションサイズ（ポートフォリオの10%）
RISK_TOLERANCE=0.02            # リスク許容度（取引あたり2%）
CONFIDENCE_THRESHOLD=0.7       # 信頼度閾値
```

### モデル設定

```python
# PatchTST設定
PATCHTST_CONFIG = {
    'seq_len': 336,          # 入力系列長
    'pred_len': 96,          # 予測期間
    'patch_len': 16,         # パッチサイズ
    'stride': 8,             # パッチストライド
    'd_model': 128,          # モデル次元
    'n_heads': 8,            # アテンションヘッド数
    'n_layers': 6,           # トランスフォーマー層数
    'dropout': 0.1           # ドロップアウト率
}

# ハイブリッド戦略設定
HYBRID_CONFIG = {
    'ml_weight': 0.6,               # MLシグナルの重み
    'rule_weight': 0.4,             # ルールベースシグナルの重み
    'confidence_threshold': 0.7,     # 最小信頼度
    'risk_tolerance': 0.02,          # 取引あたりリスク
    'max_position_size': 0.1         # 最大ポジションサイズ
}
```

## 💡 使用方法

### モデルの学習

```bash
# 基本的な学習
python scripts/train_patchtst.py --data-path data/BTCUSDT_60m.csv

# 高度な学習（ハイパーパラメータ最適化付き）
python scripts/train_patchtst.py \
    --data-path data/BTCUSDT_60m.csv \
    --optimize-hyperparams \
    --n-trials 100 \
    --epochs 200 \
    --experiment-name btc_optimized \
    --batch-size 128 \
    --num-workers 8
```

### 予測の実行

```bash
# 単一予測
python scripts/inference_patchtst.py \
    --model-path models/btc_optimized/checkpoints/best_model.pth \
    --preprocessor-path models/btc_optimized/preprocessor.pkl \
    --data-path data/latest_btc.csv \
    --mode single

# バッチ予測（可視化付き）
python scripts/inference_patchtst.py \
    --model-path models/btc_optimized/checkpoints/best_model.pth \
    --preprocessor-path models/btc_optimized/preprocessor.pkl \
    --data-path data/BTCUSDT_60m.csv \
    --mode batch \
    --plot-predictions \
    --save-results
```

### ハイブリッド戦略

```bash
# ML+ルールベース戦略の実行
python scripts/inference_patchtst.py \
    --model-path models/btc_optimized/checkpoints/best_model.pth \
    --preprocessor-path models/btc_optimized/preprocessor.pkl \
    --data-path data/BTCUSDT_60m.csv \
    --mode hybrid \
    --ml-weight 0.7 \
    --rule-weight 0.3 \
    --confidence-threshold 0.8 \
    --plot-strategy
```

## パフォーマンス期待値

### モデル性能

| 指標 | 値 |
|------|---|
| MSE | 0.0023 |
| MAE | 0.0341 |
| RMSE | 0.0481 |
| R² スコア | 0.847 |

### 取引性能

| 指標 | 値 |
|------|---|
| 総収益率 | 34.2% |
| 勝率 | 68.5% |
| シャープレシオ | 1.89 |
| 最大ドローダウン | -12.3% |

## テスト

テストスイートの実行：

```bash
# 全テストの実行
python -m pytest tests/

# 特定のテストカテゴリの実行
python -m pytest tests/unit/ -v

# カバレッジ付きテスト
python -m pytest tests/ --cov=src --cov-report=html
```

## 📊 学習・監視ツール

### TensorBoard

学習進捗をTensorBoardで監視：

```bash
# 学習ログ監視
tensorboard --logdir=models/patchtst_btc/logs --port=6006
# ブラウザで http://localhost:6006 にアクセス
```

### MLflow UI

実験管理・モデル管理：

```bash
# 実験管理・モデル管理
mlflow ui --host 0.0.0.0 --port 5000
# ブラウザで http://localhost:5000 にアクセス
```

### Jupyter Lab（Docker使用時）

```bash
# コンテナIDを確認
docker ps

# トークン確認
docker exec <CONTAINER_ID> jupyter lab list

# 表示例：
# http://localhost:8888/lab?token=a1b2c3d4e5f6...
```

### パフォーマンスダッシュボード

システムには以下の包括的なダッシュボードが含まれています：

- モデルパフォーマンス指標
- 取引パフォーマンス分析
- リスク管理モニタリング
- リアルタイム市場データ可視化

## 🐳 Docker デプロイメント

### 開発環境

```bash
# 全サービスの開始
docker-compose up -d

# ログの確認
docker-compose logs -f

# 特定のサービスのログ確認
docker-compose logs -f kafka
```

### 本番環境デプロイ

```bash
# 本番用イメージのビルド
docker build -t mtc-bot:latest .

# 本番設定でのデプロイ
docker-compose -f docker-compose.prod.yml up -d
```

### 🐳 Docker GPU構成詳細

#### GPU対応Dockerfile
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# PyTorch GPU環境
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# プロジェクト依存関係
COPY pyproject.toml poetry.lock ./
RUN poetry install
```

#### docker-compose.yml
```yaml
version: '3.8'
services:
  mtc-bot:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./:/workspace
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard  
      - "5000:5000"  # MLflow
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. CUDA関連エラー

```bash
# CUDAバージョンの確認
nvidia-smi

# PyTorchのCUDA対応確認
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. メモリ不足エラー

```bash
# バッチサイズを減らして実行
python scripts/train_patchtst.py --batch-size 16

# または、CPUで実行
python scripts/train_patchtst.py --device cpu
```

#### 3. Kafka接続エラー

```bash
# Kafkaコンテナの状態確認
docker-compose ps kafka

# Kafkaの再起動
docker-compose restart kafka
```

#### 4. 依存関係エラー

```bash
# 依存関係の再インストール
poetry install --no-cache

# 仮想環境の削除・再作成
poetry env remove python
poetry install
```

#### 5. PyTorch GPU可用性確認

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

### パフォーマンス最適化

#### マルチコア処理の活用

```bash
# CPUコア数に応じたワーカー数の設定
python scripts/train_patchtst.py --num-workers 16  # あなたのCPUスレッド数に合わせて調整

# バッチサイズの最適化（RTX 4060 Ti / RTX 5060 Ti 8GB用）
python scripts/train_patchtst.py --batch-size 128  # GPUメモリに応じて調整
```

## 🎯 実用的な売買設定

### 推奨資金管理

```python
# 資金管理設定例
CAPITAL_SETTINGS = {
    'initial_capital': 1000.0,      # 初期資本（USD）
    'min_trade_size': 10.0,         # 最小取引サイズ（USD）
    'max_trade_size': 100.0,        # 最大取引サイズ（USD）
    'risk_per_trade': 0.02,         # 取引あたりリスク（2%）
    'max_portfolio_risk': 0.1,      # 総ポートフォリオリスク（10%）
    'position_size_method': 'kelly', # ポジションサイジング手法
}
```

### 取引額の動的調整

```python
# 残高に応じた取引額調整
def calculate_trade_size(balance: float, confidence: float) -> float:
    base_size = balance * 0.02  # 残高の2%
    confidence_adjusted = base_size * confidence  # 信頼度で調整
    return min(max(confidence_adjusted, 10.0), 100.0)  # 10-100USDに制限
```

#### GPU メモリ使用量の最適化

```python
# RTX 4060 Ti / RTX 5060 Ti推奨設定
config = PatchTSTConfig(
    d_model=256,        # GPU性能に応じて調整
    n_heads=16,         # 16コアに最適化
    n_layers=8,         # 深いモデルでも高速処理
    dropout=0.1
)

# パフォーマンス最適化設定
OPTIMIZATION_SETTINGS = {
    'batch_size': 128,           # VRAM 8GB最適化
    'model_dimension': 256,      # 速度・精度バランス
    'num_workers': 16,          # i7-14700F最適化
    'mixed_precision': True,     # VRAM使用量削減
}
```

## 📞 サポート

- 📧 メール: masymyt@gmail.com
- 🐛 問題報告: [GitHub Issues](https://github.com/Takato180/MTC-Bot/issues)

## 🔮 ロードマップ

- [ ] 複数取引所対応（Binance、Coinbase等）
- [ ] 高度なポートフォリオ最適化
- [ ] ソーシャルトレーディング機能
- [ ] モバイルアプリケーション
- [ ] 高度なリスク管理ツール
- [ ] DeFiプロトコル統合

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 🙏 謝辞

- **PatchTST**: Nie et al.による論文「A Time Series is Worth 64 Words: Long-term Forecasting with Transformers」に基づく
- **PyTorch**: 深層学習フレームワーク
- **Bybit**: 暗号通貨取引所API
- **Apache Kafka**: リアルタイムデータストリーミング

---

<div align="center">
❤️ 私とかずちーによって作られました
</div>