# MTC-Bot プロジェクト構造説明書

## 📁 整理済みプロジェクト構造

MTC-Botプロジェクトは以下のように整理されています：

```
MTC-Bot/
├── 📄 設定・ドキュメントファイル
│   ├── README.md                    # メインドキュメント（日本語完全版）
│   ├── CONTRIBUTING.md              # 開発者向けガイド
│   ├── LICENSE                      # MITライセンス
│   ├── .gitignore                   # Git除外設定（API キー保護）
│   ├── .env.example                 # 環境変数テンプレート
│   ├── pyproject.toml               # Poetry依存関係管理
│   ├── docker-compose.yml           # Kafka等サービス定義
│   └── PROJECT_STRUCTURE.md         # このファイル
│
├── 📂 src/                          # メインソースコード
│   ├── strategy_service/            # 取引戦略サービス
│   │   ├── patchtst/               # PatchTST機械学習コンポーネント
│   │   │   ├── model.py            # PatchTSTモデル実装
│   │   │   ├── trainer.py          # 学習・推論パイプライン
│   │   │   ├── data_loader.py      # データ処理・前処理
│   │   │   └── hybrid_strategy.py  # ML+ルールベース戦略
│   │   ├── rule_based/             # ルールベース戦略
│   │   ├── optimizer/              # 戦略最適化
│   │   ├── config.py               # 設定管理
│   │   └── live_engine.py          # ライブ取引エンジン
│   │
│   ├── broker_adapters/            # 取引所統合
│   │   └── bybit_adapter.py        # Bybit API統合
│   │
│   ├── data_service/               # データ取得・処理サービス
│   │   └── bybit_ingestor/         # Bybitデータ取得
│   │
│   ├── strategy_dsl/               # 戦略定義言語
│   │   ├── indicator.py            # テクニカル指標
│   │   ├── strategies.py           # 戦略定義
│   │   └── examples/              # 戦略例
│   │
│   ├── api_gateway/                # APIゲートウェイ
│   ├── bot_core/                   # コア取引ロジック
│   ├── user_service/               # ユーザー管理
│   ├── web_ui/                     # ウェブインターフェース
│   └── common/                     # 共通ユーティリティ
│
├── 📂 scripts/                      # ユーティリティスクリプト（整理済み）
│   ├── collect_data.py             # 市場データ収集
│   ├── train_patchtst.py           # モデル学習（マルチコア最適化済み）
│   ├── inference_patchtst.py       # 推論・バックテスト
│   └── fetch_bybit_data.py         # Bybitデータ取得
│
├── 📂 tests/                        # テストファイル（整理済み）
│   ├── unit/                       # 単体テスト
│   ├── integration/                # 統合テスト
│   ├── e2e/                        # エンドツーエンドテスト
│   ├── test_kafka_full.py          # Kafka統合テスト
│   ├── test_kafka_producer.py      # Kafkaプロデューサーテスト
│   └── test_live_engine.py         # ライブエンジンテスト
│
├── 📂 data/                        # データストレージ
│   └── BTCUSDT_1h.csv             # サンプル市場データ
│
├── 📂 models/                      # 学習済みモデル保存先
├── 📂 docs/                        # 詳細ドキュメント
├── 📂 config/                      # 環境別設定
├── 📂 infra/                       # インフラ設定（Terraform、Helm）
└── 📂 mlruns/                      # MLflow実験ログ
```

## 🔧 主な整理内容

### 1. ファイル移動・整理
- **学習・推論スクリプト**: `train_patchtst.py`, `inference_patchtst.py` → `scripts/`フォルダ
- **テストファイル**: `test_*.py` → `tests/`フォルダ
- **階層の明確化**: 散らばっていたファイルを適切なディレクトリに配置

### 2. セキュリティ強化
- **`.gitignore`**: APIキー、機密情報、学習済みモデルを除外
- **`.env.example`**: 安全な環境変数テンプレート提供

### 3. パフォーマンス最適化
- **マルチコア対応**: Intel Core i7 14700F（16スレッド）対応
- **GPU最適化**: RTX 4060 Ti用バッチサイズ・モデル設定
- **デフォルト設定**: 高性能ハードウェア向けに調整

### 4. 日本語化対応
- **README.md**: 完全日本語版、詳細な環境構築手順
- **コメント**: 主要ファイルのコメント日本語化（進行中）
- **エラーメッセージ**: 初心者にもわかりやすい説明

## 推奨使用方法

### 初回セットアップ
```bash
# 1. リポジトリクローン
git clone https://github.com/Takato180/MTC-Bot.git
cd MTC-Bot

# 2. 環境設定
cp .env.example .env
# .envファイルを編集してAPIキーを設定

# 3. 依存関係インストール
poetry install

# 4. GPU環境確認
nvidia-smi

# 5. RTX 50シリーズの場合はPyTorch nightlyをインストール
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 6. GPU動作確認
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"

# 7. Docker起動
docker-compose up -d kafka
```

### データ収集・学習・推論

⚠️ **Poetry 2.0以降注意**: `poetry shell`の代わりに`poetry run`を使用

```bash
# データ収集
poetry run python scripts/collect_data.py --symbol BTCUSDT --days 365

# モデル学習（マルチコア最適化）
poetry run python scripts/train_patchtst.py \
    --data-path data/BTCUSDT_60m.csv \
    --batch-size 128 \
    --num-workers 16 \
    --epochs 100 \
    --device cuda  # RTX 50シリーズではナイトリー版必須

# 推論・バックテスト
poetry run python scripts/inference_patchtst.py \
    --model-path models/best_model.pth \
    --mode hybrid \
    --plot-strategy
```

## 💻 ハードウェア最適化設定

### Intel Core i7 14700F + RTX 4060 Ti / RTX 5060 Ti向け設定

```python
# 学習設定
TRAINING_CONFIG = {
    'batch_size': 128,      # RTX 4060 Ti / RTX 5060 Ti 16GB最適化
    'num_workers': 16,      # CPUスレッド数フル活用
    'd_model': 256,         # GPU性能に合わせたモデルサイズ
    'n_heads': 16,          # マルチコア最適化
    'n_layers': 8,          # 深いモデルでも高速処理
    'device': 'cuda',       # RTX 50シリーズはナイトリー版必須
}

# データローダー設定
DATALOADER_CONFIG = {
    'pin_memory': True,     # GPU転送高速化
    'persistent_workers': True,  # ワーカー再利用
    'prefetch_factor': 4,   # プリフェッチ最適化
}
```

## 🔧 新しい知見と注意事項

### RTX 50シリーズ（Blackwell）対応
- **CUDA Compute Capability**: sm_120（新しいアーキテクチャ）
- **PyTorch安定版**: 未対応（sm_90までのみサポート）
- **解決策**: PyTorch nightly CUDA 12.8版を使用
- **将来性**: 2025年中に安定版でサポート予定

### Poetry 2.0以降の変更
- **`poetry shell`**: デフォルトでは利用不可
- **推奨方法**: `poetry run`でコマンド実行
- **手動アクティベーション**: `poetry env activate`でコマンド取得

### Windows Unicode問題
- **問題**: 日本語文字の文字化け
- **解決策**: `PYTHONIOENCODING=utf-8`環境変数設定
- **永続設定**: `.env`ファイルに追加

## 🔒 セキュリティ対策

### 保護されている情報
- API キーとシークレット
- 学習済みモデル（.pth, .pt ファイル）
- 実験ログ（mlruns/）
- 個人設定ファイル
- データファイル（data/*.csv）

### 安全な設定方法
1. `.env.example`をコピーして`.env`を作成
2. 必要なAPIキーのみを設定
3. テストネット環境から開始
4. 本番取引前に十分なバックテスト実施

## 📊 パフォーマンス指標

### システム要件達成度
- ✅ **マルチコア活用**: 16スレッドフル活用
- ✅ **GPU最適化**: RTX 4060 Ti向けバッチサイズ
- ✅ **メモリ効率**: パッチングによる大幅削減
- ✅ **高速学習**: 最大22倍高速化（PatchTST）

### 期待パフォーマンス
- **学習時間**: 従来比1/10以下
- **推論速度**: リアルタイム対応
- **メモリ使用量**: 効率的なパッチング
- **精度**: R² > 0.8の高精度予測
- **RTX 50シリーズ**: ナイトリー版で高速動作確認済み

## 📝 開発者向け情報

### コード品質
- **型ヒント**: 全関数に型注釈
- **ドキュメント**: 日本語コメント対応
- **テスト**: 包括的テストスイート
- **静的解析**: ruff, mypyによる品質保証

### 拡張性
- **モジュラー設計**: 独立したコンポーネント
- **プラグイン対応**: 新戦略の容易な追加
- **API統合**: 複数取引所対応準備
- **スケーラビリティ**: マイクロサービス対応

---

このプロジェクト構造により、初心者から上級者まで、効率的にMTC-Botを活用できる環境が整備されています。