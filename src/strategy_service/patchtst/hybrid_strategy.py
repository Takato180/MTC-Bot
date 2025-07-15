"""
PatchTST機械学習予測とルールベース指標を組み合わせたハイブリッド戦略

このモジュールは以下を組み合わせた高度な取引戦略を実装します：
1. 価格予測のためのPatchTSTモデル予測
2. シグナル確認のための従来のテクニカル指標
3. リスク管理とポジションサイジング
4. マルチタイムフレーム分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from ..patchtst.trainer import PatchTSTInference
from ..patchtst.data_loader import CryptoDataPreprocessor
from ...strategy_dsl.strategies import RuleStrategy
from ...strategy_dsl.indicator import SMA, EMA, RSI, MACD, BollingerBands


class SignalType(Enum):
    """取引判断のためのシグナルタイプ"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


class ConfidenceLevel(Enum):
    """予測の信頼度レベル"""
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4


class HybridTradingStrategy:
    """
    機械学習予測とルールベース指標を組み合わせたハイブリッド取引戦略
    """
    
    def __init__(self,
                 model_path: str,
                 preprocessor: CryptoDataPreprocessor,
                 ml_weight: float = 0.6,
                 rule_weight: float = 0.4,
                 confidence_threshold: float = 0.7,
                 risk_tolerance: float = 0.02,
                 max_position_size: float = 0.1):
        """
        Args:
            model_path: 学習済みPatchTSTモデルのパス
            preprocessor: データ前処理器
            ml_weight: ML予測の重み (0-1)
            rule_weight: ルールベースシグナルの重み (0-1)
            confidence_threshold: 取引の最小信頼度
            risk_tolerance: 取引あたりの最大リスク（ポートフォリオの割合）
            max_position_size: 最大ポジションサイズ（ポートフォリオの割合）
        """
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.confidence_threshold = confidence_threshold
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        
        # ML推論の初期化
        self.ml_inference = PatchTSTInference(model_path, preprocessor)
        
        # ルールベース指標の初期化
        self.indicators = {
            'sma_short': SMA(period=10),
            'sma_long': SMA(period=50),
            'ema_short': EMA(period=12),
            'ema_long': EMA(period=26),
            'rsi': RSI(period=14),
            'macd': MACD(fast=12, slow=26, signal=9),
            'bb': BollingerBands(period=20, std_multiplier=2)
        }
        
        # 取引履歴
        self.trading_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def calculate_ml_signal(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """
        PatchTST予測を使用したML基盤の取引シグナル計算
        
        Args:
            data: 過去のOHLCVデータ
        
        Returns:
            (シグナルタイプ, 信頼度)のタプル
        """
        try:
            # ML予測の取得
            prediction_result = self.ml_inference.predict_next_prices(data)
            predictions = prediction_result['prediction']
            
            # 価格予測の抽出（終値が予測に含まれていると仮定）
            # 最初の列または特定の列に終値が含まれていると仮定
            current_price = data['close'].iloc[-1]
            
            # 予測価格変化の計算
            # 予測期間の予測価格の平均を取る
            predicted_price = np.mean(predictions[:, 0])  # 最初の列が終値と仮定
            price_change_pct = (predicted_price - current_price) / current_price
            
            # 予測の一貫性に基づく信頼度の計算
            pred_std = np.std(predictions[:, 0])
            confidence = max(0.1, 1.0 - (pred_std / current_price))
            
            # 予測価格変化に基づくシグナル生成
            if price_change_pct > 0.03:  # 3%上昇
                signal = SignalType.STRONG_BUY
            elif price_change_pct > 0.01:  # 1%上昇
                signal = SignalType.BUY
            elif price_change_pct < -0.03:  # 3%下落
                signal = SignalType.STRONG_SELL
            elif price_change_pct < -0.01:  # 1%下落
                signal = SignalType.SELL
            else:
                signal = SignalType.HOLD
            
            return signal, confidence
            
        except Exception as e:
            print(f"MLシグナル計算エラー: {e}")
            return SignalType.HOLD, 0.0
    
    def calculate_rule_based_signal(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """
        テクニカル指標を使用したルールベース取引シグナルの計算
        
        Args:
            data: 過去のOHLCVデータ
        
        Returns:
            (シグナルタイプ, 信頼度)のタプル
        """
        try:
            signals = []
            
            # 指標の計算
            sma_short = self.indicators['sma_short'](data['close'])
            sma_long = self.indicators['sma_long'](data['close'])
            ema_short = self.indicators['ema_short'](data['close'])
            ema_long = self.indicators['ema_long'](data['close'])
            rsi = self.indicators['rsi'](data['close'])
            macd_line, macd_signal, macd_histogram = self.indicators['macd'](data['close'])
            bb_upper, bb_middle, bb_lower = self.indicators['bb'](data['close'])
            
            current_price = data['close'].iloc[-1]
            
            # 移動平均シグナル
            if sma_short.iloc[-1] > sma_long.iloc[-1]:
                signals.append(1)  # 強気
            else:
                signals.append(-1)  # 弱気
            
            # EMAクロスオーバー
            if ema_short.iloc[-1] > ema_long.iloc[-1]:
                signals.append(1)
            else:
                signals.append(-1)
            
            # RSIシグナル
            rsi_current = rsi.iloc[-1]
            if rsi_current < 30:
                signals.append(2)  # 売られ過ぎ - 強い買い
            elif rsi_current < 40:
                signals.append(1)  # 買い
            elif rsi_current > 70:
                signals.append(-2)  # 買われ過ぎ - 強い売り
            elif rsi_current > 60:
                signals.append(-1)  # 売り
            else:
                signals.append(0)  # 中立
            
            # MACDシグナル
            if macd_line.iloc[-1] > macd_signal.iloc[-1]:
                signals.append(1)
            else:
                signals.append(-1)
            
            # ボリンジャーバンド
            if current_price < bb_lower.iloc[-1]:
                signals.append(1)  # 下限線を下回る価格 - 買い
            elif current_price > bb_upper.iloc[-1]:
                signals.append(-1)  # 上限線を上回る価格 - 売り
            else:
                signals.append(0)  # 中立
            
            # シグナルの集約
            signal_sum = sum(signals)
            signal_count = len(signals)
            
            # シグナルの一致度に基づく信頼度の計算
            confidence = abs(signal_sum) / (signal_count * 2)  # 0-1に正規化
            
            # 最終シグナルの決定
            if signal_sum >= 4:
                signal = SignalType.STRONG_BUY
            elif signal_sum >= 2:
                signal = SignalType.BUY
            elif signal_sum <= -4:
                signal = SignalType.STRONG_SELL
            elif signal_sum <= -2:
                signal = SignalType.SELL
            else:
                signal = SignalType.HOLD
            
            return signal, confidence
            
        except Exception as e:
            print(f"ルールベースシグナル計算エラー: {e}")
            return SignalType.HOLD, 0.0
    
    def combine_signals(self, ml_signal: SignalType, ml_confidence: float,
                       rule_signal: SignalType, rule_confidence: float) -> Tuple[SignalType, float]:
        """
        MLとルールベースシグナルの重み付き結合
        
        Args:
            ml_signal: MLシグナルタイプ
            ml_confidence: ML信頼度レベル
            rule_signal: ルールベースシグナルタイプ
            rule_confidence: ルールベース信頼度レベル
        
        Returns:
            (結合シグナル, 結合信頼度)のタプル
        """
        # シグナルを数値に変換
        ml_value = ml_signal.value
        rule_value = rule_signal.value
        
        # シグナルの重み付け
        weighted_ml = ml_value * self.ml_weight * ml_confidence
        weighted_rule = rule_value * self.rule_weight * rule_confidence
        
        # シグナルの結合
        combined_value = weighted_ml + weighted_rule
        
        # 信頼度の結合
        combined_confidence = (ml_confidence * self.ml_weight + 
                             rule_confidence * self.rule_weight)
        
        # シグナルタイプへの変換
        if combined_value >= 1.5:
            combined_signal = SignalType.STRONG_BUY
        elif combined_value >= 0.5:
            combined_signal = SignalType.BUY
        elif combined_value <= -1.5:
            combined_signal = SignalType.STRONG_SELL
        elif combined_value <= -0.5:
            combined_signal = SignalType.SELL
        else:
            combined_signal = SignalType.HOLD
        
        return combined_signal, combined_confidence
    
    def calculate_position_size(self, signal: SignalType, confidence: float,
                              current_price: float, portfolio_value: float) -> float:
        """
        シグナル強度とリスク管理に基づくポジションサイズの計算
        
        Args:
            signal: 取引シグナル
            confidence: シグナル信頼度
            current_price: 現在の資産価格
            portfolio_value: 総ポートフォリオ価値
        
        Returns:
            ポジションサイズ（ロングは正、ショートは負）
        """
        if signal == SignalType.HOLD or confidence < self.confidence_threshold:
            return 0.0
        
        # ベースポジションサイズ
        base_size = self.max_position_size * portfolio_value
        
        # シグナル強度に基づく調整
        signal_multiplier = {
            SignalType.STRONG_BUY: 1.0,
            SignalType.BUY: 0.6,
            SignalType.SELL: -0.6,
            SignalType.STRONG_SELL: -1.0
        }
        
        # 信頼度に基づく調整
        confidence_multiplier = confidence
        
        # 最終ポジションサイズの計算
        position_size = (base_size * signal_multiplier[signal] * confidence_multiplier) / current_price
        
        # リスク管理の適用
        max_risk_size = (portfolio_value * self.risk_tolerance) / current_price
        position_size = np.clip(position_size, -max_risk_size, max_risk_size)
        
        return position_size
    
    def generate_trading_decision(self, data: pd.DataFrame, 
                                portfolio_value: float) -> Dict[str, Any]:
        """
        包括的な取引判断の生成
        
        Args:
            data: 過去のOHLCVデータ
            portfolio_value: 現在のポートフォリオ価値
        
        Returns:
            取引判断と詳細を含む辞書
        """
        # シグナルの計算
        ml_signal, ml_confidence = self.calculate_ml_signal(data)
        rule_signal, rule_confidence = self.calculate_rule_based_signal(data)
        
        # シグナルの結合
        combined_signal, combined_confidence = self.combine_signals(
            ml_signal, ml_confidence, rule_signal, rule_confidence
        )
        
        # 現在価格の取得
        current_price = data['close'].iloc[-1]
        
        # ポジションサイズの計算
        position_size = self.calculate_position_size(
            combined_signal, combined_confidence, current_price, portfolio_value
        )
        
        # 取引判断の作成
        decision = {
            'timestamp': data.index[-1] if hasattr(data, 'index') else pd.Timestamp.now(),
            'current_price': current_price,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'rule_signal': rule_signal,
            'rule_confidence': rule_confidence,
            'combined_signal': combined_signal,
            'combined_confidence': combined_confidence,
            'position_size': position_size,
            'action': 'BUY' if position_size > 0 else 'SELL' if position_size < 0 else 'HOLD',
            'portfolio_value': portfolio_value,
            'risk_metrics': {
                'max_risk_per_trade': self.risk_tolerance,
                'max_position_size': self.max_position_size,
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        return decision
    
    def backtest_strategy(self, data: pd.DataFrame, 
                         initial_capital: float = 10000.0,
                         transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        過去データでハイブリッド戦略のバックテストを実行
        
        Args:
            data: 過去のOHLCVデータ
            initial_capital: 初期資本
            transaction_cost: 取引コスト（取引価値の割合）
        
        Returns:
            バックテスト結果を含む辞書
        """
        results = []
        portfolio_value = initial_capital
        position = 0.0
        
        # 十分なデータがあることを確認
        min_data_points = max(50, self.ml_inference.config.seq_len)
        
        for i in range(min_data_points, len(data)):
            # 現在の時点までのデータを取得
            current_data = data.iloc[:i+1]
            
            # 取引判断の生成
            decision = self.generate_trading_decision(current_data, portfolio_value)
            
            # 取引の実行
            current_price = decision['current_price']
            new_position = decision['position_size']
            
            # 取引の計算
            trade_size = new_position - position
            
            if abs(trade_size) > 0.001:  # 最小取引サイズ
                # 取引コストの計算
                trade_value = abs(trade_size) * current_price
                cost = trade_value * transaction_cost
                
                # ポートフォリオの更新
                portfolio_value -= cost
                position = new_position
                
                # 取引の記録
                results.append({
                    'timestamp': decision['timestamp'],
                    'price': current_price,
                    'position': position,
                    'trade_size': trade_size,
                    'portfolio_value': portfolio_value,
                    'signal': decision['combined_signal'],
                    'confidence': decision['combined_confidence']
                })
        
        # パフォーマンス指標の計算
        if results:
            df_results = pd.DataFrame(results)
            
            # リターンの計算
            df_results['portfolio_return'] = df_results['portfolio_value'].pct_change()
            
            # パフォーマンス指標
            total_return = (portfolio_value - initial_capital) / initial_capital
            winning_trades = sum(1 for r in results if r['trade_size'] * 
                               (data.loc[r['timestamp']:].iloc[1]['close'] - r['price']) > 0)
            total_trades = len(results)
            
            performance = {
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'final_portfolio_value': portfolio_value,
                'max_drawdown': self.calculate_max_drawdown(df_results['portfolio_value']),
                'sharpe_ratio': self.calculate_sharpe_ratio(df_results['portfolio_return']),
                'trade_history': results
            }
        else:
            performance = {
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'final_portfolio_value': initial_capital,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trade_history': []
            }
        
        return performance
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        最大ドローダウンの計算
        
        Args:
            portfolio_values: ポートフォリオ価値のシリーズ
        
        Returns:
            パーセンテージとしての最大ドローダウン
        """
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        シャープレシオの計算
        
        Args:
            returns: リターンのシリーズ
            risk_free_rate: リスクフリーレート（年率）
        
        Returns:
            シャープレシオ
        """
        excess_returns = returns - risk_free_rate / 252  # 日次リスクフリーレート
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
    
    def save_strategy_config(self, filepath: str):
        """
        戦略設定の保存
        
        Args:
            filepath: 設定保存先のパス
        """
        config = {
            'ml_weight': self.ml_weight,
            'rule_weight': self.rule_weight,
            'confidence_threshold': self.confidence_threshold,
            'risk_tolerance': self.risk_tolerance,
            'max_position_size': self.max_position_size,
            'performance_metrics': self.performance_metrics
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_strategy_config(self, filepath: str):
        """
        戦略設定の読み込み
        
        Args:
            filepath: 設定読み込み元のパス
        """
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.ml_weight = config.get('ml_weight', 0.6)
        self.rule_weight = config.get('rule_weight', 0.4)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.risk_tolerance = config.get('risk_tolerance', 0.02)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.performance_metrics = config.get('performance_metrics', self.performance_metrics)


class StrategyOptimizer:
    """
    ハイブリッド戦略パラメータの最適化器
    """
    
    def __init__(self, data: pd.DataFrame, model_path: str, preprocessor: CryptoDataPreprocessor):
        """
        Args:
            data: 最適化用の過去データ
            model_path: 学習済みモデルのパス
            preprocessor: データ前処理器
        """
        self.data = data
        self.model_path = model_path
        self.preprocessor = preprocessor
    
    def optimize_parameters(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optunaを使用した戦略パラメータの最適化
        
        Args:
            n_trials: 最適化試行回数
        
        Returns:
            最適パラメータと結果を含む辞書
        """
        import optuna
        
        def objective(trial):
            # パラメータの提案
            ml_weight = trial.suggest_float('ml_weight', 0.3, 0.8)
            rule_weight = 1.0 - ml_weight
            confidence_threshold = trial.suggest_float('confidence_threshold', 0.5, 0.9)
            risk_tolerance = trial.suggest_float('risk_tolerance', 0.01, 0.05)
            max_position_size = trial.suggest_float('max_position_size', 0.05, 0.2)
            
            # 提案されたパラメータで戦略を作成
            strategy = HybridTradingStrategy(
                model_path=self.model_path,
                preprocessor=self.preprocessor,
                ml_weight=ml_weight,
                rule_weight=rule_weight,
                confidence_threshold=confidence_threshold,
                risk_tolerance=risk_tolerance,
                max_position_size=max_position_size
            )
            
            # バックテストの実行
            try:
                results = strategy.backtest_strategy(self.data)
                
                # リスク調整後リターンの最適化
                objective_value = results['total_return'] / (abs(results['max_drawdown']) + 0.01)
                
                return objective_value
            except Exception as e:
                print(f"最適化試行エラー: {e}")
                return -float('inf')
        
        # 最適化の実行
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }