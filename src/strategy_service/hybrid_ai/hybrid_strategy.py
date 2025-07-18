#!/usr/bin/env python3
"""
ハイブリッドAIトレーディング戦略
ルールエンジン + 強化学習 + ベイズ推論の統合システム
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from .rule_engine import RuleEngine, SignalType as RuleSignalType
from .rl_agent import DQNAgent, ActionType, TradingState, RLTradingEnvironment
from .bayesian_optimizer import BayesianParameterOptimizer

class HybridSignalType(Enum):
    """統合シグナルタイプ"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

@dataclass
class HybridTradingSignal:
    """ハイブリッドトレーディングシグナル"""
    signal: HybridSignalType
    confidence: float
    rule_engine_contribution: float
    rl_agent_contribution: float
    combined_score: float
    market_regime: str
    risk_level: float
    position_size: float
    stop_loss: float
    take_profit: float
    reasoning: Dict[str, Any]

class HybridAITradingStrategy:
    """
    ハイブリッドAIトレーディング戦略
    
    従来の予測モデル（CNN+Transformer）を廃止し、
    以下の3つのコンポーネントを統合：
    1. ルールエンジン（人間が定義したロジック）
    2. 強化学習（AIが導き出すロジック）
    3. ベイズ推論（パラメータリアルタイム最適化）
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 rule_engine_weight: float = 0.4,
                 rl_agent_weight: float = 0.6,
                 risk_tolerance: float = 0.02,
                 max_position_size: float = 0.3,
                 lookback_period: int = 100):
        
        self.initial_capital = initial_capital
        self.rule_engine_weight = rule_engine_weight
        self.rl_agent_weight = rl_agent_weight
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.lookback_period = lookback_period
        
        # コンポーネント初期化
        self.rule_engine = RuleEngine(lookback_period)
        self.bayesian_optimizer = BayesianParameterOptimizer()
        
        # 強化学習エージェント（後で初期化）
        self.rl_agent = None
        self.rl_environment = None
        
        # 取引状態
        self.current_position = 0.0
        self.current_cash = initial_capital
        self.portfolio_value = initial_capital
        self.trade_history = []
        self.signal_history = []
        
        # パフォーマンス追跡
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_time': 0.0
        }
        
    def initialize_rl_agent(self, data: pd.DataFrame):
        """強化学習エージェント初期化"""
        # 状態サイズ計算
        state_size = self._calculate_state_size()
        
        # 強化学習エージェント
        self.rl_agent = DQNAgent(
            state_size=state_size,
            learning_rate=self.bayesian_optimizer.parameters['rl_learning_rate'].current_value,
            epsilon=self.bayesian_optimizer.parameters['rl_epsilon'].current_value
        )
        
        # 取引環境
        self.rl_environment = RLTradingEnvironment(
            data=data,
            initial_capital=self.initial_capital
        )
        
    def _calculate_state_size(self) -> int:
        """状態サイズ計算"""
        # 価格特徴量: OHLCV × lookback_period
        price_features = 5 * 20  # 20期間のOHLCV
        
        # テクニカル特徴量: RSI, MACD, BB_width, ATR
        technical_features = 4
        
        # ポートフォリオ特徴量: position, portfolio_value, unrealized_pnl, market_volatility
        portfolio_features = 4
        
        return price_features + technical_features + portfolio_features
    
    def _create_trading_state(self, data: pd.DataFrame, index: int) -> TradingState:
        """取引状態作成"""
        # 価格データ
        start_idx = max(0, index - 20)
        price_data = data.iloc[start_idx:index]
        
        price_features = np.array([
            price_data['open'].values,
            price_data['high'].values,
            price_data['low'].values,
            price_data['close'].values,
            price_data['volume'].values
        ]).T
        
        # パディング
        if len(price_features) < 20:
            padding = np.zeros((20 - len(price_features), 5))
            price_features = np.vstack([padding, price_features])
        
        # 正規化
        price_features = (price_features - price_features.mean(axis=0)) / (price_features.std(axis=0) + 1e-8)
        
        # テクニカル指標
        current_data = data.iloc[index]
        technical_features = np.array([
            current_data.get('rsi', 50) / 100,
            current_data.get('macd', 0),
            current_data.get('bb_width', 0),
            current_data.get('atr', 0) / current_data['close']
        ])
        
        # ボラティリティ
        returns = data['close'].pct_change().iloc[max(0, index-20):index]
        market_volatility = returns.std() if len(returns) > 1 else 0.0
        
        # 未実現損益
        current_price = current_data['close']
        unrealized_pnl = 0.0
        if self.current_position != 0:
            # 簡易計算
            unrealized_pnl = self.current_position * current_price - abs(self.current_position) * current_price
        
        return TradingState(
            price_features=price_features,
            technical_features=technical_features,
            position=self.current_position / 100,  # 正規化
            portfolio_value=self.portfolio_value / self.initial_capital,
            unrealized_pnl=unrealized_pnl / self.initial_capital,
            market_volatility=market_volatility
        )
    
    def generate_hybrid_signal(self, data: pd.DataFrame, index: int) -> HybridTradingSignal:
        """ハイブリッドシグナル生成"""
        
        # 現在のパラメータ取得
        current_params = {name: param.current_value 
                         for name, param in self.bayesian_optimizer.parameters.items()}
        
        # 1. ルールエンジンシグナル
        rule_signal, rule_confidence, rule_details = self.rule_engine.generate_signal(
            data.iloc[max(0, index-self.lookback_period):index+1]
        )
        
        # 2. 強化学習シグナル
        rl_signal = ActionType.HOLD
        rl_confidence = 0.0
        
        if self.rl_agent is not None:
            trading_state = self._create_trading_state(data, index)
            rl_action = self.rl_agent.act(trading_state, training=False)
            
            # ActionTypeをシグナルに変換
            if rl_action == ActionType.BUY:
                rl_signal = ActionType.BUY
                rl_confidence = 0.8
            elif rl_action == ActionType.SELL:
                rl_signal = ActionType.SELL
                rl_confidence = 0.8
            else:
                rl_signal = ActionType.HOLD
                rl_confidence = 0.0
        
        # 3. 市場レジーム検出
        market_regime = self.bayesian_optimizer.detect_market_regime(
            data.iloc[max(0, index-50):index+1]
        )
        
        # 4. シグナル統合
        rule_score = 0.0
        if rule_signal == RuleSignalType.BUY:
            rule_score = rule_confidence
        elif rule_signal == RuleSignalType.SELL:
            rule_score = -rule_confidence
        
        rl_score = 0.0
        if rl_signal == ActionType.BUY:
            rl_score = rl_confidence
        elif rl_signal == ActionType.SELL:
            rl_score = -rl_confidence
        
        # 重み付き統合
        combined_score = (rule_score * self.rule_engine_weight + 
                         rl_score * self.rl_agent_weight)
        
        # 市場レジーム別調整
        if market_regime == 'high_volatility':
            combined_score *= 0.7  # 高ボラティリティ時は保守的
        elif market_regime == 'trending':
            combined_score *= 1.2  # トレンド時は積極的
        
        # 最終シグナル決定
        confidence = abs(combined_score)
        
        if combined_score > 0.6:
            final_signal = HybridSignalType.STRONG_BUY
        elif combined_score > 0.3:
            final_signal = HybridSignalType.BUY
        elif combined_score < -0.6:
            final_signal = HybridSignalType.STRONG_SELL
        elif combined_score < -0.3:
            final_signal = HybridSignalType.SELL
        else:
            final_signal = HybridSignalType.HOLD
        
        # リスク管理
        risk_level = self._calculate_risk_level(data, index, market_regime)
        position_size = self._calculate_position_size(confidence, risk_level, current_params)
        stop_loss = self._calculate_stop_loss(final_signal, risk_level, current_params)
        take_profit = self._calculate_take_profit(final_signal, risk_level, current_params)
        
        # 推論理由
        reasoning = {
            'rule_engine': {
                'signal': rule_signal.name,
                'confidence': rule_confidence,
                'details': rule_details
            },
            'rl_agent': {
                'signal': rl_signal.name,
                'confidence': rl_confidence
            },
            'market_regime': market_regime,
            'parameters_used': current_params
        }
        
        return HybridTradingSignal(
            signal=final_signal,
            confidence=confidence,
            rule_engine_contribution=rule_score * self.rule_engine_weight,
            rl_agent_contribution=rl_score * self.rl_agent_weight,
            combined_score=combined_score,
            market_regime=market_regime,
            risk_level=risk_level,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning
        )
    
    def _calculate_risk_level(self, data: pd.DataFrame, index: int, market_regime: str) -> float:
        """リスクレベル計算"""
        # ボラティリティベースのリスク
        returns = data['close'].pct_change().iloc[max(0, index-20):index]
        volatility_risk = returns.std() if len(returns) > 1 else 0.0
        
        # 市場レジームベースのリスク
        regime_risk = {
            'low_volatility': 0.2,
            'medium_volatility': 0.5,
            'high_volatility': 0.8,
            'trending': 0.3,
            'sideways': 0.4
        }.get(market_regime, 0.5)
        
        # 統合リスク
        total_risk = 0.6 * volatility_risk + 0.4 * regime_risk
        
        return min(max(total_risk, 0.0), 1.0)
    
    def _calculate_position_size(self, confidence: float, risk_level: float, params: Dict[str, float]) -> float:
        """ポジションサイズ計算"""
        base_size = params['position_size_factor'] * confidence
        risk_adjusted_size = base_size * (1 - risk_level)
        
        return min(risk_adjusted_size, self.max_position_size)
    
    def _calculate_stop_loss(self, signal: HybridSignalType, risk_level: float, params: Dict[str, float]) -> float:
        """ストップロス計算"""
        base_stop = params['stop_loss_factor']
        
        # シグナル強度による調整
        if signal in [HybridSignalType.STRONG_BUY, HybridSignalType.STRONG_SELL]:
            base_stop *= 1.5
        
        # リスクレベルによる調整
        risk_adjusted_stop = base_stop * (1 + risk_level)
        
        return min(risk_adjusted_stop, 0.1)  # 最大10%
    
    def _calculate_take_profit(self, signal: HybridSignalType, risk_level: float, params: Dict[str, float]) -> float:
        """テイクプロフィット計算"""
        base_take_profit = params['stop_loss_factor'] * 2  # ストップロスの2倍
        
        # シグナル強度による調整
        if signal in [HybridSignalType.STRONG_BUY, HybridSignalType.STRONG_SELL]:
            base_take_profit *= 1.5
        
        # リスクレベルによる調整
        risk_adjusted_take_profit = base_take_profit * (1 + risk_level * 0.5)
        
        return min(risk_adjusted_take_profit, 0.2)  # 最大20%
    
    def execute_trade(self, signal: HybridTradingSignal, current_price: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """取引実行"""
        trade_result = {
            'timestamp': timestamp,
            'action': 'HOLD',
            'price': current_price,
            'quantity': 0.0,
            'cost': 0.0,
            'portfolio_value_before': self.portfolio_value,
            'portfolio_value_after': self.portfolio_value,
            'signal': signal.signal.name,
            'confidence': signal.confidence,
            'reasoning': signal.reasoning
        }
        
        # 取引実行判定
        if signal.signal in [HybridSignalType.BUY, HybridSignalType.STRONG_BUY]:
            if self.current_position <= 0:  # ロングポジション開設
                trade_amount = self.current_cash * signal.position_size
                quantity = trade_amount / current_price
                transaction_cost = trade_amount * 0.001  # 0.1%の取引コスト
                
                if trade_amount + transaction_cost <= self.current_cash:
                    self.current_cash -= (trade_amount + transaction_cost)
                    self.current_position = quantity
                    
                    trade_result.update({
                        'action': 'BUY',
                        'quantity': quantity,
                        'cost': trade_amount + transaction_cost
                    })
                    
                    self.performance_metrics['total_trades'] += 1
        
        elif signal.signal in [HybridSignalType.SELL, HybridSignalType.STRONG_SELL]:
            if self.current_position > 0:  # ロングポジション決済
                proceeds = self.current_position * current_price
                transaction_cost = proceeds * 0.001
                
                self.current_cash += (proceeds - transaction_cost)
                
                trade_result.update({
                    'action': 'SELL',
                    'quantity': self.current_position,
                    'cost': transaction_cost,
                    'proceeds': proceeds - transaction_cost
                })
                
                self.current_position = 0
        
        # ポートフォリオ価値更新
        position_value = self.current_position * current_price
        self.portfolio_value = self.current_cash + position_value
        
        trade_result['portfolio_value_after'] = self.portfolio_value
        
        # 取引履歴記録
        self.trade_history.append(trade_result)
        
        return trade_result
    
    def backtest(self, data: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """バックテスト実行"""
        
        # 期間設定
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # 初期化
        self.initialize_rl_agent(data)
        
        # 指標計算
        data = self.rule_engine.calculate_indicators(data)
        
        print(f"バックテスト開始: {len(data)}期間")
        print(f"期間: {data.index[0]} ~ {data.index[-1]}")
        
        # バックテスト実行
        for i in range(self.lookback_period, len(data)):
            # シグナル生成
            signal = self.generate_hybrid_signal(data, i)
            
            # 取引実行
            current_price = data.iloc[i]['close']
            timestamp = data.index[i]
            
            trade_result = self.execute_trade(signal, current_price, timestamp)
            
            # シグナル履歴記録
            self.signal_history.append({
                'timestamp': timestamp,
                'signal': signal.signal.name,
                'confidence': signal.confidence,
                'rule_contribution': signal.rule_engine_contribution,
                'rl_contribution': signal.rl_agent_contribution,
                'market_regime': signal.market_regime,
                'executed_trade': trade_result['action'] != 'HOLD'
            })
            
            # パフォーマンス更新（定期的）
            if i % 50 == 0:
                self._update_performance_metrics(data, i)
                
                # ベイズ最適化
                performance_data = self._calculate_current_performance()
                self.bayesian_optimizer.adaptive_parameter_update(
                    data.iloc[max(0, i-100):i+1],
                    performance_data,
                    trade_result
                )
            
            if i % 100 == 0:
                print(f"進捗: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
        
        # 最終パフォーマンス計算
        final_performance = self._calculate_final_performance(data)
        
        return final_performance
    
    def _update_performance_metrics(self, data: pd.DataFrame, current_index: int):
        """パフォーマンス指標更新"""
        if len(self.trade_history) < 2:
            return
        
        # 収益率計算
        returns = []
        for i in range(1, len(self.trade_history)):
            prev_value = self.trade_history[i-1]['portfolio_value_after']
            curr_value = self.trade_history[i]['portfolio_value_after']
            returns.append((curr_value - prev_value) / prev_value)
        
        if len(returns) > 0:
            self.performance_metrics['total_return'] = (self.portfolio_value - self.initial_capital) / self.initial_capital
            
            # シャープレシオ
            returns_array = np.array(returns)
            if returns_array.std() > 0:
                self.performance_metrics['sharpe_ratio'] = returns_array.mean() / returns_array.std() * np.sqrt(252)
            
            # 最大ドローダウン
            portfolio_values = [trade['portfolio_value_after'] for trade in self.trade_history]
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = [(val - max_val) / max_val for val, max_val in zip(portfolio_values, running_max)]
            self.performance_metrics['max_drawdown'] = abs(min(drawdown)) if drawdown else 0.0
    
    def _calculate_current_performance(self) -> Dict[str, float]:
        """現在のパフォーマンス計算"""
        if len(self.trade_history) < 2:
            return {'total_score': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        
        # 勝率
        winning_trades = sum(1 for trade in self.trade_history 
                           if trade['action'] == 'SELL' and 
                           trade['portfolio_value_after'] > trade['portfolio_value_before'])
        
        total_trades = sum(1 for trade in self.trade_history if trade['action'] != 'HOLD')
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # 総合スコア
        total_score = (self.performance_metrics['sharpe_ratio'] * 0.4 + 
                      win_rate * 0.3 + 
                      self.performance_metrics['total_return'] * 0.3 -
                      self.performance_metrics['max_drawdown'] * 0.2)
        
        return {
            'total_score': total_score,
            'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'win_rate': win_rate,
            'total_return': self.performance_metrics['total_return']
        }
    
    def _calculate_final_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """最終パフォーマンス計算"""
        self._update_performance_metrics(data, len(data)-1)
        
        # 詳細分析
        total_trades = len([t for t in self.trade_history if t['action'] != 'HOLD'])
        buy_trades = len([t for t in self.trade_history if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.trade_history if t['action'] == 'SELL'])
        
        # 月次リターン
        monthly_returns = self._calculate_monthly_returns()
        
        # 結果まとめ
        result = {
            'period': f"{data.index[0]} ~ {data.index[-1]}",
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'initial_capital': self.initial_capital,
            'final_capital': self.portfolio_value,
            'total_return': self.performance_metrics['total_return'],
            'total_return_pct': self.performance_metrics['total_return'] * 100,
            'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'max_drawdown_pct': self.performance_metrics['max_drawdown'] * 100,
            'win_rate': self.performance_metrics.get('win_rate', 0.0),
            'monthly_returns': monthly_returns,
            'optimization_summary': self.bayesian_optimizer.get_optimization_summary()
        }
        
        return result
    
    def _calculate_monthly_returns(self) -> List[float]:
        """月次リターン計算"""
        if len(self.trade_history) < 2:
            return []
        
        # 月ごとにグループ化
        monthly_data = {}
        for trade in self.trade_history:
            month_key = trade['timestamp'].strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(trade['portfolio_value_after'])
        
        # 月次リターン計算
        monthly_returns = []
        prev_value = self.initial_capital
        
        for month in sorted(monthly_data.keys()):
            month_end_value = monthly_data[month][-1]
            monthly_return = (month_end_value - prev_value) / prev_value
            monthly_returns.append(monthly_return)
            prev_value = month_end_value
        
        return monthly_returns
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """戦略要約"""
        return {
            'strategy_type': 'Hybrid AI Trading (Rule Engine + RL + Bayesian)',
            'components': {
                'rule_engine': self.rule_engine.get_rule_summary(),
                'rl_agent': {
                    'initialized': self.rl_agent is not None,
                    'state_size': self._calculate_state_size() if self.rl_agent else 0
                },
                'bayesian_optimizer': self.bayesian_optimizer.get_optimization_summary()
            },
            'weights': {
                'rule_engine': self.rule_engine_weight,
                'rl_agent': self.rl_agent_weight
            },
            'performance': self.performance_metrics,
            'trades_executed': len(self.trade_history),
            'signals_generated': len(self.signal_history)
        }