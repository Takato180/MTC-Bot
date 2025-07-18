#!/usr/bin/env python3
"""
ルールエンジン: 人間が定義したロジック
従来の予測モデルを廃止し、明確な判断基準を持つトレードルールを実装
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import ta

class SignalType(Enum):
    """シグナルタイプ"""
    BUY = 1
    SELL = -1
    HOLD = 0

class RuleType(Enum):
    """ルールタイプ"""
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"

@dataclass
class TradingRule:
    """トレーディングルール"""
    name: str
    rule_type: RuleType
    weight: float
    enabled: bool = True
    
class RuleEngine:
    """
    ルールエンジン：人間が定義したロジック
    明確な判断基準を持つトレードルールを組み合わせる
    """
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[TradingRule]:
        """ルール初期化"""
        return [
            # テクニカル指標ルール
            TradingRule("rsi_oversold", RuleType.TECHNICAL, 0.15),
            TradingRule("rsi_overbought", RuleType.TECHNICAL, 0.15),
            TradingRule("macd_crossover", RuleType.TECHNICAL, 0.20),
            TradingRule("sma_crossover", RuleType.TECHNICAL, 0.15),
            
            # モメンタムルール
            TradingRule("momentum_breakout", RuleType.MOMENTUM, 0.12),
            TradingRule("momentum_reversal", RuleType.MOMENTUM, 0.08),
            
            # ボラティリティルール
            TradingRule("bollinger_squeeze", RuleType.VOLATILITY, 0.10),
            TradingRule("atr_expansion", RuleType.VOLATILITY, 0.05),
        ]
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標計算"""
        df = data.copy()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # 移動平均
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # ボリンジャーバンド
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # ボリューム指標
        df['volume_sma'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume']).volume_sma()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 価格変動率
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_20'] = df['close'].pct_change(20)
        
        return df
    
    def rule_rsi_oversold(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """RSI売られ過ぎルール"""
        current_rsi = data['rsi'].iloc[-1]
        prev_rsi = data['rsi'].iloc[-2]
        
        if current_rsi < 30 and prev_rsi >= 30:
            confidence = (30 - current_rsi) / 30
            return SignalType.BUY, confidence
        return SignalType.HOLD, 0.0
    
    def rule_rsi_overbought(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """RSI買われ過ぎルール"""
        current_rsi = data['rsi'].iloc[-1]
        prev_rsi = data['rsi'].iloc[-2]
        
        if current_rsi > 70 and prev_rsi <= 70:
            confidence = (current_rsi - 70) / 30
            return SignalType.SELL, confidence
        return SignalType.HOLD, 0.0
    
    def rule_macd_crossover(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """MACDクロスオーバールール"""
        current_macd = data['macd'].iloc[-1]
        current_signal = data['macd_signal'].iloc[-1]
        prev_macd = data['macd'].iloc[-2]
        prev_signal = data['macd_signal'].iloc[-2]
        
        # ゴールデンクロス
        if current_macd > current_signal and prev_macd <= prev_signal:
            histogram = data['macd_histogram'].iloc[-1]
            confidence = min(abs(histogram) / 0.01, 1.0)
            return SignalType.BUY, confidence
        
        # デッドクロス
        elif current_macd < current_signal and prev_macd >= prev_signal:
            histogram = data['macd_histogram'].iloc[-1]
            confidence = min(abs(histogram) / 0.01, 1.0)
            return SignalType.SELL, confidence
        
        return SignalType.HOLD, 0.0
    
    def rule_sma_crossover(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """SMAクロスオーバールール"""
        current_price = data['close'].iloc[-1]
        sma_20 = data['sma_20'].iloc[-1]
        sma_50 = data['sma_50'].iloc[-1]
        prev_sma_20 = data['sma_20'].iloc[-2]
        prev_sma_50 = data['sma_50'].iloc[-2]
        
        # ゴールデンクロス
        if sma_20 > sma_50 and prev_sma_20 <= prev_sma_50:
            price_above_sma = (current_price - sma_20) / sma_20
            confidence = min(abs(price_above_sma) * 10, 1.0)
            return SignalType.BUY, confidence
        
        # デッドクロス
        elif sma_20 < sma_50 and prev_sma_20 >= prev_sma_50:
            price_below_sma = (sma_20 - current_price) / sma_20
            confidence = min(abs(price_below_sma) * 10, 1.0)
            return SignalType.SELL, confidence
        
        return SignalType.HOLD, 0.0
    
    def rule_momentum_breakout(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """モメンタムブレイクアウトルール"""
        returns_5 = data['returns_5'].iloc[-1]
        returns_20 = data['returns_20'].iloc[-1]
        volume_ratio = data['volume_ratio'].iloc[-1]
        
        # 上昇ブレイクアウト
        if returns_5 > 0.05 and returns_20 > 0.02 and volume_ratio > 1.5:
            confidence = min(returns_5 * 10, 1.0) * min(volume_ratio / 2, 1.0)
            return SignalType.BUY, confidence
        
        # 下降ブレイクアウト
        elif returns_5 < -0.05 and returns_20 < -0.02 and volume_ratio > 1.5:
            confidence = min(abs(returns_5) * 10, 1.0) * min(volume_ratio / 2, 1.0)
            return SignalType.SELL, confidence
        
        return SignalType.HOLD, 0.0
    
    def rule_momentum_reversal(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """モメンタムリバーサルルール"""
        returns_5 = data['returns_5'].iloc[-1]
        returns_20 = data['returns_20'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        
        # 過度な下落後の反発
        if returns_5 < -0.08 and returns_20 < -0.05 and rsi < 25:
            confidence = min(abs(returns_5) * 8, 1.0) * (30 - rsi) / 30
            return SignalType.BUY, confidence
        
        # 過度な上昇後の反落
        elif returns_5 > 0.08 and returns_20 > 0.05 and rsi > 75:
            confidence = min(returns_5 * 8, 1.0) * (rsi - 70) / 30
            return SignalType.SELL, confidence
        
        return SignalType.HOLD, 0.0
    
    def rule_bollinger_squeeze(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """ボリンジャーバンドスクイーズルール"""
        bb_width = data['bb_width'].iloc[-1]
        bb_width_avg = data['bb_width'].rolling(20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        
        # スクイーズ後のブレイクアウト
        if bb_width < bb_width_avg * 0.8:
            if current_price > bb_upper:
                confidence = min((current_price - bb_upper) / bb_upper, 1.0)
                return SignalType.BUY, confidence
            elif current_price < bb_lower:
                confidence = min((bb_lower - current_price) / bb_lower, 1.0)
                return SignalType.SELL, confidence
        
        return SignalType.HOLD, 0.0
    
    def rule_atr_expansion(self, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """ATR拡張ルール"""
        atr = data['atr'].iloc[-1]
        atr_avg = data['atr'].rolling(20).mean().iloc[-1]
        returns = data['returns'].iloc[-1]
        
        # ボラティリティ拡張時のトレンドフォロー
        if atr > atr_avg * 1.5:
            if returns > 0.02:
                confidence = min(atr / atr_avg - 1, 1.0) * min(returns * 20, 1.0)
                return SignalType.BUY, confidence
            elif returns < -0.02:
                confidence = min(atr / atr_avg - 1, 1.0) * min(abs(returns) * 20, 1.0)
                return SignalType.SELL, confidence
        
        return SignalType.HOLD, 0.0
    
    def evaluate_rules(self, data: pd.DataFrame) -> Dict[str, Tuple[SignalType, float]]:
        """全ルール評価"""
        # 指標計算
        data_with_indicators = self.calculate_indicators(data)
        
        # ルール評価
        rule_results = {}
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            try:
                method_name = f"rule_{rule.name}"
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    signal, confidence = method(data_with_indicators)
                    rule_results[rule.name] = (signal, confidence)
            except Exception as e:
                print(f"ルール {rule.name} でエラー: {e}")
                rule_results[rule.name] = (SignalType.HOLD, 0.0)
        
        return rule_results
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[SignalType, float, Dict[str, Any]]:
        """
        統合シグナル生成
        
        Returns:
            signal: 統合シグナル
            confidence: 信頼度
            details: 詳細情報
        """
        rule_results = self.evaluate_rules(data)
        
        # 重み付き投票
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        rule_details = {}
        
        for rule in self.rules:
            if rule.name in rule_results:
                signal, confidence = rule_results[rule.name]
                weighted_confidence = confidence * rule.weight
                
                if signal == SignalType.BUY:
                    buy_score += weighted_confidence
                elif signal == SignalType.SELL:
                    sell_score += weighted_confidence
                
                total_weight += rule.weight
                rule_details[rule.name] = {
                    'signal': signal.name,
                    'confidence': confidence,
                    'weight': rule.weight,
                    'weighted_score': weighted_confidence
                }
        
        # 最終判定
        if buy_score > sell_score and buy_score > 0.3:
            final_signal = SignalType.BUY
            final_confidence = min(buy_score / total_weight, 1.0)
        elif sell_score > buy_score and sell_score > 0.3:
            final_signal = SignalType.SELL
            final_confidence = min(sell_score / total_weight, 1.0)
        else:
            final_signal = SignalType.HOLD
            final_confidence = 0.0
        
        details = {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'total_weight': total_weight,
            'rule_details': rule_details,
            'rule_count': len(rule_results)
        }
        
        return final_signal, final_confidence, details
    
    def update_rule_weights(self, performance_data: Dict[str, float]):
        """ルール重み更新（パフォーマンスベース）"""
        for rule in self.rules:
            if rule.name in performance_data:
                performance = performance_data[rule.name]
                # パフォーマンスに基づいて重みを調整
                adjustment = 0.1 * performance
                rule.weight = max(0.01, min(1.0, rule.weight + adjustment))
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """ルール要約"""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'rules': [
                {
                    'name': rule.name,
                    'type': rule.rule_type.value,
                    'weight': rule.weight,
                    'enabled': rule.enabled
                }
                for rule in self.rules
            ]
        }