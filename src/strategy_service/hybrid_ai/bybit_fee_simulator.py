#!/usr/bin/env python3
"""
Bybit手数料シミュレーション
リアルな取引環境での手数料計算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

class OrderType(Enum):
    """注文タイプ"""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"

class OrderSide(Enum):
    """注文サイド"""
    BUY = "Buy"
    SELL = "Sell"

class VIPLevel(Enum):
    """VIPレベル"""
    REGULAR = "Regular"
    VIP1 = "VIP1"
    VIP2 = "VIP2"
    VIP3 = "VIP3"
    VIP4 = "VIP4"
    VIP5 = "VIP5"

@dataclass
class FeeStructure:
    """手数料構造"""
    maker_fee: float
    taker_fee: float
    funding_fee_rate: float

@dataclass
class TradeExecution:
    """取引実行結果"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    executed_price: float
    executed_quantity: float
    fee_amount: float
    fee_currency: str
    timestamp: pd.Timestamp
    is_maker: bool

class BybitFeeSimulator:
    """
    Bybit手数料シミュレーター
    リアルな取引環境での手数料を正確に計算
    """
    
    def __init__(self, vip_level: VIPLevel = VIPLevel.REGULAR):
        self.vip_level = vip_level
        self.fee_structures = self._initialize_fee_structures()
        self.trade_history = []
        self.daily_volume = 0.0
        self.monthly_volume = 0.0
        
    def _initialize_fee_structures(self) -> Dict[str, FeeStructure]:
        """手数料構造初期化（2025年1月現在）"""
        return {
            VIPLevel.REGULAR.value: FeeStructure(
                maker_fee=0.001,  # 0.1%
                taker_fee=0.006,  # 0.6%
                funding_fee_rate=0.0001  # 0.01%
            ),
            VIPLevel.VIP1.value: FeeStructure(
                maker_fee=0.0008,  # 0.08%
                taker_fee=0.0055,  # 0.55%
                funding_fee_rate=0.0001
            ),
            VIPLevel.VIP2.value: FeeStructure(
                maker_fee=0.0006,  # 0.06%
                taker_fee=0.005,   # 0.5%
                funding_fee_rate=0.0001
            ),
            VIPLevel.VIP3.value: FeeStructure(
                maker_fee=0.0004,  # 0.04%
                taker_fee=0.0045,  # 0.45%
                funding_fee_rate=0.0001
            ),
            VIPLevel.VIP4.value: FeeStructure(
                maker_fee=0.0002,  # 0.02%
                taker_fee=0.004,   # 0.4%
                funding_fee_rate=0.0001
            ),
            VIPLevel.VIP5.value: FeeStructure(
                maker_fee=0.0,     # 0%
                taker_fee=0.0035,  # 0.35%
                funding_fee_rate=0.0001
            )
        }
    
    def calculate_trading_fee(self, 
                            symbol: str,
                            side: OrderSide,
                            order_type: OrderType,
                            quantity: float,
                            price: float,
                            market_depth: Optional[Dict[str, Any]] = None) -> Tuple[float, bool]:
        """
        取引手数料計算
        
        Returns:
            fee_amount: 手数料金額
            is_maker: Maker注文かどうか
        """
        
        fee_structure = self.fee_structures[self.vip_level.value]
        notional_value = quantity * price
        
        # Maker/Taker判定
        is_maker = self._is_maker_order(order_type, side, price, market_depth)
        
        if is_maker:
            fee_rate = fee_structure.maker_fee
        else:
            fee_rate = fee_structure.taker_fee
        
        # 手数料計算
        fee_amount = notional_value * fee_rate
        
        return fee_amount, is_maker
    
    def _is_maker_order(self, 
                       order_type: OrderType,
                       side: OrderSide,
                       price: float,
                       market_depth: Optional[Dict[str, Any]] = None) -> bool:
        """Maker注文判定"""
        
        if order_type == OrderType.MARKET:
            return False  # 成行注文は常にTaker
        
        if market_depth is None:
            # 市場深度情報がない場合の簡易判定
            return order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]
        
        # 実際の市場深度に基づく判定
        best_bid = market_depth.get('best_bid', 0)
        best_ask = market_depth.get('best_ask', 0)
        
        if side == OrderSide.BUY:
            # 買い注文：指値がbest_bidより低い場合はMaker
            return price < best_bid
        else:
            # 売り注文：指値がbest_askより高い場合はMaker
            return price > best_ask
    
    def calculate_funding_fee(self, 
                            position_size: float,
                            position_value: float,
                            funding_rate: float) -> float:
        """資金調達手数料計算"""
        
        # 資金調達手数料は8時間ごと（1日3回）
        return position_value * funding_rate
    
    def calculate_slippage(self, 
                         order_type: OrderType,
                         side: OrderSide,
                         quantity: float,
                         target_price: float,
                         market_depth: Optional[Dict[str, Any]] = None) -> float:
        """スリッページ計算"""
        
        if order_type != OrderType.MARKET:
            return 0.0  # 指値注文はスリッページなし
        
        if market_depth is None:
            # 簡易スリッページ計算
            base_slippage = 0.0005  # 0.05%
            volume_impact = min(quantity / 10000, 0.002)  # 大口注文の影響
            return base_slippage + volume_impact
        
        # 市場深度に基づく詳細スリッページ計算
        if side == OrderSide.BUY:
            asks = market_depth.get('asks', [])
            return self._calculate_depth_slippage(asks, quantity, target_price)
        else:
            bids = market_depth.get('bids', [])
            return self._calculate_depth_slippage(bids, quantity, target_price)
    
    def _calculate_depth_slippage(self, 
                                 depth_levels: List[Tuple[float, float]],
                                 quantity: float,
                                 target_price: float) -> float:
        """市場深度に基づくスリッページ計算"""
        
        if not depth_levels:
            return 0.001  # デフォルト値
        
        remaining_quantity = quantity
        weighted_price = 0.0
        total_executed = 0.0
        
        for price, available_quantity in depth_levels:
            if remaining_quantity <= 0:
                break
            
            executed_quantity = min(remaining_quantity, available_quantity)
            weighted_price += price * executed_quantity
            total_executed += executed_quantity
            remaining_quantity -= executed_quantity
        
        if total_executed > 0:
            average_execution_price = weighted_price / total_executed
            slippage = abs(average_execution_price - target_price) / target_price
            return slippage
        
        return 0.001
    
    def execute_trade(self, 
                     symbol: str,
                     side: OrderSide,
                     order_type: OrderType,
                     quantity: float,
                     price: float,
                     market_depth: Optional[Dict[str, Any]] = None,
                     timestamp: Optional[pd.Timestamp] = None) -> TradeExecution:
        """取引実行シミュレーション"""
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # 手数料計算
        fee_amount, is_maker = self.calculate_trading_fee(
            symbol, side, order_type, quantity, price, market_depth
        )
        
        # スリッページ計算
        slippage = self.calculate_slippage(
            order_type, side, quantity, price, market_depth
        )
        
        # 実行価格計算
        if order_type == OrderType.MARKET:
            if side == OrderSide.BUY:
                executed_price = price * (1 + slippage)
            else:
                executed_price = price * (1 - slippage)
        else:
            executed_price = price  # 指値注文は指定価格で実行
        
        # 約定数量（通常は全量約定と仮定）
        executed_quantity = quantity
        
        # 取引実行結果
        trade_execution = TradeExecution(
            order_id=f"ORDER_{timestamp.strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            fee_amount=fee_amount,
            fee_currency="USDT",
            timestamp=timestamp,
            is_maker=is_maker
        )
        
        # 取引履歴に追加
        self.trade_history.append(trade_execution)
        
        # 取引量更新
        notional_value = executed_quantity * executed_price
        self.daily_volume += notional_value
        self.monthly_volume += notional_value
        
        return trade_execution
    
    def calculate_total_cost(self, 
                           trade_execution: TradeExecution,
                           include_funding: bool = True,
                           holding_hours: float = 8.0) -> Dict[str, float]:
        """総取引コスト計算"""
        
        notional_value = trade_execution.executed_quantity * trade_execution.executed_price
        
        costs = {
            'trading_fee': trade_execution.fee_amount,
            'slippage_cost': abs(trade_execution.executed_price - trade_execution.price) * trade_execution.executed_quantity,
            'funding_fee': 0.0,
            'total_cost': 0.0
        }
        
        if include_funding:
            # 資金調達手数料（8時間ごと）
            funding_periods = max(1, int(holding_hours / 8))
            funding_rate = self.fee_structures[self.vip_level.value].funding_fee_rate
            costs['funding_fee'] = notional_value * funding_rate * funding_periods
        
        costs['total_cost'] = costs['trading_fee'] + costs['slippage_cost'] + costs['funding_fee']
        
        return costs
    
    def get_fee_analysis(self, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Dict[str, Any]:
        """手数料分析"""
        
        if not self.trade_history:
            return {'status': 'no_trades'}
        
        # 期間フィルタ
        filtered_trades = self.trade_history
        if start_date:
            filtered_trades = [t for t in filtered_trades if t.timestamp >= pd.Timestamp(start_date)]
        if end_date:
            filtered_trades = [t for t in filtered_trades if t.timestamp <= pd.Timestamp(end_date)]
        
        if not filtered_trades:
            return {'status': 'no_trades_in_period'}
        
        # 統計計算
        total_trades = len(filtered_trades)
        total_volume = sum(t.executed_quantity * t.executed_price for t in filtered_trades)
        total_fees = sum(t.fee_amount for t in filtered_trades)
        
        maker_trades = [t for t in filtered_trades if t.is_maker]
        taker_trades = [t for t in filtered_trades if not t.is_maker]
        
        buy_trades = [t for t in filtered_trades if t.side == OrderSide.BUY]
        sell_trades = [t for t in filtered_trades if t.side == OrderSide.SELL]
        
        # 詳細分析
        analysis = {
            'period': {
                'start': filtered_trades[0].timestamp.strftime('%Y-%m-%d'),
                'end': filtered_trades[-1].timestamp.strftime('%Y-%m-%d')
            },
            'trading_summary': {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'total_fees': total_fees,
                'average_fee_per_trade': total_fees / total_trades,
                'fee_rate': total_fees / total_volume * 100  # パーセンテージ
            },
            'maker_taker_breakdown': {
                'maker_trades': len(maker_trades),
                'taker_trades': len(taker_trades),
                'maker_ratio': len(maker_trades) / total_trades * 100,
                'maker_fees': sum(t.fee_amount for t in maker_trades),
                'taker_fees': sum(t.fee_amount for t in taker_trades)
            },
            'buy_sell_breakdown': {
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'buy_volume': sum(t.executed_quantity * t.executed_price for t in buy_trades),
                'sell_volume': sum(t.executed_quantity * t.executed_price for t in sell_trades)
            },
            'vip_level': self.vip_level.value,
            'fee_structure': {
                'maker_fee': self.fee_structures[self.vip_level.value].maker_fee * 100,
                'taker_fee': self.fee_structures[self.vip_level.value].taker_fee * 100
            }
        }
        
        return analysis
    
    def optimize_trading_strategy(self, 
                                expected_trades: int,
                                expected_volume: float) -> Dict[str, Any]:
        """取引戦略最適化提案"""
        
        # VIPレベル別コスト計算
        vip_costs = {}
        
        for vip_level in VIPLevel:
            fee_structure = self.fee_structures[vip_level.value]
            
            # Maker/Taker比率を仮定（通常は50/50）
            maker_ratio = 0.5
            taker_ratio = 0.5
            
            estimated_fees = (
                expected_volume * maker_ratio * fee_structure.maker_fee +
                expected_volume * taker_ratio * fee_structure.taker_fee
            )
            
            vip_costs[vip_level.value] = {
                'estimated_fees': estimated_fees,
                'maker_fee_rate': fee_structure.maker_fee * 100,
                'taker_fee_rate': fee_structure.taker_fee * 100,
                'potential_savings': 0.0
            }
        
        # 現在のレベルとの比較
        current_cost = vip_costs[self.vip_level.value]['estimated_fees']
        
        for vip_level, cost_info in vip_costs.items():
            cost_info['potential_savings'] = current_cost - cost_info['estimated_fees']
        
        # 最適化提案
        optimization_tips = []
        
        # Maker注文の増加
        if self.trade_history:
            current_maker_ratio = sum(1 for t in self.trade_history if t.is_maker) / len(self.trade_history)
            if current_maker_ratio < 0.6:
                optimization_tips.append(
                    f"Maker注文比率を{current_maker_ratio*100:.1f}%から60%以上に増やすことで手数料を削減できます"
                )
        
        # VIPレベルアップ
        if self.vip_level != VIPLevel.VIP5:
            next_vip_levels = [v for v in VIPLevel if v.value > self.vip_level.value]
            if next_vip_levels:
                next_vip = next_vip_levels[0]
                savings = vip_costs[next_vip.value]['potential_savings']
                optimization_tips.append(
                    f"{next_vip.value}にアップグレードすることで約{savings:.2f}USDTの手数料削減が可能です"
                )
        
        return {
            'current_level': self.vip_level.value,
            'vip_cost_comparison': vip_costs,
            'optimization_tips': optimization_tips,
            'expected_trades': expected_trades,
            'expected_volume': expected_volume
        }
    
    def export_trade_history(self, filepath: str):
        """取引履歴エクスポート"""
        if not self.trade_history:
            return
        
        # DataFrameに変換
        trade_data = []
        for trade in self.trade_history:
            trade_data.append({
                'timestamp': trade.timestamp,
                'order_id': trade.order_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'order_type': trade.order_type.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'executed_price': trade.executed_price,
                'executed_quantity': trade.executed_quantity,
                'fee_amount': trade.fee_amount,
                'fee_currency': trade.fee_currency,
                'is_maker': trade.is_maker,
                'notional_value': trade.executed_quantity * trade.executed_price,
                'slippage': abs(trade.executed_price - trade.price) / trade.price * 100
            })
        
        df = pd.DataFrame(trade_data)
        df.to_csv(filepath, index=False)
        print(f"取引履歴を{filepath}に保存しました")
    
    def reset_simulator(self):
        """シミュレーターリセット"""
        self.trade_history = []
        self.daily_volume = 0.0
        self.monthly_volume = 0.0