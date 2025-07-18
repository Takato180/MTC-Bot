#!/usr/bin/env python3
"""
ハイブリッドAIトレーディングシステムのデモンストレーション
ルールエンジン + 強化学習 + ベイズ推論 + Bybit手数料シミュレーション
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

# ハイブリッドAIシステムコンポーネント
from src.strategy_service.hybrid_ai.hybrid_strategy import HybridAITradingStrategy
from src.strategy_service.hybrid_ai.bybit_fee_simulator import BybitFeeSimulator, VIPLevel, OrderType, OrderSide

def generate_sample_data(days: int = 90) -> pd.DataFrame:
    """サンプル市場データ生成"""
    np.random.seed(42)
    
    # 日付範囲
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # 価格データ生成（BTC/USDTを想定）
    initial_price = 50000
    price_data = []
    current_price = initial_price
    
    for i in range(len(dates)):
        # トレンドとランダム要素
        trend = np.sin(i / 100) * 0.001  # 長期トレンド
        volatility = np.random.normal(0, 0.02)  # ボラティリティ
        
        # 価格更新
        price_change = trend + volatility
        current_price *= (1 + price_change)
        
        # OHLCV生成
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.normal(1000, 200)
        
        price_data.append({
            'timestamp': dates[i],
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': max(volume, 100)
        })
    
    df = pd.DataFrame(price_data)
    df.set_index('timestamp', inplace=True)
    return df

def demonstrate_rule_engine(strategy, data):
    """ルールエンジンのデモンストレーション"""
    print("\n=== ルールエンジンのデモンストレーション ===")
    
    # 最新データでシグナル生成
    signal, confidence, details = strategy.rule_engine.generate_signal(data.tail(100))
    
    print(f"統合シグナル: {signal.name}")
    print(f"信頼度: {confidence:.3f}")
    print(f"買いスコア: {details['buy_score']:.3f}")
    print(f"売りスコア: {details['sell_score']:.3f}")
    
    print("\n個別ルール結果:")
    for rule_name, rule_detail in details['rule_details'].items():
        print(f"  {rule_name}: {rule_detail['signal']} "
              f"(信頼度: {rule_detail['confidence']:.3f}, "
              f"重み: {rule_detail['weight']:.3f})")
    
    return signal, confidence

def demonstrate_bayesian_optimization(strategy, data):
    """ベイズ最適化のデモンストレーション"""
    print("\n=== ベイズ最適化のデモンストレーション ===")
    
    # 市場レジーム検出
    market_regime = strategy.bayesian_optimizer.detect_market_regime(data.tail(50))
    print(f"検出された市場レジーム: {market_regime}")
    
    # 現在のパラメータ表示
    print("\n現在のパラメータ:")
    for name, param in strategy.bayesian_optimizer.parameters.items():
        print(f"  {name}: {param.current_value:.4f} "
              f"(範囲: {param.min_value:.4f} - {param.max_value:.4f})")
    
    return market_regime

def demonstrate_fee_simulation():
    """手数料シミュレーションのデモンストレーション"""
    print("\n=== Bybit手数料シミュレーションのデモンストレーション ===")
    
    # 通常ユーザーとVIP5の比較
    regular_simulator = BybitFeeSimulator(VIPLevel.REGULAR)
    vip5_simulator = BybitFeeSimulator(VIPLevel.VIP5)
    
    # サンプル取引
    trade_params = {
        'symbol': 'BTCUSDT',
        'side': OrderSide.BUY,
        'order_type': OrderType.MARKET,
        'quantity': 0.1,
        'price': 50000
    }
    
    # 通常ユーザーの取引
    regular_trade = regular_simulator.execute_trade(**trade_params)
    regular_costs = regular_simulator.calculate_total_cost(regular_trade)
    
    # VIP5ユーザーの取引
    vip5_trade = vip5_simulator.execute_trade(**trade_params)
    vip5_costs = vip5_simulator.calculate_total_cost(vip5_trade)
    
    print(f"取引量: {trade_params['quantity']} BTC @ ${trade_params['price']}")
    print(f"名目価値: ${trade_params['quantity'] * trade_params['price']}")
    
    print(f"\n通常ユーザー:")
    print(f"  取引手数料: ${regular_costs['trading_fee']:.2f}")
    print(f"  スリッページ: ${regular_costs['slippage_cost']:.2f}")
    print(f"  総コスト: ${regular_costs['total_cost']:.2f}")
    
    print(f"\nVIP5ユーザー:")
    print(f"  取引手数料: ${vip5_costs['trading_fee']:.2f}")
    print(f"  スリッページ: ${vip5_costs['slippage_cost']:.2f}")
    print(f"  総コスト: ${vip5_costs['total_cost']:.2f}")
    
    savings = regular_costs['total_cost'] - vip5_costs['total_cost']
    print(f"\n手数料削減額: ${savings:.2f}")
    
    return regular_simulator, vip5_simulator

def run_hybrid_strategy_demo(data, strategy):
    """ハイブリッド戦略のデモ実行"""
    print("\n=== ハイブリッド戦略バックテストのデモンストレーション ===")
    
    # バックテスト実行（短期間）
    test_data = data.tail(500)  # 直近500時間のデータ
    
    print(f"バックテスト期間: {test_data.index[0]} ~ {test_data.index[-1]}")
    print(f"データ数: {len(test_data)}時間")
    
    # バックテスト実行
    results = strategy.backtest(test_data)
    
    print(f"\n=== バックテスト結果 ===")
    print(f"総取引数: {results['total_trades']}")
    print(f"初期資本: ${results['initial_capital']:,.2f}")
    print(f"最終資本: ${results['final_capital']:,.2f}")
    print(f"総リターン: {results['total_return_pct']:.2f}%")
    print(f"シャープレシオ: {results['sharpe_ratio']:.3f}")
    print(f"最大ドローダウン: {results['max_drawdown_pct']:.2f}%")
    
    # 月次リターン
    if results['monthly_returns']:
        monthly_avg = np.mean(results['monthly_returns']) * 100
        print(f"平均月次リターン: {monthly_avg:.2f}%")
    
    return results

def create_performance_visualization(results, strategy):
    """パフォーマンス可視化"""
    print("\n=== パフォーマンス可視化 ===")
    
    # 取引履歴からポートフォリオ価値の推移を取得
    portfolio_values = [trade['portfolio_value_after'] for trade in strategy.trade_history]
    timestamps = [trade['timestamp'] for trade in strategy.trade_history]
    
    if len(portfolio_values) > 0:
        plt.figure(figsize=(12, 8))
        
        # ポートフォリオ価値の推移
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, portfolio_values)
        plt.title('ポートフォリオ価値の推移')
        plt.xlabel('時間')
        plt.ylabel('価値 (USDT)')
        plt.xticks(rotation=45)
        
        # 月次リターン
        if results['monthly_returns']:
            plt.subplot(2, 2, 2)
            plt.bar(range(len(results['monthly_returns'])), 
                   [r * 100 for r in results['monthly_returns']])
            plt.title('月次リターン')
            plt.xlabel('月')
            plt.ylabel('リターン (%)')
        
        # シグナル分布
        signal_counts = {}
        for signal_record in strategy.signal_history:
            signal = signal_record['signal']
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        plt.subplot(2, 2, 3)
        plt.pie(signal_counts.values(), labels=signal_counts.keys(), autopct='%1.1f%%')
        plt.title('シグナル分布')
        
        # ルール vs RL 貢献度
        rule_contributions = [s['rule_contribution'] for s in strategy.signal_history]
        rl_contributions = [s['rl_contribution'] for s in strategy.signal_history]
        
        plt.subplot(2, 2, 4)
        plt.scatter(rule_contributions, rl_contributions, alpha=0.6)
        plt.xlabel('ルールエンジン貢献度')
        plt.ylabel('強化学習貢献度')
        plt.title('コンポーネント貢献度分析')
        
        plt.tight_layout()
        plt.savefig('hybrid_ai_performance.png', dpi=300, bbox_inches='tight')
        print("パフォーマンスチャートを 'hybrid_ai_performance.png' に保存しました")

def main():
    """メインデモンストレーション"""
    print("="*60)
    print("ハイブリッドAIトレーディングシステム デモンストレーション")
    print("ルールエンジン + 強化学習 + ベイズ推論 + Bybit手数料シミュレーション")
    print("="*60)
    
    # 1. サンプルデータ生成
    print("\n1. サンプル市場データ生成中...")
    data = generate_sample_data(days=90)
    print(f"生成完了: {len(data)}時間分のデータ")
    
    # 2. ハイブリッド戦略初期化
    print("\n2. ハイブリッド戦略初期化中...")
    strategy = HybridAITradingStrategy(
        initial_capital=10000,
        rule_engine_weight=0.4,
        rl_agent_weight=0.6,
        risk_tolerance=0.02,
        max_position_size=0.3
    )
    print("初期化完了")
    
    # 3. 各コンポーネントのデモンストレーション
    signal, confidence = demonstrate_rule_engine(strategy, data)
    market_regime = demonstrate_bayesian_optimization(strategy, data)
    regular_sim, vip5_sim = demonstrate_fee_simulation()
    
    # 4. 統合システムのデモ
    results = run_hybrid_strategy_demo(data, strategy)
    
    # 5. パフォーマンス可視化
    try:
        create_performance_visualization(results, strategy)
    except Exception as e:
        print(f"可視化エラー: {e}")
    
    # 6. 戦略要約
    print("\n=== 戦略要約 ===")
    summary = strategy.get_strategy_summary()
    print(f"戦略タイプ: {summary['strategy_type']}")
    print(f"実行済み取引数: {summary['trades_executed']}")
    print(f"生成シグナル数: {summary['signals_generated']}")
    
    # 7. 最適化要約
    opt_summary = strategy.bayesian_optimizer.get_optimization_summary()
    if opt_summary['status'] == 'active':
        print(f"\n最適化状態: {opt_summary['status']}")
        print(f"現在の市場レジーム: {opt_summary['current_regime']}")
        print(f"最適化実行回数: {opt_summary['optimization_count']}")
    
    print("\n" + "="*60)
    print("デモンストレーション完了")
    print("="*60)

if __name__ == "__main__":
    main()