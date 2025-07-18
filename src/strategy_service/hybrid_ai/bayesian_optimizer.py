#!/usr/bin/env python3
"""
ベイズ推論パラメータ最適化
市場の変化に応じてパラメータをリアルタイムに最適化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import scipy.stats as stats
import scipy.optimize as optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Parameter:
    """パラメータ定義"""
    name: str
    min_value: float
    max_value: float
    current_value: float
    prior_mean: float
    prior_std: float
    
@dataclass
class MarketRegime:
    """市場レジーム"""
    name: str
    volatility_range: Tuple[float, float]
    trend_strength_range: Tuple[float, float]
    volume_range: Tuple[float, float]

class BayesianParameterOptimizer:
    """
    ベイズ推論によるパラメータ最適化
    市場状況に応じてパラメータをリアルタイムに調整
    """
    
    def __init__(self, 
                 lookback_window: int = 50,
                 update_frequency: int = 10,
                 confidence_threshold: float = 0.8):
        
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency
        self.confidence_threshold = confidence_threshold
        
        # パラメータ定義
        self.parameters = self._initialize_parameters()
        
        # 市場レジーム定義
        self.market_regimes = self._initialize_market_regimes()
        
        # 最適化履歴
        self.optimization_history = []
        
        # ガウス過程
        self.gp_models = {}
        
        # 市場状況履歴
        self.market_history = []
        
    def _initialize_parameters(self) -> Dict[str, Parameter]:
        """パラメータ初期化"""
        return {
            # ルールエンジンパラメータ
            'rsi_oversold_threshold': Parameter(
                name='rsi_oversold_threshold',
                min_value=20.0, max_value=40.0, current_value=30.0,
                prior_mean=30.0, prior_std=5.0
            ),
            'rsi_overbought_threshold': Parameter(
                name='rsi_overbought_threshold', 
                min_value=60.0, max_value=80.0, current_value=70.0,
                prior_mean=70.0, prior_std=5.0
            ),
            'macd_sensitivity': Parameter(
                name='macd_sensitivity',
                min_value=0.5, max_value=2.0, current_value=1.0,
                prior_mean=1.0, prior_std=0.3
            ),
            'volatility_threshold': Parameter(
                name='volatility_threshold',
                min_value=0.01, max_value=0.05, current_value=0.02,
                prior_mean=0.02, prior_std=0.01
            ),
            
            # 強化学習パラメータ
            'rl_learning_rate': Parameter(
                name='rl_learning_rate',
                min_value=0.0001, max_value=0.01, current_value=0.001,
                prior_mean=0.001, prior_std=0.002
            ),
            'rl_epsilon': Parameter(
                name='rl_epsilon',
                min_value=0.01, max_value=0.5, current_value=0.1,
                prior_mean=0.1, prior_std=0.05
            ),
            
            # リスク管理パラメータ
            'position_size_factor': Parameter(
                name='position_size_factor',
                min_value=0.1, max_value=1.0, current_value=0.5,
                prior_mean=0.5, prior_std=0.2
            ),
            'stop_loss_factor': Parameter(
                name='stop_loss_factor',
                min_value=0.02, max_value=0.1, current_value=0.05,
                prior_mean=0.05, prior_std=0.02
            )
        }
    
    def _initialize_market_regimes(self) -> Dict[str, MarketRegime]:
        """市場レジーム初期化"""
        return {
            'low_volatility': MarketRegime(
                name='low_volatility',
                volatility_range=(0.0, 0.02),
                trend_strength_range=(0.0, 0.3),
                volume_range=(0.0, 1.2)
            ),
            'medium_volatility': MarketRegime(
                name='medium_volatility',
                volatility_range=(0.02, 0.05),
                trend_strength_range=(0.3, 0.7),
                volume_range=(1.2, 2.0)
            ),
            'high_volatility': MarketRegime(
                name='high_volatility',
                volatility_range=(0.05, 1.0),
                trend_strength_range=(0.7, 1.0),
                volume_range=(2.0, 10.0)
            ),
            'trending': MarketRegime(
                name='trending',
                volatility_range=(0.01, 0.1),
                trend_strength_range=(0.5, 1.0),
                volume_range=(1.0, 5.0)
            ),
            'sideways': MarketRegime(
                name='sideways',
                volatility_range=(0.005, 0.03),
                trend_strength_range=(0.0, 0.4),
                volume_range=(0.5, 2.0)
            )
        }
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """市場レジーム検出"""
        # 市場指標計算
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年率ボラティリティ
        
        # トレンド強度
        prices = data['close'].values
        trend_strength = abs(np.corrcoef(np.arange(len(prices)), prices)[0, 1])
        
        # ボリューム指標
        volume_mean = data['volume'].mean()
        volume_std = data['volume'].std()
        current_volume = data['volume'].iloc[-1]
        volume_ratio = (current_volume - volume_mean) / (volume_std + 1e-8)
        
        # レジーム判定
        regime_scores = {}
        
        for regime_name, regime in self.market_regimes.items():
            score = 0
            
            # ボラティリティスコア
            if regime.volatility_range[0] <= volatility <= regime.volatility_range[1]:
                score += 1
            
            # トレンド強度スコア
            if regime.trend_strength_range[0] <= trend_strength <= regime.trend_strength_range[1]:
                score += 1
            
            # ボリュームスコア
            if regime.volume_range[0] <= volume_ratio <= regime.volume_range[1]:
                score += 1
            
            regime_scores[regime_name] = score
        
        # 最高スコアのレジームを選択
        best_regime = max(regime_scores, key=regime_scores.get)
        
        return best_regime
    
    def calculate_performance_metrics(self, 
                                    returns: np.ndarray,
                                    positions: np.ndarray,
                                    parameters: Dict[str, float]) -> Dict[str, float]:
        """パフォーマンス指標計算"""
        if len(returns) == 0:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}
        
        # シャープレシオ
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 最大ドローダウン
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # 勝率
        win_rate = (returns > 0).mean()
        
        # 総合スコア（最適化対象）
        total_score = sharpe_ratio - max_drawdown * 2 + win_rate
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_score': total_score
        }
    
    def bayesian_update(self, 
                       parameter_name: str,
                       observed_performance: float,
                       parameter_value: float) -> float:
        """ベイズ更新"""
        param = self.parameters[parameter_name]
        
        # 事前分布
        prior_mean = param.prior_mean
        prior_var = param.prior_std ** 2
        
        # 尤度関数（パフォーマンスに基づく）
        likelihood_precision = 1.0 / (0.1 ** 2)  # 観測ノイズ
        
        # 事後分布計算
        posterior_precision = 1.0 / prior_var + likelihood_precision
        posterior_var = 1.0 / posterior_precision
        posterior_mean = (prior_mean / prior_var + observed_performance * parameter_value * likelihood_precision) / posterior_precision
        
        # 信頼区間
        posterior_std = np.sqrt(posterior_var)
        
        # 新しい値を提案
        new_value = np.random.normal(posterior_mean, posterior_std)
        new_value = np.clip(new_value, param.min_value, param.max_value)
        
        return new_value
    
    def gaussian_process_optimization(self, 
                                    parameter_name: str,
                                    performance_history: List[Tuple[float, float]],
                                    n_suggestions: int = 5) -> List[float]:
        """ガウス過程による最適化"""
        if len(performance_history) < 3:
            # データが少ない場合はランダム探索
            param = self.parameters[parameter_name]
            return [np.random.uniform(param.min_value, param.max_value) 
                   for _ in range(n_suggestions)]
        
        # データ準備
        X = np.array([p[0] for p in performance_history]).reshape(-1, 1)
        y = np.array([p[1] for p in performance_history])
        
        # ガウス過程モデル
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        try:
            gp.fit(X, y)
            
            # 取得関数（Expected Improvement）
            def expected_improvement(x):
                x = np.array(x).reshape(-1, 1)
                mu, sigma = gp.predict(x, return_std=True)
                
                # 現在の最良値
                f_best = np.max(y)
                
                # Expected Improvement
                with np.errstate(divide='ignore'):
                    z = (mu - f_best) / sigma
                    ei = (mu - f_best) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
                    ei[sigma == 0.0] = 0.0
                
                return -ei.flatten()  # 最小化問題として扱う
            
            # 最適化
            param = self.parameters[parameter_name]
            suggestions = []
            
            for _ in range(n_suggestions):
                result = optimize.minimize_scalar(
                    expected_improvement,
                    bounds=(param.min_value, param.max_value),
                    method='bounded'
                )
                suggestions.append(result.x)
            
            return suggestions
            
        except Exception as e:
            print(f"ガウス過程最適化エラー: {e}")
            # フォールバック
            param = self.parameters[parameter_name]
            return [np.random.uniform(param.min_value, param.max_value) 
                   for _ in range(n_suggestions)]
    
    def adaptive_parameter_update(self, 
                                market_data: pd.DataFrame,
                                performance_data: Dict[str, float],
                                trading_results: Dict[str, Any]) -> Dict[str, float]:
        """適応的パラメータ更新"""
        
        # 市場レジーム検出
        current_regime = self.detect_market_regime(market_data)
        
        # パラメータ更新
        updated_parameters = {}
        
        for param_name, param in self.parameters.items():
            # パフォーマンス履歴取得
            param_history = [
                (record['parameters'].get(param_name, param.current_value), 
                 record['performance']['total_score'])
                for record in self.optimization_history
                if 'parameters' in record and 'performance' in record
            ]
            
            # 最適化手法選択
            if len(param_history) >= 10:
                # ガウス過程最適化
                suggestions = self.gaussian_process_optimization(param_name, param_history)
                new_value = suggestions[0]  # 最良の提案を採用
            else:
                # ベイズ更新
                current_performance = performance_data.get('total_score', 0.0)
                new_value = self.bayesian_update(param_name, current_performance, param.current_value)
            
            # 市場レジーム別調整
            if current_regime == 'high_volatility':
                # 高ボラティリティ時はより保守的に
                if param_name == 'position_size_factor':
                    new_value *= 0.8
                elif param_name == 'stop_loss_factor':
                    new_value *= 0.9
            elif current_regime == 'trending':
                # トレンド時はより積極的に
                if param_name == 'position_size_factor':
                    new_value *= 1.1
                elif param_name == 'macd_sensitivity':
                    new_value *= 1.05
            
            # 値の制限
            new_value = np.clip(new_value, param.min_value, param.max_value)
            
            # 更新
            param.current_value = new_value
            updated_parameters[param_name] = new_value
        
        # 履歴記録
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'market_regime': current_regime,
            'parameters': updated_parameters.copy(),
            'performance': performance_data,
            'trading_results': trading_results
        })
        
        return updated_parameters
    
    def get_parameter_confidence(self, parameter_name: str) -> float:
        """パラメータ信頼度計算"""
        if parameter_name not in self.parameters:
            return 0.0
        
        # 履歴から信頼度を計算
        param_history = [
            record['parameters'].get(parameter_name, 0)
            for record in self.optimization_history
            if 'parameters' in record
        ]
        
        if len(param_history) < 5:
            return 0.5  # 十分なデータがない場合
        
        # 最近の値の安定性を評価
        recent_values = param_history[-5:]
        stability = 1.0 - (np.std(recent_values) / np.mean(recent_values))
        
        return min(max(stability, 0.0), 1.0)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化要約"""
        if not self.optimization_history:
            return {'status': 'no_history'}
        
        latest_record = self.optimization_history[-1]
        
        parameter_summary = {}
        for param_name, param in self.parameters.items():
            parameter_summary[param_name] = {
                'current_value': param.current_value,
                'min_value': param.min_value,
                'max_value': param.max_value,
                'confidence': self.get_parameter_confidence(param_name)
            }
        
        return {
            'status': 'active',
            'current_regime': latest_record.get('market_regime', 'unknown'),
            'parameters': parameter_summary,
            'last_performance': latest_record.get('performance', {}),
            'optimization_count': len(self.optimization_history)
        }
    
    def save_optimization_state(self, filepath: str):
        """最適化状態保存"""
        state = {
            'parameters': {name: {
                'name': param.name,
                'min_value': param.min_value,
                'max_value': param.max_value,
                'current_value': param.current_value,
                'prior_mean': param.prior_mean,
                'prior_std': param.prior_std
            } for name, param in self.parameters.items()},
            'optimization_history': self.optimization_history,
            'market_history': self.market_history
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_optimization_state(self, filepath: str):
        """最適化状態読み込み"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # パラメータ復元
        for name, param_data in state['parameters'].items():
            if name in self.parameters:
                param = self.parameters[name]
                param.current_value = param_data['current_value']
        
        # 履歴復元
        self.optimization_history = state.get('optimization_history', [])
        self.market_history = state.get('market_history', [])