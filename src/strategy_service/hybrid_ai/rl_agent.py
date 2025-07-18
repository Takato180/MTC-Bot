#!/usr/bin/env python3
"""
強化学習エージェント: AI が導き出すロジック
Deep Q-Network (DQN) ベースの取引エージェント
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
import pickle

class ActionType(Enum):
    """アクションタイプ"""
    BUY = 0
    SELL = 1
    HOLD = 2

@dataclass
class TradingState:
    """取引状態"""
    price_features: np.ndarray
    technical_features: np.ndarray
    position: float
    portfolio_value: float
    unrealized_pnl: float
    market_volatility: float

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """経験再生バッファ"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN強化学習エージェント
    市場データから最適な取引行動を学習
    """
    
    def __init__(self, 
                 state_size: int,
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 32,
                 update_target_freq: int = 100,
                 device: str = 'cuda'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # ネットワーク初期化
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # ターゲットネットワークを同期
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 経験再生バッファ
        self.replay_buffer = ReplayBuffer()
        
        # 学習カウンター
        self.step_count = 0
        
    def encode_state(self, state: TradingState) -> np.ndarray:
        """状態エンコーディング"""
        features = np.concatenate([
            state.price_features.flatten(),
            state.technical_features.flatten(),
            [state.position],
            [state.portfolio_value],
            [state.unrealized_pnl],
            [state.market_volatility]
        ])
        return features
    
    def act(self, state: TradingState, training: bool = True) -> ActionType:
        """行動選択"""
        if training and random.random() < self.epsilon:
            return ActionType(random.randint(0, self.action_size - 1))
        
        state_tensor = torch.FloatTensor(self.encode_state(state)).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        action_idx = q_values.argmax().item()
        
        return ActionType(action_idx)
    
    def remember(self, state: TradingState, action: ActionType, reward: float, 
                 next_state: TradingState, done: bool):
        """経験の記録"""
        experience = Experience(
            state=self.encode_state(state),
            action=action.value,
            reward=reward,
            next_state=self.encode_state(next_state),
            done=done
        )
        self.replay_buffer.push(experience)
    
    def replay(self) -> float:
        """経験再生学習"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # ターゲットネットワーク更新
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def calculate_reward(self, 
                        prev_state: TradingState,
                        action: ActionType,
                        current_state: TradingState,
                        transaction_cost: float = 0.001) -> float:
        """報酬計算"""
        
        # 基本報酬: ポートフォリオ価値の変化
        portfolio_return = (current_state.portfolio_value - prev_state.portfolio_value) / prev_state.portfolio_value
        
        # 取引コスト
        cost_penalty = 0.0
        if action != ActionType.HOLD:
            cost_penalty = transaction_cost
        
        # リスク調整
        volatility_penalty = current_state.market_volatility * 0.1
        
        # ポジション保有リスク
        position_risk = abs(current_state.position) * 0.05
        
        # 最終報酬
        reward = portfolio_return - cost_penalty - volatility_penalty - position_risk
        
        # 報酬のクリッピング
        reward = np.clip(reward, -1.0, 1.0)
        
        return reward
    
    def save_model(self, filepath: str):
        """モデル保存"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """モデル読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']

class RLTradingEnvironment:
    """
    強化学習用取引環境
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 10000,
                 transaction_cost: float = 0.001,
                 lookback_window: int = 20):
        
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        self.reset()
    
    def reset(self) -> TradingState:
        """環境リセット"""
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> TradingState:
        """現在の状態取得"""
        current_idx = self.current_step
        
        # 価格特徴量
        price_data = self.data.iloc[current_idx-self.lookback_window:current_idx]
        price_features = np.array([
            price_data['open'].values,
            price_data['high'].values,
            price_data['low'].values,
            price_data['close'].values,
            price_data['volume'].values
        ]).T
        
        # 正規化
        price_features = (price_features - price_features.mean(axis=0)) / (price_features.std(axis=0) + 1e-8)
        
        # テクニカル特徴量
        current_data = self.data.iloc[current_idx]
        technical_features = np.array([
            current_data.get('rsi', 50) / 100,
            current_data.get('macd', 0),
            current_data.get('bb_width', 0),
            current_data.get('atr', 0) / current_data['close']
        ])
        
        # 市場ボラティリティ
        returns = price_data['close'].pct_change().dropna()
        market_volatility = returns.std() if len(returns) > 1 else 0.0
        
        # 未実現損益
        current_price = current_data['close']
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
        else:
            unrealized_pnl = 0.0
        
        return TradingState(
            price_features=price_features,
            technical_features=technical_features,
            position=self.position / 100,  # 正規化
            portfolio_value=self.portfolio_value / self.initial_capital,  # 正規化
            unrealized_pnl=unrealized_pnl / self.initial_capital,  # 正規化
            market_volatility=market_volatility
        )
    
    def step(self, action: ActionType) -> Tuple[TradingState, float, bool]:
        """環境ステップ"""
        prev_state = self._get_state()
        current_price = self.data.iloc[self.current_step]['close']
        
        # アクション実行
        if action == ActionType.BUY and self.position <= 0:
            # 買い注文
            trade_amount = self.cash * 0.95  # 95%の資金を使用
            shares = trade_amount / current_price
            cost = shares * current_price * (1 + self.transaction_cost)
            
            if cost <= self.cash:
                self.cash -= cost
                self.position += shares
                self.entry_price = current_price
                
        elif action == ActionType.SELL and self.position >= 0:
            # 売り注文
            if self.position > 0:
                # ロングポジション決済
                proceeds = self.position * current_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.position = 0
                self.entry_price = 0
            else:
                # ショート注文
                trade_amount = self.cash * 0.95
                shares = trade_amount / current_price
                proceeds = shares * current_price * (1 - self.transaction_cost)
                
                self.cash += proceeds
                self.position -= shares
                self.entry_price = current_price
        
        # ポートフォリオ価値更新
        position_value = self.position * current_price if self.position > 0 else 0
        unrealized_pnl = (current_price - self.entry_price) * self.position if self.position != 0 else 0
        self.portfolio_value = self.cash + position_value + unrealized_pnl
        
        # 次のステップへ
        self.current_step += 1
        
        # 終了判定
        done = self.current_step >= len(self.data) - 1
        
        # 新しい状態
        new_state = self._get_state() if not done else prev_state
        
        return new_state, self.portfolio_value, done
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """ポートフォリオ要約"""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position,
            'return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'entry_price': self.entry_price
        }

class RLTrainingManager:
    """
    強化学習学習管理
    """
    
    def __init__(self, agent: DQNAgent, environment: RLTradingEnvironment):
        self.agent = agent
        self.environment = environment
        self.training_history = []
    
    def train_episode(self) -> Dict[str, Any]:
        """1エピソードの学習"""
        state = self.environment.reset()
        total_reward = 0
        steps = 0
        losses = []
        
        while True:
            # 行動選択
            action = self.agent.act(state, training=True)
            
            # 環境ステップ
            next_state, portfolio_value, done = self.environment.step(action)
            
            # 報酬計算
            reward = self.agent.calculate_reward(state, action, next_state)
            total_reward += reward
            
            # 経験記録
            self.agent.remember(state, action, reward, next_state, done)
            
            # 学習
            loss = self.agent.replay()
            if loss > 0:
                losses.append(loss)
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        portfolio_summary = self.environment.get_portfolio_summary()
        
        episode_result = {
            'total_reward': total_reward,
            'steps': steps,
            'avg_loss': np.mean(losses) if losses else 0,
            'epsilon': self.agent.epsilon,
            'portfolio_return': portfolio_summary['return'],
            'final_portfolio_value': portfolio_summary['portfolio_value']
        }
        
        return episode_result
    
    def train(self, episodes: int = 1000, save_freq: int = 100) -> List[Dict[str, Any]]:
        """学習実行"""
        training_results = []
        
        for episode in range(episodes):
            result = self.train_episode()
            training_results.append(result)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward={result['total_reward']:.4f}, "
                      f"Return={result['portfolio_return']:.4f}, "
                      f"Epsilon={result['epsilon']:.4f}")
            
            if episode % save_freq == 0:
                self.agent.save_model(f'models/rl_agent_episode_{episode}.pth')
        
        return training_results