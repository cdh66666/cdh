
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import time

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def add_episode(self, states, actions, log_probs, rewards, next_states, dones):
        """
        添加一整个回合的数据到经验回放池
        :param states: 该回合内所有状态的序列
        :param actions: 该回合内所有动作的序列
        :param log_probs: 该回合内所有动作对数概率的序列
        :param rewards: 该回合内所有奖励的序列
        :param next_states: 该回合内所有下一个状态的序列
        :param dones: 该回合内所有是否结束标志的序列
        """
        self.buffer.append((states, actions, log_probs, rewards, next_states, dones))

    def sample(self, batch_size):
        """
        从经验回放池中采样一批回合数据
        :param batch_size: 采样的回合数量
        :return: 采样的状态、动作、对数概率、奖励、下一个状态、是否结束标志的序列
        """
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        states_batch = []
        actions_batch = []
        log_probs_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        for idx in indices:
            states, actions, log_probs, rewards, next_states, dones = self.buffer[idx]
            states_batch.append(states)
            actions_batch.append(actions)
            log_probs_batch.append(log_probs)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
        return (torch.stack(states_batch),
                torch.stack(actions_batch),
                torch.stack(log_probs_batch),
                torch.stack(rewards_batch),
                torch.stack(next_states_batch),
                torch.stack(dones_batch))

    def size(self):
        return len(self.buffer)
    

# 策略网络，生成动作概率分布
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)
        # 中间层使用 ReLU 激活函数
        self.relu = nn.ReLU()
        # 最后一层使用 Tanh 激活函数
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x))
        std = torch.nn.functional.softplus(self.fc_std(x))
        return mean, std

# 价值网络，估计状态价值
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        # 中间层使用 ReLU 激活函数
        self.relu = nn.ReLU()
        # 最后一层使用 Tanh 激活函数
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.tanh(self.fc_value(x))
        return value

# PPO算法类
class PPO:
    def __init__(self, input_dim, output_dim, device, gamma=0.9,
                 clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 lr=0.01, num_epochs=10, num_mini_batches=4, hidden_dim=64,
                 max_grad_norm=0.5, lr_decay=True, total_updates=1000):
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.total_updates = total_updates
        self.update_count = 0
        # 传入隐藏层维度
        self.policy = PolicyNetwork(input_dim, output_dim, hidden_dim).to(device)
        self.value = ValueNetwork(input_dim, hidden_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)
        if self.lr_decay:
            self.policy_lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.policy_optimizer, lr_lambda=lambda epoch: 1 - (self.update_count / self.total_updates))
            self.value_lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.value_optimizer, lr_lambda=lambda epoch: 1 - (self.update_count / self.total_updates))

    def get_action(self, state):
        state = state.to(self.device)
        mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach(), log_prob.detach()

    def calculate_reward(self, state):
        cart_pos = state[:, 0]
        pole_pos = state[:, 2]
        # 修改奖励计算方式，使用指数函数，k 是衰减系数，可根据实际情况调整
        # reward = torch.exp(-1.0 * torch.abs(pole_pos)) + torch.exp(-1.0 * torch.abs(cart_pos))
        reward =  torch.exp(-3.0 * torch.abs(cart_pos))
        return reward.detach()

    def check_done(self, state):
        pole_pos = state[:, 2]
        done = torch.abs(pole_pos) > torch.tensor(np.pi / 4, device=self.device)
        return done.detach()

    def update(self, states_batch, actions_batch, log_probs_batch, rewards_batch, dones_batch):
        t1 = time.time()
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []

        for states, actions, log_probs, rewards, dones in zip(states_batch, actions_batch, log_probs_batch, rewards_batch, dones_batch):
            values = self.value(states).squeeze().detach()
            returns = []
            R = values[-1]
            for r, d in zip(reversed(rewards), reversed(dones)):
                # Convert the boolean tensor d to a floating - point tensor
                d = d.float()
                R = r + self.gamma * (1 - d) * R
                returns.append(R)
            returns = torch.stack(returns[::-1]).to(self.device)
            advantages = returns - values

            all_states.append(states)
            all_actions.append(actions)
            all_old_log_probs.append(log_probs)
            all_returns.append(returns)
            all_advantages.append(advantages)

        # Flatten the data
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        all_old_log_probs = torch.cat(all_old_log_probs, dim=0)
        all_returns = torch.cat(all_returns, dim=0)
        all_advantages = torch.cat(all_advantages, dim=0)

        # 优势标准化
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        for _ in range(self.num_epochs):
            for i in range(self.num_mini_batches):
                indices = torch.randint(0, len(all_states), (len(all_states) // self.num_mini_batches,), device=self.device)
                batch_states = all_states[indices]
                batch_actions = all_actions[indices]
                batch_old_log_probs = all_old_log_probs[indices]
                batch_advantages = all_advantages[indices]
                batch_returns = all_returns[indices]

                mean, std = self.policy(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                # 把熵项添加到策略损失中
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                value_loss = (self.value(batch_states).squeeze() - batch_returns).pow(2).mean()

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)

                self.policy_optimizer.step()
                self.value_optimizer.step()

        if self.lr_decay:
            self.policy_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.update_count += 1

        t2 = time.time()
        # print("update time:", t2 - t1)
        return all_returns.mean().item()