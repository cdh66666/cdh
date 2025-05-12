
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import time

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity, device):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出
        self.device = device

    def add(self, state, action, log_prob, reward, next_state, done):  # 将数据加入buffer，新增 log_prob 参数
        # 确保数据是 torch 张量并移到指定设备
        state = state.to(self.device)
        action = action.to(self.device) if isinstance(action, torch.Tensor) else torch.tensor(action, device=self.device)
        log_prob = log_prob.to(self.device)  # 新增：处理 log_prob
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        next_state = next_state.to(self.device)
        done = torch.tensor([done], device=self.device, dtype=torch.float32)
        self.buffer.append((state, action, log_prob, reward, next_state, done))  # 新增 log_prob

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, log_prob, reward, next_state, done = zip(*transitions)  # 新增 log_prob
        # 将采样的数据堆叠成张量
        state = torch.stack(state).to(self.device)
        action = torch.stack(action).to(self.device)
        log_prob = torch.stack(log_prob).to(self.device)  # 新增：处理 log_prob
        reward = torch.cat(reward).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        done = torch.cat(done).to(self.device)
        return state, action, log_prob, reward, next_state, done  # 新增 log_prob

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
# 策略网络，生成动作概率分布
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.nn.functional.softplus(self.fc_std(x))
        return mean, std

# 价值网络，估计状态价值
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        value = self.fc_value(x)
        return value

# PPO算法类
class PPO:
    def __init__(self, input_dim, output_dim, device, gamma=0.9,
                 clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 lr=0.01, num_epochs=10, num_mini_batches=4, hidden_dim=64):
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        # 传入隐藏层维度
        self.policy = PolicyNetwork(input_dim, output_dim, hidden_dim).to(device)
        self.value = ValueNetwork(input_dim, hidden_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)

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
        k = 1.0  # 可以调整这个值来控制奖励衰减的速度
        # reward = torch.exp(-2.0 * torch.abs(pole_pos))+torch.exp(-1.0 * torch.abs(cart_pos))
        reward = torch.exp(-1.0 * torch.abs(pole_pos))
        # print("pole_pos:", pole_pos)
        # print("reward:", reward)
        return reward.detach()

    def check_done(self, state):
        pole_pos = state[:, 2]
        done = torch.abs(pole_pos) > torch.tensor(np.pi / 2, device=self.device)
        # print("done:", done)
        return done.detach()

    def update(self, states, actions, log_probs, rewards, dones):
        t1 = time.time()
        values = self.value(states).squeeze().detach()
        returns = []
        R = values[-1]
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * (1 - d) * R
            returns.append(R)  # 使用 append 方法
        returns = torch.stack(returns[::-1]).to(self.device)  # 反转列表
        advantages = returns - values

        for _ in range(self.num_epochs):
            for i in range(self.num_mini_batches):
                indices = torch.randint(0, len(states), (len(states) // self.num_mini_batches,), device=self.device)
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = log_probs[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]

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
                self.policy_optimizer.step()
                self.value_optimizer.step()

        t2 = time.time()
        # print("update time:", t2 - t1)
        return returns.mean().item()