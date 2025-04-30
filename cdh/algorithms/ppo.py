import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# PPO算法超参数
GAMMA = 0.9
CLIP_PARAM = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
LR = 3e-4
NUM_EPOCHS = 10
NUM_MINI_BATCHES = 4

# 策略网络，生成动作概率分布
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, output_dim)
        self.fc_std = nn.Linear(64, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.nn.functional.softplus(self.fc_std(x))
        return mean, std

# 价值网络，估计状态价值
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_value = nn.Linear(64, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        value = self.fc_value(x)
        return value

# PPO算法类
class PPO:
    def __init__(self, input_dim, output_dim, device):
        self.device = device
        self.policy = PolicyNetwork(input_dim, output_dim).to(device)
        self.value = ValueNetwork(input_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LR)

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
        reward = 1 - torch.abs(pole_pos) - 0.1 * torch.abs(cart_pos)
        return reward.detach()

    def check_done(self, state):
        pole_pos = state[:, 2]
        done = torch.abs(pole_pos) > torch.tensor(np.pi / 2, device=self.device)
        return done.detach()

    def update(self, states, actions, log_probs, rewards, dones):
        values = self.value(states).squeeze().detach()
        returns = []
        R = values[-1]
        for r, d in zip(reversed(rewards), reversed(dones)):
            d = d.float()
            R = r + GAMMA * (1 - d) * R
            returns.insert(0, R)
        returns = torch.stack(returns).to(self.device)
        advantages = returns - values

        for _ in range(NUM_EPOCHS):
            for i in range(NUM_MINI_BATCHES):
                indices = torch.randint(0, len(states), (len(states) // NUM_MINI_BATCHES,), device=self.device)
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
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() 

                value_loss = (self.value(batch_states).squeeze() - batch_returns).pow(2).mean()

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()
        return returns.mean().item()