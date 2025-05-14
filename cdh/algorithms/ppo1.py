'''
@Author  ：Yan JP
@Created on Date：2023/4/19 17:31 
'''
# https://blog.csdn.net/dgvv4/article/details/129496576?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-129496576-blog-117329002.235%5Ev29%5Epc_relevant_default_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-129496576-blog-117329002.235%5Ev29%5Epc_relevant_default_base3&utm_relevant_index=6
# 代码用于离散环境的模型
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------- #
# 构建策略网络--actor
# ----------------------------------- #
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc_mean = nn.Linear(n_hiddens, n_actions)
        self.fc_std = nn.Linear(n_hiddens, n_actions)
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

# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)
        # 最后一层使用 Tanh 激活函数
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.tanh(self.fc2(x))  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x


# ----------------------------------- #
# 构建模型
# ----------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    # 动作选择
    def take_action(self, state):
        state = state.to(self.device)
 
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample() 
 
        return action.detach()

    # 训练
    def learn(self, transition_dict):
        # 提取数据集并转换为张量
        states = torch.stack(transition_dict['states']).to(self.device).squeeze().float()
        actions = torch.stack(transition_dict['actions']).to(self.device).float()
        rewards = torch.stack(transition_dict['rewards']).to(self.device).float()
        next_states = torch.stack(transition_dict['next_states']).to(self.device).squeeze().float()
        dones = torch.stack(transition_dict['dones']).to(self.device).float()
        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states).squeeze() 
        # 目标，当前状态的state_value  [b,1]
 
        print("next_q_target shape: ",next_q_target.shape)
        print("rewards shape: ",rewards.shape)
        print("dones shape: ",dones.shape)
   
 

 
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states).squeeze()
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value
        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式 :计算优势函数估计，使用时间差分误差和上一个时间步的优势函数估计
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        old_log_probs = dist.log_prob(actions).sum(-1).detach()
        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1)
 
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states).squeeze(), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()
 