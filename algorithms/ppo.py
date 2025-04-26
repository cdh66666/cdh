# numpy用于数值计算
import numpy as np
# torch用于神经网络构建 和 优化器构建
import torch
# nn用于神经网络构建
import torch.nn as nn
# optim用于优化器构建
import torch.optim as optim

'''--------------------------------定义PPO算法超参数----------------------------------'''
# PPO算法的超参数
GAMMA = 0.9
CLIP_PARAM = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
# LR为优化器学习率
LR = 3e-4
NUM_EPOCHS = 10
NUM_MINI_BATCHES = 4



'''--------------------------------定义策略网络----------------------------------'''
# 定义策略网络，用于在强化学习中生成动作的概率分布
class PolicyNetwork(nn.Module):
    """
    策略网络类，继承自 nn.Module。用于根据输入状态生成动作分布的均值和标准差。

    参数:
    input_dim (int): 输入状态的维度。
    output_dim (int): 输出动作的维度。
    """
    def __init__(self, input_dim, output_dim):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 定义第一个全连接层，将输入维度映射到 64 维
        self.fc1 = nn.Linear(input_dim, 64)
        # 定义第二个全连接层，将 64 维映射到 64 维
        self.fc2 = nn.Linear(64, 64)
        # 定义全连接层，用于输出动作分布的均值
        self.fc_mean = nn.Linear(64, output_dim)
        # 定义全连接层，用于输出动作分布的标准差
        self.fc_std = nn.Linear(64, output_dim)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        前向传播方法，定义网络的计算流程。

        参数:
        x (torch.Tensor): 输入的状态张量。

        返回:
        tuple: 包含动作分布的均值和标准差的元组。
        """
        # 输入数据通过第一个全连接层，然后经过激活函数
        x = self.activation(self.fc1(x))
        # 数据通过第二个全连接层，然后经过激活函数
        x = self.activation(self.fc2(x))
        # 计算动作分布的均值
        mean = self.fc_mean(x)
        # 计算动作分布的标准差，先将输出值限制在 [-7, 2] 范围内，再取指数
        # 确保标准差最小约为 0.001，最大约为 7.38
        std = torch.clamp(self.fc_std(x), min=-7, max=2).exp()
        return mean, std 


'''--------------------------------定义价值网络----------------------------------'''
# 定义价值网络，用于在强化学习中估计状态的价值
class ValueNetwork(nn.Module):
    """
    价值网络类，继承自 nn.Module。用于根据输入状态估计该状态的价值。

    参数:
    input_dim (int): 输入状态的维度。
    """
    def __init__(self, input_dim):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 定义第一个全连接层，将输入维度映射到 64 维
        self.fc1 = nn.Linear(input_dim, 64)
        # 定义第二个全连接层，将 64 维映射到 64 维
        self.fc2 = nn.Linear(64, 64)
        # 定义全连接层，用于输出状态的价值，输出维度为 1
        self.fc_value = nn.Linear(64, 1)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        前向传播方法，定义网络的计算流程。

        参数:
        x (torch.Tensor): 输入的状态张量。

        返回:
        torch.Tensor: 输入状态对应的价值估计值。
        """
        # 输入数据通过第一个全连接层，然后经过激活函数
        x = self.activation(self.fc1(x))
        # 数据通过第二个全连接层，然后经过激活函数
        x = self.activation(self.fc2(x))
        # 计算输入状态的价值估计值
        value = self.fc_value(x)
        return value 


'''--------------------------------定义PPO算法类----------------------------------'''
# 定义PPO（Proximal Policy Optimization）算法类，用于实现PPO强化学习算法
class PPO:
    def __init__(self, input_dim, output_dim, device):
        """
        初始化PPO算法类。

        参数:
        input_dim (int): 输入状态的维度。
        output_dim (int): 输出动作的维度。
        device (torch.device): 计算设备，如 'cpu' 或 'cuda'。
        """
        # 保存计算设备
        self.device = device
        # 初始化策略网络，并将其移动到指定设备
        self.policy = PolicyNetwork(input_dim, output_dim).to(device)
        # 初始化价值网络，并将其移动到指定设备
        self.value = ValueNetwork(input_dim).to(device)
        # 初始化策略网络的优化器，使用Adam优化器，学习率为LR
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        # 初始化价值网络的优化器，使用Adam优化器，学习率为LR
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LR)

    def get_action(self, state):
        """
        根据当前状态获取动作及其对数概率。

        参数:
        state (torch.Tensor): 当前环境状态。

        返回:
        tuple: 包含动作 (torch.Tensor) 和动作对数概率 (torch.Tensor) 的元组。
        """
        # 确保状态张量在指定设备上
        state = state.to(self.device)
        # 通过策略网络获取动作分布的均值和标准差
        mean, std = self.policy(state)
        # 根据均值和标准差创建正态分布
        dist = torch.distributions.Normal(mean, std)
        # 从正态分布中采样得到动作
        action = dist.sample()
        # 计算动作的对数概率，并对最后一个维度求和
        log_prob = dist.log_prob(action).sum(-1)
        # 直接返回分离后的张量
        return action.detach(), log_prob.detach()

    def calculate_reward(self, state):
        """
        计算奖励值。

        参数:
        state (torch.Tensor): 机器人的状态。
  
        返回:
        torch.Tensor: 计算得到的奖励值。
        """
        # 从 state 张量中提取 cart_pos 和 pole_pos
        # 假设 state 的顺序是 [cart_pos, cart_vel, pole_pos, pole_vel]
        cart_pos = state[:, 0]
        pole_pos = state[:, 2]
        reward = 1 - torch.abs(pole_pos) - 0.1 * torch.abs(cart_pos)
        return reward


    def check_done(self, state):
        """
        根据状态判断回合是否结束。

        参数:
        state (torch.Tensor): 机器人的状态。

        返回:
        torch.Tensor: 布尔类型的张量，指示每个环境是否结束。
        """
        pole_pos = state[:, 2]
        return torch.abs(pole_pos) > torch.tensor(np.pi / 2, device=self.device)



    def update(self, states, actions, log_probs, rewards, dones):
        """
        使用PPO算法更新策略网络和价值网络的参数。

        参数:
        states (torch.Tensor): 环境状态的张量。
        actions (torch.Tensor): 执行动作的张量。
        log_probs (torch.Tensor): 动作的旧对数概率的张量。
        rewards (torch.Tensor): 采取动作获得的奖励的张量。
        dones (torch.Tensor): 环境是否结束的标志张量。
        """
        # 假设输入已经是张量，且在正确的设备上，无需转换
        # 计算优势函数
        # 通过价值网络计算每个状态的价值估计，并去除多余维度
        # print("states:", states)
        # print("states shape:", states.shape)
        values = self.value(states).squeeze()
        # print("values:", values)
        # print("values shape:", values.shape)
        # # 打印 values 的版本号
        # print("values version before:", values._version)
        # 存储每个时间步的回报
        returns = []
        # 初始化回报为最后一个状态的价值估计
        R = values[-1]
        # 反向遍历奖励和结束标志，计算每个时间步的回报
        for r, d in zip(reversed(rewards), reversed(dones)):
            d = d.float()
            # 根据贝尔曼方程更新回报，GAMMA是折扣因子
            R = r + GAMMA * (1 - d) * R
            # 将计算得到的回报插入到列表开头
            returns.insert(0, R)
        # print("returns:", returns)
        
        # 将回报列表转换为张量，并移动到指定设备
        returns = torch.stack(returns).to(self.device)
        # print("returns:", returns)
        # print("returns shape:", returns.shape)
        # 计算优势函数，即实际回报与价值估计的差值
        advantages = returns - values
        # print("advantages:", advantages)
        # print("advantages shape:", advantages.shape)
        print("\n----------------------------------------------\n")
        # 训练策略网络和价值网络
        # 迭代多个轮次更新网络参数
        for _ in range(NUM_EPOCHS):
            # 将数据划分为多个小批量进行训练
            for i in range(NUM_MINI_BATCHES):
                # 随机生成小批量数据的索引
                indices = torch.randint(0, len(states), (len(states) // NUM_MINI_BATCHES,), device=self.device)
                # print("indices:", indices)
                # print("indices shape:", indices.shape)
                # 根据索引提取小批量的状态
                batch_states = states[indices]
                # 根据索引提取小批量的动作
                batch_actions = actions[indices]
                # 根据索引提取小批量的旧对数概率
                batch_old_log_probs = log_probs[indices]
                # print("log_probs:",log_probs)
                # print("batch_old_log_probs:",batch_old_log_probs.shape)
                # print("batch_old_log_probs:",batch_old_log_probs.squeeze().shape)
 
                # 根据索引提取小批量的优势函数值
                batch_advantages = advantages[indices]
                # 根据索引提取小批量的回报值
                batch_returns = returns[indices]

                # 计算新的对数概率和熵
                # 通过策略网络计算小批量状态对应的动作分布的均值和标准差
                mean, std = self.policy(batch_states)
                # 根据均值和标准差创建正态分布对象
                dist = torch.distributions.Normal(mean, std)
                # 计算小批量动作在新分布下的对数概率，并在最后一个维度求和
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                # 计算新分布的熵，并在最后一个维度求和
                entropy = dist.entropy().sum(-1)
                '''
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                '''
                # 计算策略损失
                # 计算新对数概率与旧对数概率的比值的指数
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                # 计算第一个替代损失
                surr1 = ratio * batch_advantages
                # 对比例进行裁剪，并计算第二个替代损失
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * batch_advantages
                # 取两个替代损失的最小值的负均值作为策略损失
                policy_loss = -torch.min(surr1, surr2).mean()



                '''
                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
    
                '''
                # 计算价值损失
                # 计算价值网络估计的价值与实际回报的均方误差作为价值损失
                value_loss = (self.value(batch_states).squeeze() - batch_returns).pow(2).mean()


                '''
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                '''
                # 计算总损失，总损失由策略损失、价值损失和熵正则项三部分组成
                # 策略损失 (policy_loss)：衡量策略网络生成的动作分布与最优动作分布之间的差异。
                #                      优化器通过最小化策略损失，让优势函数更大，从而使智能体在环境中获得更高的累积奖励。
                # 价值损失 (value_loss)：使用均方误差计算价值网络估计的状态价值与实际回报之间的误差。
                #                      优化器通过最小化价值损失，让价值网络更准确地估计状态的价值。
                # 熵正则项 (- ENTROPY_COEF * entropy.mean())：熵表示动作分布的不确定性，熵越大，探索性越强。
                #                                           熵正则项前面是负号，优化器在最小化总损失时会增大熵值，
                total_loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy.mean()
 

                '''
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                '''
                # 更新策略网络和价值网络
                # 清零策略网络优化器的梯度
                self.policy_optimizer.zero_grad()
                # 清零价值网络优化器的梯度
                self.value_optimizer.zero_grad()
 
                # 反向传播计算梯度
                total_loss.backward()
                # 更新策略网络的参数
                self.policy_optimizer.step()
                # 更新价值网络的参数
                self.value_optimizer.step()








