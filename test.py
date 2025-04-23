'''--------------------------------导入必要的库----------------------------------'''

# gymutil用于解析命令行参数，gymapi用于创建环境，gymtorch用于将numpy数组转换为torch张量
from isaacgym import gymutil, gymapi, gymtorch
# 导入Isaac Gym环境中的一些函数和类，用于创建地形
from isaacgym.terrain_utils import *
# numpy用于数值计算
import numpy as np
# torch用于神经网络构建 和 优化器构建
import torch
# nn用于神经网络构建
import torch.nn as nn
# optim用于优化器构建
import torch.optim as optim
# 在代码开头启用异常检测
torch.autograd.set_detect_anomaly(True)

'''--------------------------------定义PPO算法超参数----------------------------------'''
# PPO算法的超参数
GAMMA = 0.9
LAMBDA = 0.95
CLIP_PARAM = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
# LR为优化器学习率
LR = 3e-4
NUM_EPOCHS = 10
NUM_MINI_BATCHES = 4
EPISODE_LENGTH = 200  # 每个回合的长度

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
        values = self.value(states).squeeze()
        # 打印 values 的版本号
        print("values version before:", values._version)
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
        # 将回报列表转换为张量，并移动到指定设备
        returns = torch.stack(returns).to(self.device)
        # 计算优势函数，即实际回报与价值估计的差值
        advantages = returns - values

        # 训练策略网络和价值网络
        # 迭代多个轮次更新网络参数
        for _ in range(NUM_EPOCHS):
            # 将数据划分为多个小批量进行训练
            for i in range(NUM_MINI_BATCHES):
                # 随机生成小批量数据的索引
                indices = torch.randint(0, len(states), (len(states) // NUM_MINI_BATCHES,), device=self.device)
                # 根据索引提取小批量的状态
                batch_states = states[indices]
                # 根据索引提取小批量的动作
                batch_actions = actions[indices]
                # 根据索引提取小批量的旧对数概率
                batch_old_log_probs = log_probs[indices]
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

                # 计算策略损失
                # 计算新对数概率与旧对数概率的比值的指数
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                # 计算第一个替代损失
                surr1 = ratio * batch_advantages
                # 对比例进行裁剪，并计算第二个替代损失
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * batch_advantages
                # 取两个替代损失的最小值的负均值作为策略损失
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                # 计算价值网络估计的价值与实际回报的均方误差作为价值损失
                value_loss = (self.value(batch_states).squeeze() - batch_returns).pow(2).mean()

                # 计算总损失，总损失由策略损失、价值损失和熵正则项三部分组成
                # 策略损失 (policy_loss)：衡量策略网络生成的动作分布与最优动作分布之间的差异。
                #                      优化器通过最小化策略损失，让优势函数更大，从而使智能体在环境中获得更高的累积奖励。
                # 价值损失 (value_loss)：使用均方误差计算价值网络估计的状态价值与实际回报之间的误差。
                #                      优化器通过最小化价值损失，让价值网络更准确地估计状态的价值。
                # 熵正则项 (- ENTROPY_COEF * entropy.mean())：熵表示动作分布的不确定性，熵越大，探索性越强。
                #                                           熵正则项前面是负号，优化器在最小化总损失时会增大熵值，
                #             
                total_loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy.mean()
                # 打印相关张量的版本号
                print("total_loss version:", total_loss._version)
                # 更新策略网络和价值网络
                # 清零策略网络优化器的梯度
                self.policy_optimizer.zero_grad()
                # 清零价值网络优化器的梯度
                self.value_optimizer.zero_grad()
                # 反向传播计算梯度，设置 retain_graph=True 以保留计算图
                total_loss.backward(retain_graph=True)
                # 更新策略网络的参数
                self.policy_optimizer.step()
                # 更新价值网络的参数
                self.value_optimizer.step()

'''--------------------------------初始化仿真环境----------------------------------'''
# 初始化gym
gym = gymapi.acquire_gym()

# 转换自定义参数为 custom_parameters 所需格式
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 90, "help": "环境数量"}
]
# 指定使用GPU进行计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("GPU不可用，将使用CPU运行。")

# 调用 parse_arguments 函数
args = gymutil.parse_arguments(description="4. 实现 PPO 算法（策略网络、\
                价值网络、更新步骤等），构建奖励函数，计算奖励，实现DRL训练等。",\
                               custom_parameters=custom_parameters)
# PhysX引擎线程
args.num_threads = 8
# 并行模拟的 PhysX 子场景数量
args.subscenes = 8

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_FLEX:
    print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
    args.physics_engine = gymapi.SIM_PHYSX
sim_params.substeps = 4
sim_params.use_gpu_pipeline = True  # 开启GPU管线
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 4
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = True  # 使用GPU
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

'''--------------------------------加载倒立摆资产----------------------------------'''
# 加载倒立摆资产
asset_root = "resources/cartpole"
asset_file = "urdf/cartpole.urdf"
# 提供默认选项，将基础链接固定为 True
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
# 按照默认选项加载资产
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
# 获取资产的自由度数量
num_dof = gym.get_asset_dof_count(cartpole_asset)
# 打印资产的自由度数量
print(f"\n倒立摆自由度数量: {num_dof}")

'''--------------------------------设置环境参数----------------------------------'''
# 设置环境数量
num_envs = args.num_envs
# 每行放置的环境数量
num_per_row = 30
# 环境的间距
env_spacing = 0.5
# 环境的下限和上限
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
# 创建演员的初始位置和旋转
pose = gymapi.Transform()
# 设置初始位置为z方向高度为2.0
pose.p = gymapi.Vec3(0.0, 0.0, 4.0)
# 设置为单位四元数，即不旋转
pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

'''--------------------------------创建倒立摆环境----------------------------------'''
# 创建倒立摆环境
torch.manual_seed(0)
cartpole_handles = []
cartpole_dof_handles = []
envs = []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    cartpole_handle = gym.create_actor(env, cartpole_asset, pose, "cartpole", i + 1, 1, 0)
    cartpole_handles.append(cartpole_handle)
    dof_props = gym.get_actor_dof_properties(env, cartpole_handle)
    dof_props['driveMode'] = (gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_NONE)
    dof_props['stiffness'] = (0.0, 0.0)
    dof_props['damping'] = (0.0, 0.0)

    cart_dof_handle = gym.find_actor_dof_handle(env, cartpole_handle, 'slider_to_cart')
    cartpole_dof_handles.append(cart_dof_handle)
    gym.set_actor_dof_properties(env, cartpole_handle, dof_props)

# # 创建初始状态的拷贝，用于重置环境
# initial_state = torch.tensor(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL), device=device)
 
'''--------------------------------创建仿真环境的地形----------------------------------'''
# 创建仿真环境的地形
num_terains = 6
terrain_width = 5.
terrain_length = 5.
horizontal_scale = 0.1  # [m]
vertical_scale = 0.005  # [m]
num_rows = int(terrain_width / horizontal_scale)
num_cols = int(terrain_length / horizontal_scale)
heightfield = torch.zeros((num_terains * num_rows, num_cols), dtype=torch.int16, device=device)

def new_sub_terrain():
    return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

# 平地
heightfield[:1 * num_rows, :] = torch.zeros((num_rows, num_cols), dtype=torch.int16, device=device)

# 上斜坡
heightfield[1 * num_rows:2 * num_rows, :] = torch.tensor(sloped_terrain(new_sub_terrain(), slope=0.5).height_field_raw, device=device)

# 下斜坡
heightfield[2 * num_rows:3 * num_rows, :] = torch.tensor(sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw, device=device) + 490
# 将上下斜坡区域超过 340 的值改为 340
heightfield[1 * num_rows:3 * num_rows, :] = torch.clamp(heightfield[1 * num_rows:3 * num_rows, :], max=340)

# 离散地形
heightfield[3 * num_rows:4 * num_rows, :] = torch.tensor(discrete_obstacles_terrain(new_sub_terrain(), max_height=0.25, min_size=0.25, max_size=1., num_rects=20).height_field_raw, device=device)

# 上台阶
heightfield[4 * num_rows:5 * num_rows, :] = torch.tensor(stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.17).height_field_raw, device=device)
# 将上台阶区域超过 340 的值改为 340
heightfield[4 * num_rows:5 * num_rows, :] = torch.clamp(heightfield[4 * num_rows:5 * num_rows, :], max=340)

# 下台阶
heightfield[5 * num_rows:6 * num_rows, :] = torch.tensor(stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=-0.17).height_field_raw, device=device) + 340
# 将下台阶区域小于0 的值改为0
heightfield[5 * num_rows:6 * num_rows, :] = torch.clamp(heightfield[5 * num_rows:6 * num_rows, :], min=0)

# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(heightfield.cpu().numpy(), horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

'''--------------------------------创建Viewer查看器----------------------------------'''
# 创建Viewer查看器
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 设置查看器初始位置和目标位置
cam_pos = gymapi.Vec3(-2.124770, -6.526505, 10.728952)
cam_target = gymapi.Vec3(6, 2.5, 2)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

'''--------------------------------订阅按键功能----------------------------------'''
# 订阅按键功能
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

'''--------------------------------初始化状态----------------------------------'''
# 初始化状态
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
env_ids = torch.arange(num_envs, device=device)

# 初始化GPU上环境变量
gym.prepare_sim(sim)

# 定义随机位置和速度的范围
pos_range = 0.2  # 位置的随机范围
vel_range = 0.5  # 速度的随机范围

# 生成随机的位置和速度
random_positions = (torch.rand((num_envs * num_dof), device=device) - 0.5) * pos_range
random_velocities = (torch.rand((num_envs * num_dof), device=device) - 0.5) * vel_range

# 创建一个张量，用于存储每个环境中每个自由度的状态（位置和速度）
random_state_tensor = torch.zeros((num_envs * num_dof, 2), device=device, dtype=torch.float)
# 赋值随机位置
random_state_tensor[:, 0] = random_positions
# 赋值随机速度
random_state_tensor[:, 1] = random_velocities

# 将 PyTorch 张量转换为 Isaac Gym 可以处理的格式并更新仿真状态
gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(random_state_tensor))

'''--------------------------------初始化PPO算法----------------------------------'''
# 初始化PPO算法
ppo = PPO(input_dim=4, output_dim=1, device=device)

'''--------------------------------开始仿真----------------------------------'''
# 开始仿真
states = []
actions = []
log_probs = []
rewards = []
dones = []
step_count = 0

while not gym.query_viewer_has_closed(viewer):
    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            # 定义随机位置和速度的范围
            pos_range = 0.2  # 位置的随机范围
            vel_range = 1  # 速度的随机范围

            # 生成随机的位置和速度
            random_positions = (torch.rand((num_envs * num_dof), device=device) - 0.5) * pos_range
            random_velocities = (torch.rand((num_envs * num_dof), device=device) - 0.5) * vel_range

            # 创建一个张量，用于存储每个环境中每个自由度的状态（位置和速度）
            random_state_tensor = torch.zeros((num_envs * num_dof, 2), device=device, dtype=torch.float)
            # 赋值随机位置
            random_state_tensor[:, 0] = random_positions
            # 赋值随机速度
            random_state_tensor[:, 1] = random_velocities

            # 将 PyTorch 张量转换为 Isaac Gym 可以处理的格式并更新仿真状态
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(random_state_tensor))

    # for env in envs:
    #     # 绘制球体
    #     pass

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 获取环境的状态
    gym.refresh_dof_state_tensor(sim)
    # 获取环境的位置和速度
    cart_pos = dof_pos[env_ids, 0].squeeze()
    cart_vel = dof_vel[env_ids, 0].squeeze()
    pole_pos = dof_pos[env_ids, 1].squeeze()
    pole_vel = dof_vel[env_ids, 1].squeeze()
    # 组合状态
    state = torch.stack([cart_pos, cart_vel, pole_pos, pole_vel], dim=1)

    # 执行动作
    action, log_prob = ppo.get_action(state)
    actions.append(action)
    log_probs.append(log_prob)
    states.append(state)

    # 设置 DOF 驱动力
    dof_actuation_force_tensor = torch.zeros((num_envs * num_dof), device=device, dtype=torch.float)
    # 假设第一个自由度受动作影响
    dof_actuation_force_tensor[::num_dof] = action.squeeze()
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(dof_actuation_force_tensor))


    # 计算奖励
    reward = 1 - torch.abs(pole_pos) - 0.1 * torch.abs(cart_pos)
    rewards.append(reward)

    # 判断是否完成
    done = torch.abs(pole_pos) > torch.tensor(np.pi / 2, device=device)
    dones.append(done)

    step_count += 1

    # 检查是否达到回合长度
    if step_count % EPISODE_LENGTH == 0:
        # 转换为张量
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        rewards = torch.cat(rewards, dim=0)
        dones = torch.cat(dones, dim=0)

        # 更新PPO算法
        ppo.update(states, actions, log_probs, rewards, dones)

        # 重置存储
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []

        # # 重置环境
        # gym.set_sim_rigid_body_states(sim, initial_state , gymapi.STATE_ALL)

        # 生成随机的位置和速度
        random_positions = (torch.rand((num_envs * num_dof), device=device) - 0.5) * pos_range
        random_velocities = (torch.rand((num_envs * num_dof), device=device) - 0.5) * vel_range

        # 创建一个张量，用于存储每个环境中每个自由度的状态（位置和速度）
        random_state_tensor = torch.zeros((num_envs * num_dof, 2), device=device, dtype=torch.float)
        # 赋值随机位置
        random_state_tensor[:, 0] = random_positions
        # 赋值随机速度
        random_state_tensor[:, 1] = random_velocities

        # 将 PyTorch 张量转换为 Isaac Gym 可以处理的格式并更新仿真状态
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(random_state_tensor))

'''--------------------------------仿真结束清理----------------------------------'''
# 释放资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)