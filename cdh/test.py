'''--------------------------------导入必要的库----------------------------------'''

# gymutil用于解析命令行参数，gymapi用于创建环境，gymtorch用于将numpy数组转换为torch张量
from isaacgym import gymapi, gymtorch
# numpy用于数值计算
import numpy as np
# torch用于神经网络构建 和 优化器构建
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt  # 导入 matplotlib 库
 
# 导入自定义库中的函数
from utils.cdh_utils import draw_sphere,initialize_environment,reset_environment_states
from algorithms.ppo1 import PPO 
 
'''--------------------------------初始化仿真环境----------------------------------'''
 
# 传入自定义参数
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 100, "help": "环境数量"},
    {"name": "--episode_length", "type": int, "default": 300, "help": "每个回合的长度"},
    {"name": "--total_train_episodes", "type": int, "default": 20, "help": "总共训练轮数"},
    {"name": "--batch_size", "type": int, "default": 2, "help": "采样批次大小"},
    {"name": "--buffer_capacity", "type": int, "default": 10, "help": "经验回放池容量"}
]
desc = "5. 完善DRL代码，优化算法，快速训练，实现倒立摆直立。"

gym, sim, args, device,viewer = initialize_environment(custom_parameters=custom_parameters, description=desc)

 
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
print(f"\n自由度数量: {num_dof}")

'''--------------------------------设置环境参数----------------------------------'''
# 设置环境数量
num_envs = args.num_envs
episode_length = args.episode_length
total_train_episodes = args.total_train_episodes
batch_size = args.batch_size
buffer_capacity = args.buffer_capacity
 
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
np.random.seed(0)
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
    dof_props['driveMode'] = (gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_POS)
    dof_props['stiffness'] = (0.0, 10.0)
    dof_props['damping'] = (0.0, 1.0)
    #cart_dof_handle可以获取和设置dof位置和速度，但是力矩不行
    cart_dof_handle = gym.find_actor_dof_handle(env, cartpole_handle, 'slider_to_cart')
    cartpole_dof_handles.append(cart_dof_handle)
    gym.set_actor_dof_properties(env, cartpole_handle, dof_props)

# 绘制环境原点
for env in envs:
    # 绘制球体
    draw_sphere(gym, viewer, env, pos=[0, 0, 4])
    pass
# # 创建初始状态的拷贝，用于重置环境
# initial_state = torch.tensor(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL), device=device)
 



'''--------------------------------初始化PPO算法----------------------------------'''

agent = PPO(n_states=4,  # 状态数
            n_hiddens=16,  # 隐含层数
            n_actions=1,  # 动作数
            actor_lr=1e-2 ,  # 策略网络学习率
            critic_lr=1e-2,  # 价值网络学习率
            lmbda=0.95,  # 优势函数的缩放因子
            epochs=10,  # 一组序列训练的轮次
            eps=0.2,  # PPO中截断范围的参数
            gamma=0.9,  # 折扣因子
            device=device
            )

# 开始仿真
num_episodes = 300  # 总迭代次数
return_list = []  # 保存每个回合的return
step_count=0
'''--------------------------------初始化状态----------------------------------'''
# 初始化状态
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
env_ids = torch.arange(num_envs, device=device)
  
# 初始化GPU上环境变量
gym.prepare_sim(sim)

# 随机初始化GPU上环境变量
reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=0, vel_range=0.1)
 
# ----------------------------------------- #
# 训练--回合更新 on_policy
# ----------------------------------------- #

for i in range(num_episodes):
    # 重置环境
    state = reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=1, vel_range=0.2)  
    done = False  # 任务完成的标记
    episode_return = 0  # 累计每回合的reward

    # 构造数据集，保存每个回合的状态数据
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    while step_count<episode_length:
        step_count+=1
        # print("state: ",state)
        action = agent.take_action(state)  # 动作选择

        # 设置 DOF 驱动力
        dof_actuation_force_tensor = torch.zeros((num_envs * num_dof), device=device, dtype=torch.float)
        # 假设第一个自由度受动作影响
        # print("action:",action.item())
        dof_actuation_force_tensor[::num_dof] = action.squeeze()
        
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(dof_actuation_force_tensor))

        # print("action: ",action)

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 
        # 更新图形
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # # 处理查看器事件
        # gym.sync_frame_time(sim)
        # 获取环境的状态
        gym.refresh_dof_state_tensor(sim)
        # 获取环境的位置和速度
        cart_pos = dof_pos[env_ids, 0] 
        cart_vel = dof_vel[env_ids, 0] 
        pole_pos = dof_pos[env_ids, 1] 
        pole_vel = dof_vel[env_ids, 1]
        # 组合状态
        next_state = torch.stack([cart_pos, cart_vel, pole_pos, pole_vel], dim=1)
 
        reward =  1.0-torch.abs(cart_pos) 
        done = torch.abs(pole_pos) > torch.tensor(np.pi / 4, device=device)
        # print("next_state: ",next_state)
        # print("reward: ",reward)
        # print("done: ",done)
        # 保存每个时刻的状态\动作\...
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 更新状态
        state = next_state
        # 累计回合奖励
        episode_return += reward
    step_count=0
    # 保存每个回合的return
    return_list.append(episode_return)
    # 模型训练
    agent.learn(transition_dict)

    # 将列表转换为张量后再计算均值
    last_returns = torch.stack(return_list[-10:]) if len(return_list) >= 10 else torch.stack(return_list)
    avg_return = torch.mean(last_returns)

    # 打印回合信息
    print(f'iter:{i}, return:{avg_return}')

# 释放资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)