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
from utils.cdh_utils import draw_sphere,initialize_environment 
from utils.cdh_utils import create_simulation_terrain,reset_environment_states
from algorithms.ppo import PPO, ReplayBuffer  # 导入 ReplayBuffer 类

import time

'''--------------------------------初始化仿真环境----------------------------------'''
 
# 传入自定义参数
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 512, "help": "环境数量"},
    {"name": "--episode_length", "type": int, "default": 300, "help": "每个回合的长度"},
    {"name": "--total_train_episodes", "type": int, "default": 100, "help": "总共训练轮数"},
    {"name": "--batch_size", "type": int, "default": 2, "help": "采样批次大小"},
    {"name": "--buffer_capacity", "type": int, "default": 10, "help": "经验回放池容量"}
]
desc = "5. 完善DRL代码，优化算法，快速训练，实现倒立摆直立。"

gym, sim, args, device = initialize_environment(custom_parameters=custom_parameters, description=desc)

 
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


# # 创建初始状态的拷贝，用于重置环境
# initial_state = torch.tensor(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL), device=device)
 
'''--------------------------------创建仿真环境的地形----------------------------------'''
# 创建仿真环境的地形
create_simulation_terrain(gym, sim, device)
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
for env in envs:
    # 绘制球体
    draw_sphere(gym, viewer, env, pos=[0, 0, 4])
    pass
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

# # 设置第二个 DOF 的期望位置
# # 这里假设期望位置为 0，你可以根据需求修改
# desired_position = 100.0
# # 创建一个形状为 (num_envs * num_dof) 的张量，初始值都为 0
# dof_position_targets = torch.zeros((num_envs * num_dof), device=device, dtype=torch.float)
# # 为每个环境的第二个 DOF 设置期望位置
# dof_position_targets[1::num_dof] = desired_position
# dof_position_targets = dof_position_targets
# # 将 PyTorch 张量转换为 Isaac Gym 可接受的格式
# dof_position_targets_tensor = gymtorch.unwrap_tensor(dof_position_targets)
# # 设置 DOF 位置目标张量
# gym.set_dof_position_target_tensor(sim, dof_position_targets_tensor)


# print("env_ids: ",env_ids)
# 初始化GPU上环境变量
gym.prepare_sim(sim)

# 随机初始化GPU上环境变量
reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=0, vel_range=0.1)

'''--------------------------------初始化PPO算法----------------------------------'''
# 初始化PPO算法
ppo = PPO(input_dim=4, output_dim=1, device=device)

# 初始化经验回放池
replay_buffer = ReplayBuffer(capacity=buffer_capacity, device=device)
 

# 创建实时绘图窗口
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Returns')
ax.set_xlabel('Episodes')
ax.set_ylabel('Returns')
ax.set_title('PPO on CartPole')
ax.legend()
plt.show()

# 开始仿真
step_count = 0
return_list = []  # 用于记录每次更新后的 returns
update_count = 0  # 记录更新次数

# 创建 tqdm 进度条
pbar = tqdm(total=total_train_episodes, desc="Training Progress")

# 用于收集每个回合的数据
episode_states = []
episode_actions = []
episode_log_probs = []
episode_rewards = []
episode_next_states = []
episode_dones = []

while not gym.query_viewer_has_closed(viewer) and update_count < total_train_episodes:
    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            # 随机初始化GPU上环境变量
            reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=0, vel_range=0.1)
            print("重置环境状态")


    # 获取环境的状态
    gym.refresh_dof_state_tensor(sim)
    # 获取环境的位置和速度
    cart_pos = dof_pos[env_ids, 0] 
    cart_vel = dof_vel[env_ids, 0] 
    pole_pos = dof_pos[env_ids, 1] 
    pole_vel = dof_vel[env_ids, 1]
    # 组合状态
    state = torch.stack([cart_pos, cart_vel, pole_pos, pole_vel], dim=1)
    # 执行动作
    action, log_prob = ppo.get_action(state)

    # 记录当前状态
    current_state = state.clone()
    # print("current_state: ",current_state)
    # 设置 DOF 驱动力
    dof_actuation_force_tensor = torch.zeros((num_envs * num_dof), device=device, dtype=torch.float)
    # 假设第一个自由度受动作影响
    dof_actuation_force_tensor[::num_dof] = action.squeeze()

    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(dof_actuation_force_tensor))

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 获取下一个状态
    gym.refresh_dof_state_tensor(sim)
    # 获取环境的位置和速度
    cart_pos = dof_pos[env_ids, 0] 
    cart_vel = dof_vel[env_ids, 0] 
    pole_pos = dof_pos[env_ids, 1] 
    pole_vel = dof_vel[env_ids, 1]
    next_state = torch.stack([cart_pos, cart_vel, pole_pos, pole_vel], dim=1)

    # 计算奖励
    reward = ppo.calculate_reward(next_state).detach()

    # 判断是否完成
    done = ppo.check_done(next_state)

    # 收集当前步骤的数据
    episode_states.append(current_state)
    episode_actions.append(action)
    episode_log_probs.append(log_prob)
    episode_rewards.append(reward)
    episode_next_states.append(next_state)
    episode_dones.append(done)

    step_count += 1

    # 更新图形
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # # 处理查看器事件
    # gym.sync_frame_time(sim)

    # 检查是否达到回合长度
    if step_count % episode_length == 0:
        # 将整个回合的数据添加到经验回放池
        replay_buffer.add_episode(
            torch.stack(episode_states),
            torch.stack(episode_actions),
            torch.stack(episode_log_probs),
            torch.stack(episode_rewards),
            torch.stack(episode_next_states),
            torch.stack(episode_dones)
        )

        # 清空回合数据收集列表
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []



        if replay_buffer.size() >= batch_size:
            # 采样时获取 log_prob
            states, actions, log_probs, rewards, next_states, dones = replay_buffer.sample(batch_size)
            # 更新PPO算法
            ppo_return = ppo.update(states, actions, log_probs, rewards, dones) 
            return_list.append(ppo_return)  # 记录本次更新后的 returns
            # 释放未使用的 GPU 显存
            torch.cuda.empty_cache()
        # 获取 GPU 利用率
        if torch.cuda.is_available():
            gpu_percent = torch.cuda.utilization()
        else:
            gpu_percent = 0

        # 更新 tqdm 进度条，添加 CPU 和 GPU 利用率信息
        pbar.update(1)
        pbar.set_postfix_str(f"GPU: {gpu_percent:.1f}%")
        # print("size:",replay_buffer.size())
        # 实时更新绘图
        episodes_list = list(range(len(return_list)))
        line.set_xdata(episodes_list)
        line.set_ydata(return_list)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
 

        # 重置环境
        reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=1, vel_range=0.2)  

        update_count += 1


# 显示最终绘图
plt.show()

'''--------------------------------仿真结束清理----------------------------------'''
# 关闭 tqdm 进度条
pbar.close()

# 关闭交互模式
plt.ioff()
# 释放资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)