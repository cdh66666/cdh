'''--------------------------------导入必要的库----------------------------------'''

# gymutil用于解析命令行参数，gymapi用于创建环境，gymtorch用于将numpy数组转换为torch张量
from isaacgym import gymapi, gymtorch
# numpy用于数值计算
import numpy as np
# torch用于神经网络构建 和 优化器构建
import torch

import time  # 导入 time 模块
# 导入自定义库中的函数
from utils.cdh_utils import draw_sphere,initialize_environment 
from utils.cdh_utils import create_simulation_terrain,reset_environment_states
from algorithms.ppo import PPO


'''--------------------------------初始化仿真环境----------------------------------'''
 
# 传入自定义参数
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 60, "help": "环境数量"},
    {"name": "--episode_length", "type": int, "default": 100, "help": "每个回合的长度"}
]
desc = "4. 实现 PPO 算法(策略网络、价值网络、更新步骤等),构建奖励函数,计算奖励,实现DRL训练等。"

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
# print(f"\n倒立摆自由度数量: {num_dof}")

'''--------------------------------设置环境参数----------------------------------'''
# 设置环境数量
num_envs = args.num_envs
episode_length = args.episode_length
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

# 随机初始化GPU上环境变量
reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=0.2, vel_range=0.5)

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
prev_ppo_return = None

while not gym.query_viewer_has_closed(viewer):
    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            # 随机初始化GPU上环境变量
            reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=0.2, vel_range=0.5)
 

    for env in envs:
        # 绘制球体
        draw_sphere(gym, viewer, env, pos=[0, 0, 4])
        pass

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
 
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

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
    # print(action, log_prob)
    actions.append(action)
    log_probs.append(log_prob)
    states.append(state)
 
    
    # 设置 DOF 驱动力
    dof_actuation_force_tensor = torch.zeros((num_envs * num_dof), device=device, dtype=torch.float)
    # 假设第一个自由度受动作影响

    dof_actuation_force_tensor[::num_dof] = action.squeeze()
 
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(dof_actuation_force_tensor))
    # print(dof_actuation_force_tensor)

    # 计算奖励
    reward = ppo.calculate_reward(state)
    rewards.append(reward)

    # 判断是否完成
    done = ppo.check_done(state)
    dones.append(done)
 
    step_count += 1

    # 检查是否达到回合长度
    if step_count % episode_length == 0:
        # 记录更新开始时间
        start_time = time.time()
        # 转换为张量
        states = torch.cat(states, dim=0).squeeze()
        actions = torch.cat(actions, dim=0).squeeze()
        log_probs = torch.cat(log_probs, dim=0).squeeze()
        rewards = torch.cat(rewards, dim=0).squeeze()
        dones = torch.cat(dones, dim=0).squeeze()

        # 更新PPO算法
        ppo_return = ppo.update(states, actions, log_probs, rewards, dones) 

        # 记录更新结束时间
        end_time = time.time()
        # 计算更新间隔时间
        update_interval = end_time - start_time

        if prev_ppo_return is not None:
            diff = ppo_return - prev_ppo_return
            print(f"本次更新后 ppo_return: {ppo_return:.4f}，相较于上次变化: {diff:.4f}，更新间隔时间: {update_interval:.4f} 秒")
        else:
            print(f"首次更新后 ppo_return: {ppo_return:.4f}，更新间隔时间: {update_interval:.4f} 秒")

        prev_ppo_return = ppo_return

        # 重置存储
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []

        # 重置环境
        reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=0.2, vel_range=0.5)  
'''--------------------------------仿真结束清理----------------------------------'''
# 释放资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)