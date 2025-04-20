import numpy as np
from isaacgym import gymutil, gymapi,gymtorch
from isaacgym.terrain_utils import *
import torch

'''--------------------------------设置仿真参数---------------------------------'''

# initialize gym
gym = gymapi.acquire_gym()

# 转换自定义参数为 custom_parameters 所需格式
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 90, "help": "环境数量"}
]
# 指定使用GPU进行计算
device = torch.device("cuda")
# 调用 parse_arguments 函数
args = gymutil.parse_arguments(description="3. 实现倒立摆pid仿真 ，gpu训练加速等",\
                               custom_parameters=custom_parameters)
#PhysX引擎线程数
args.num_threads=8
#并行模拟的 PhysX 子场景数量
args.subscenes=8
 
# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_FLEX:
    print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
    args.physics_engine = gymapi.SIM_PHYSX
sim_params.substeps = 4
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 4
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()
 
'''--------------------------------加载倒立摆资产----------------------------------'''
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
# 设置随机种子，确保每次运行结果一致
np.random.seed(0)
# 创建倒立摆演员控制句柄列表
cartpole_handles = []
# 创建倒立摆自由度控制句柄列表
cartpole_dof_handles = []
# 创建倒立摆环境列表
envs = []
# 循环创建倒立摆环境和演员
for i in range(num_envs):
    # 根据给定的环境上下限和每行放置的环境数量，计算出当前环境的位置
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    # 添加环境到环境列表中
    envs.append(env)
    # 根据给定的环境,资产,位置,名称,碰撞组,碰撞掩码,切片id创建演员
    cartpole_handle = gym.create_actor(env, cartpole_asset, pose, "cartpole", i+1, 1, 0)
    # 将演员句柄添加到演员控制句柄列表中
    cartpole_handles.append(cartpole_handle)
    # 获取倒立摆演员的自由度属性
    dof_props = gym.get_actor_dof_properties(env, cartpole_handle)
    # 设置倒立摆演员的自由度属性:可控制小车左右移动,不可控制杆旋转
    dof_props['driveMode'] = (gymapi.DOF_MODE_EFFORT,gymapi.DOF_MODE_NONE)
    # 刚度和阻尼设置为0
    dof_props['stiffness'] = (0.0,0.0)
    dof_props['damping'] = (0.0,0.0)

    cart_dof_handle = gym.find_actor_dof_handle(env, cartpole_handle, 'slider_to_cart')
    cartpole_dof_handles.append(cart_dof_handle)
    # 将倒立摆演员的自由度属性设置为新的属性
    gym.set_actor_dof_properties(env, cartpole_handle, dof_props)
 

# 创建初始状态的拷贝，用于重置环境
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
 



 
'''-------------------------------创建仿真环境的地形----------------------------------'''
# 创建所有可用的地形类型
num_terains = 6
terrain_width = 5.
terrain_length = 5.
horizontal_scale = 0.1  # [m]
vertical_scale = 0.005  # [m]
num_rows = int(terrain_width/horizontal_scale)
num_cols = int(terrain_length/horizontal_scale)
heightfield = np.zeros((num_terains * num_rows, num_cols), dtype=np.int16)

def new_sub_terrain():
    return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

# 平地
heightfield[ :1 * num_rows , :] = np.zeros((num_rows, num_cols), dtype=np.int16)
 
# 上斜坡
heightfield[1 * num_rows:2 * num_rows ,:] = sloped_terrain(new_sub_terrain(), slope=0.5).height_field_raw
 
# 下斜坡
heightfield[2 * num_rows:3 * num_rows,:] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw+490
# 将上下斜坡区域超过 340 的值改为 340
heightfield[1 * num_rows:3 * num_rows,:] = np.clip(heightfield[1 * num_rows:3 * num_rows,:], None, 340)

# 离散地形
heightfield[3 * num_rows:4 * num_rows ,:] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.25, min_size=0.25, max_size=1., num_rects=20).height_field_raw
 
# 上台阶
heightfield[4 * num_rows:5 * num_rows,:] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.17).height_field_raw
# 将上台阶区域超过 340 的值改为 340
heightfield[4 * num_rows:5 * num_rows,:] = np.clip(heightfield[4 * num_rows:5 * num_rows,:], None, 340)

# 下台阶
heightfield[5 * num_rows:6 * num_rows,:] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=-0.17).height_field_raw+340
# 将下台阶区域小于0 的值改为0
heightfield[5 * num_rows:6 * num_rows,:] = np.clip(heightfield[5 * num_rows:6 * num_rows,:], 0, None)
 
 
# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)



'''--------------------------------创建Viewer查看器----------------------------------'''
# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 设置查看器初始位置和目标位置
cam_pos = gymapi.Vec3(-2.124770, -6.526505, 10.728952)
cam_target = gymapi.Vec3(6, 2.5, 2)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)




'''--------------------------------订阅按键功能----------------------------------'''
# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")




'''--------------------------------初始化状态----------------------------------'''
# 获取自由度状态张量
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
# 将自由度状态张量转换为 torch 张量
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
# 提取自由度位置信息
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
# 提取自由度速度信息
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
# 生成环境id
env_ids = np.arange(num_envs)

#初始化GPU上环境变量
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

# 初始化积分误差
integral_error = torch.zeros(num_envs, device=device)

'''--------------------------------开始仿真----------------------------------'''
while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
 
            # # 将 cart_pos、cart_vel、pole_angle 和 pole_vel 对应的自由度位置和速度置为 0
            # # 创建一个全零张量，用于存储每个环境中每个自由度的驱动力
            # zero_tensor = torch.zeros((num_envs * num_dof,2), device=device, dtype=torch.float)
            # # 将 PyTorch 张量转换为 Isaac Gym 可以处理的格式并更新仿真状态
            # gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(zero_tensor))
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
            # 初始化积分误差
            integral_error = torch.zeros(num_envs, device=device)       



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
    pole_angle = dof_pos[env_ids, 1].squeeze()
    pole_vel = dof_vel[env_ids, 1].squeeze()

    # PID 控制器参数
    Kp = 60.0  # 比例系数
    Ki = 0    # 积分系数
    Kd = 0.0   # 微分系数

    # 期望的摆杆角度（直立状态）
    target_angle = 0.0
 
    # 计算误差
    error = target_angle - pole_angle
    integral_error += error
    derivative = -pole_vel  # 误差的微分可以近似为摆杆角速度的负值

    # 计算 PID 控制量
    pid_output = Kp * error + Ki * integral_error + Kd * derivative

    # 创建一个全零张量，用于存储每个环境中每个自由度的驱动力
    actions_tensor = torch.zeros(num_envs * num_dof, device=device, dtype=torch.float)
    # 将 PID 控制量赋值给对应的自由度
    actions_tensor[::num_dof] = pid_output
 
    # 将 PyTorch 张量转换为 Isaac Gym 可以处理的格式
    forces = gymtorch.unwrap_tensor(actions_tensor)
    # 设置仿真环境中自由度的驱动力张量
    gym.set_dof_actuation_force_tensor(sim, forces)


    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

'''--------------------------------仿真结束清理----------------------------------'''
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)