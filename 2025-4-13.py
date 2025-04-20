"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
- Get cartpole state information
- Visualize desired positions and trajectories
"""

import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="2. 创建倒立摆环境，可控制力矩，获取状态信息，期望位置&轨迹可视化")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")
 
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)
 
# add cartpole urdf asset
asset_root = "assets"
asset_file = "urdf/cartpole.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)



"""
定义一个函数，用于在指定环境中绘制一个黄色球体。

:param env: 要在其中绘制球体的环境对象，由 isaacgym 提供。
:param pos: 球体的位置，默认为原点 [0, 0, 0]，是一个包含三个元素的列表，分别表示 x、y、z 坐标。
"""
def draw_sphere(env, pos=[0, 0, 0]):
    # 创建一个线框球体的几何参数对象
    # 球体半径为 0.1，纬度和经度的分段数均为 3，颜色设置为黄色 (1, 1, 0)
    sphere_params = gymutil.WireframeSphereGeometry(0.1, 3, 3, color=(1, 1, 0))
    # 创建一个变换对象，用于指定球体的位置
    # 从传入的 pos 列表中提取 x、y、z 坐标，旋转部分设置为 None
    pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
    # 使用 gymutil 模块的 draw_lines 函数，根据球体几何参数、gym 对象、查看器、环境和变换对象绘制球体
    gymutil.draw_lines(sphere_params, gym, viewer, env, pose)






# Create environment 4
# Cart controlled by applying force.
# Pole moves freely under gravity.
env4 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole4 = gym.create_actor(env4, cartpole_asset, initial_pose, 'cartpole', 4, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env4, cartpole4)
props["driveMode"] = (gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_NONE)  # 小车设为施加力模式，杆不设驱动模式
props["stiffness"] = (0.0, 0.0)
props["damping"] = (0.0, 0.0)
gym.set_actor_dof_properties(env4, cartpole4, props)
# Find DOF handles
cart_dof_handle4 = gym.find_actor_dof_handle(env4, cartpole4, 'slider_to_cart')
pole_dof_handle4 = gym.find_actor_dof_handle(env4, cartpole4, 'cart_to_pole')
gym.apply_dof_effort(env4, cart_dof_handle4, 200)
# Define desired trajectory
desired_cart_position = 0.0
desired_pole_angle = 0.0

# Trajectory visualization
trajectory_points = []

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

draw_sphere(env4,pos=[0,2,0])

 
# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)

    # Get state information
    cart_pos = gym.get_dof_position(env4, cart_dof_handle4)
    cart_vel = gym.get_dof_velocity(env4, cart_dof_handle4)
    pole_angle = gym.get_dof_position(env4, pole_dof_handle4)
    pole_vel = gym.get_dof_velocity(env4, pole_dof_handle4)

    # print(f"Cart Position: {cart_pos:.2f}, Cart Velocity: {cart_vel:.2f}, Pole Angle: {pole_angle:.2f}, Pole Velocity: {pole_vel:.2f}")
    # Update env 4: apply an effort to the cart
    pos4 = gym.get_dof_position(env4, cart_dof_handle4)
    gym.apply_dof_effort(env4, cart_dof_handle4, -pos4 * 50)

    # Visualize desired position
    desired_cart_point = gymapi.Vec3(desired_cart_position, 2.0, 0.0)
    desired_pole_point = gymapi.Vec3(desired_cart_position + math.sin(desired_pole_angle), 2.0 + math.cos(desired_pole_angle), 0.0)
    num_desired_lines = 1
    # 提取 Vec3 对象的 x, y, z 分量并构建顶点数组
    start_point = np.array([desired_cart_point.x, desired_cart_point.y, desired_cart_point.z], dtype=np.float32)
    end_point = np.array([desired_pole_point.x, desired_pole_point.y, desired_pole_point.z], dtype=np.float32)
    vertices = np.vstack((start_point, end_point)).flatten()
    # 定义颜色数组
    colors = np.array([[1, 0, 0]], dtype=np.float32)
    gym.add_lines(viewer, env4, num_desired_lines, vertices, colors)

    # Record trajectory
    cart_point = [0.0, 2.0, -cart_pos]
    print(cart_point)
    trajectory_points.append(cart_point)
    if len(trajectory_points) > 100:
        trajectory_points.pop(0)


    draw_sphere(env4,pos=trajectory_points[-1])
    
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
