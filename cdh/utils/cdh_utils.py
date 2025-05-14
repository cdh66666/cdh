from isaacgym import gymutil, gymapi, gymtorch
# 导入Isaac Gym环境中的一些函数和类，用于创建地形
from isaacgym.terrain_utils import *
import numpy as np
import torch
 

def draw_sphere(gym,viewer,env, pos=[0, 0, 0]):
    # 创建一个线框球体的几何参数对象
    # 球体半径为 0.1，纬度和经度的分段数均为 3，颜色设置为黄色 (1, 1, 0)
    sphere_params = gymutil.WireframeSphereGeometry(0.1, 3, 3, color=(1, 1, 0))
    # 创建一个变换对象，用于指定球体的位置
    # 从传入的 pos 列表中提取 x、y、z 坐标，旋转部分设置为 None
    pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
    # 使用 gymutil 模块的 draw_lines 函数，根据球体几何参数、gym 对象、查看器、环境和变换对象绘制球体
    gymutil.draw_lines(sphere_params, gym, viewer, env, pose)



def initialize_environment(custom_parameters=None, description=None):
    """
    初始化Isaac Gym环境，按照指定参数配置仿真。

    参数:
        custom_parameters (list, 可选): 自定义命令行参数列表，默认为预定义的参数。
        description (str, 可选): 解析命令行参数时的描述信息，默认为预定义的描述。

    返回:
        tuple: 包含 gym 实例、sim 实例、args 对象和 device 对象的元组。
    """
    # 初始化默认值
    if custom_parameters is None:
        custom_parameters = []
    if description is None:
        description = ""

    # 初始化gym
    gym = gymapi.acquire_gym()

    # 指定使用GPU进行计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("GPU不可用，将使用CPU运行。")

    # 调用 parse_arguments 函数
    args = gymutil.parse_arguments(description=description, custom_parameters=custom_parameters)
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

    # 创建仿真环境的地形    
    create_simulation_terrain(gym, sim, device)

    # 订阅按键功能
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    
    return gym, sim, args, device,viewer




def create_simulation_terrain(gym, sim, device):
    """
    创建仿真环境的地形。

    参数:
        gym: Isaac Gym 实例。
        sim: Isaac Gym 仿真实例。
        device: 计算设备，如 'cpu' 或 'cuda'。

    返回:
        无
    """
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


def reset_environment_states(gym, sim, num_envs, num_dof, device, pos_range=0.2, vel_range=0.5):
    """
    随机重置环境中每个自由度的位置和速度状态。

    参数:
        gym: Isaac Gym 实例。
        sim: Isaac Gym 仿真实例。
        num_envs: 环境的数量。
        num_dof: 每个环境中自由度的数量。
        device: 计算设备，如 'cpu' 或 'cuda'。
        pos_range (float, 可选): 位置的随机范围，默认为 0.2。
        vel_range (float, 可选): 速度的随机范围，默认为 0.5。
    """
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
    return random_state_tensor.view(-1, 4)




class IsaacCartPoleEnv:
    def __init__(self):
        # 传入自定义参数
        self.custom_parameters = [
            {"name": "--num_envs", "type": int, "default": 512, "help": "环境数量"},
            {"name": "--episode_length", "type": int, "default": 300, "help": "每个回合的长度"},
            {"name": "--total_train_episodes", "type": int, "default": 100, "help": "总共训练轮数"},
            {"name": "--batch_size", "type": int, "default": 2, "help": "采样批次大小"},
            {"name": "--buffer_capacity", "type": int, "default": 10, "help": "经验回放池容量"}
        ]
        self.desc = "5. 完善DRL代码，优化算法，快速训练，实现倒立摆直立。"

        self.gym, self.sim, self.args, self.device = initialize_environment(custom_parameters=self.custom_parameters, description=self.desc)

        # 加载倒立摆资产
        self.asset_root = "resources/cartpole"
        self.asset_file = "urdf/cartpole.urdf"
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.fix_base_link = True
        self.cartpole_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, self.asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.cartpole_asset)
        print(f"\n自由度数量: {self.num_dof}")

        # 设置环境参数
        self.num_envs = self.args.num_envs
        self.episode_length = self.args.episode_length
        self.total_train_episodes = self.args.total_train_episodes
        self.batch_size = self.args.batch_size
        self.buffer_capacity = self.args.buffer_capacity

        self.num_per_row = 30
        self.env_spacing = 0.5
        self.env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
        self.pose = gymapi.Transform()
        self.pose.p = gymapi.Vec3(0.0, 0.0, 4.0)
        self.pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        # 创建倒立摆环境
        np.random.seed(0)
        torch.manual_seed(0)
        self.cartpole_handles = []
        self.cartpole_dof_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)
            cartpole_handle = self.gym.create_actor(env, self.cartpole_asset, self.pose, "cartpole", i + 1, 1, 0)
            self.cartpole_handles.append(cartpole_handle)
            dof_props = self.gym.get_actor_dof_properties(env, cartpole_handle)
            dof_props['driveMode'] = (gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_POS)
            dof_props['stiffness'] = (0.0, 10.0)
            dof_props['damping'] = (0.0, 1.0)
            cart_dof_handle = self.gym.find_actor_dof_handle(env, cartpole_handle, 'slider_to_cart')
            self.cartpole_dof_handles.append(cart_dof_handle)
            self.gym.set_actor_dof_properties(env, cartpole_handle, dof_props)

        # 创建仿真环境的地形
        create_simulation_terrain(self.gym, self.sim, self.device)

        # 创建Viewer查看器
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        # 设置查看器初始位置和目标位置
        cam_pos = gymapi.Vec3(-2.124770, -6.526505, 10.728952)
        cam_target = gymapi.Vec3(6, 2.5, 2)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        for env in self.envs:
            draw_sphere(self.gym, self.viewer, env, pos=[0, 0, 4])

        # 订阅按键功能
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        # 初始化状态
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.env_ids = torch.arange(self.num_envs, device=self.device)

        self.gym.prepare_sim(self.sim)
        reset_environment_states(self.gym, self.sim, self.num_envs, self.num_dof, self.device, pos_range=0, vel_range=0.1)

    def get_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        cart_pos = self.dof_pos[self.env_ids, 0]
        cart_vel = self.dof_vel[self.env_ids, 0]
        pole_pos = self.dof_pos[self.env_ids, 1]
        pole_vel = self.dof_vel[self.env_ids, 1]
        state = torch.stack([cart_pos, cart_vel, pole_pos, pole_vel], dim=1)
        return state

    def step(self, action):
        dof_actuation_force_tensor = torch.zeros((self.num_envs * self.num_dof), device=self.device, dtype=torch.float)
        dof_actuation_force_tensor[::self.num_dof] = action.squeeze()
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_actuation_force_tensor))

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        next_state = self.get_state()
        return next_state

    def reset(self):
        reset_environment_states(self.gym, self.sim, self.num_envs, self.num_dof, self.device, pos_range=1, vel_range=0.2)

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def handle_events(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "reset" and evt.value > 0:
                reset_environment_states(self.gym, self.sim, self.num_envs, self.num_dof, self.device, pos_range=0, vel_range=0.1)
                print("重置环境状态")

    def render(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def get_env_info(self):
        return {
            "num_envs": self.num_envs,
            "episode_length": self.episode_length,
            "total_train_episodes": self.total_train_episodes,
            "batch_size": self.batch_size,
            "buffer_capacity": self.buffer_capacity,
            "device": self.device
        }
    