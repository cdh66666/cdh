from isaacgym import gymutil, gymapi, gymtorch
# 导入Isaac Gym环境中的一些函数和类，用于创建地形
from isaacgym.terrain_utils import *
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

    return gym, sim, args, device




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
