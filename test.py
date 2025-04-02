 # 导入NumPy库，用于处理数值计算
import numpy as np
# 从isaacgym库中导入gymutil模块，用于处理命令行参数解析等实用功能
from isaacgym import gymutil
# 从isaacgym库中导入gymapi模块，用于与Isaac Gym的API进行交互
from isaacgym import gymapi
# 从math库中导入sqrt函数，用于计算平方根
from math import sqrt
#

# 初始化Isaac Gym，获取gym对象
gym = gymapi.acquire_gym()


# 解析命令行参数
args = gymutil.parse_arguments(
    # 描述脚本功能，即演示环境内和环境间的碰撞过滤
    description="测试cdh创建的环境",
    # 自定义命令行参数
    custom_parameters=[
        # 环境数量，默认值为1024
        {"name": "--num_envs", "type": int, "default": 36, "help": "创建环境的数量"},
    ]
)


# 配置模拟参数
sim_params = gymapi.SimParams()
# 设置时间步长，单位：s，较小的时间步长可以提高模拟的稳定性。
sim_params.dt = 0.01
# 设置子步数，增加子步数可以提高模拟的精度。
sim_params.substeps = 2
# 设置PhysX求解器类型
sim_params.physx.solver_type = 1
# 设置PhysX位置迭代次数，增加位置迭代次数可以提高物体位置的准确性。
sim_params.physx.num_position_iterations = 8
# 设置PhysX速度迭代次数
sim_params.physx.num_velocity_iterations = 2
# 使用 4 个线程进行模拟，可以充分利用多核 CPU 的性能
sim_params.physx.num_threads = 4
# 使用 GPU 进行模拟可以加速计算。
sim_params.physx.use_gpu = True
# 设置接触偏移和静止偏移，影响物体之间的接触和静止状态。
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
# 设置重力
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# 创建模拟环境
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
# 如果模拟环境创建失败
if sim is None:
    print("*** 创建环境失败，请检查参数是否正确 ***")
    quit()


# 定义地面平面参数
plane_params = gymapi.PlaneParams()
# 修改地面平面的法向量，例如让 z 轴朝上
plane_params.normal = gymapi.Vec3(0, 0, 1)
# 向模拟环境中添加地面平面
gym.add_ground(sim, plane_params)
 

# 创建查看器
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# 如果查看器创建失败
if viewer is None:
    print("*** 创建查看器失败，请检查参数是否正确 ***")
    quit()


 
# 定义资产根目录
asset_root = "assets" 
# 定义资产文件路径
asset_file = "urdf/ball.urdf"
# 加载球的资产
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())


# 获取环境数量
num_envs = args.num_envs
# 计算每行的环境数量
num_per_row = int(sqrt(num_envs))
# 定义环境间距
env_spacing = 1.25
# 定义环境的下限坐标
env_lower = gymapi.Vec3(-env_spacing,-env_spacing, 0.0)
# 定义环境的上限坐标
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# 存储所有环境的列表
envs = []


# 订阅查看器的键盘事件，当按下 'R' 键时触发 "reset" 动作
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
# 订阅空格键事件，当按下空格键时触发 "print_camera_info" 动作
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "print_camera_info")

 
# 设置随机数种子，确保结果可复现
np.random.seed(17)

# 循环创建环境
for i in range(num_envs):

    # 创建一个新的环境
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    # 将新环境添加到环境列表中
    envs.append(env)


    # 生成随机明亮颜色
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])


    # 创建球金字塔
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    n = 4
    radius = 0.2
    ball_spacing = 2.5 * radius
    min_coord = -0.5 * (n - 1) * ball_spacing

    z = min_coord + 4
    while n > 0:
        y = min_coord
        for j in range(n):
            x = min_coord
            for k in range(n):
                pose.p = gymapi.Vec3(x, y, 1.5 + z)
                # 所有物体都应发生碰撞
                # 将所有物体放在同一组，过滤掩码设置为0（无过滤）
                collision_group = 0
                collision_filter = 0

                # 创建一个物体并获取其句柄
                ahandle = gym.create_actor(env, asset, pose, None, collision_group, collision_filter)
                # 设置物体的刚体颜色
                gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

                x += ball_spacing
            y += ball_spacing
        z += ball_spacing
        n -= 1
        min_coord = -0.5 * (n - 1) * ball_spacing


# 设置查看器相机的视角，让相机在 z 轴正方向上，看向原点
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(5, -10, 15), gymapi.Vec3(5, 0, 5))
 
 


# 创建初始状态的本地副本，用于重置模拟
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


# 主循环，直到查看器关闭
while not gym.query_viewer_has_closed(viewer):
    # 获取查看器的输入动作并进行相应处理
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            # 如果按下 'R' 键，重置模拟状态
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        if evt.action == "print_camera_info" and evt.value > 0:
            # 如果按下 'print_camera_info' 键，打印相机信息
            print("print_camera_info")
 

    # 进行物理模拟步骤
    gym.simulate(sim)
    # 获取模拟结果
    gym.fetch_results(sim, True)


    # 更新查看器的图形显示
    gym.step_graphics(sim)
    # 绘制查看器的画面
    gym.draw_viewer(viewer, sim, True)



    # 等待一段时间，使物理模拟与渲染帧率同步
    gym.sync_frame_time(sim)

# 销毁查看器
gym.destroy_viewer(viewer)
# 销毁模拟环境
gym.destroy_sim(sim)
 