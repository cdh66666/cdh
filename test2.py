#核心 API（包括支持的数据类型和常量）在模块中定义gymapi
from isaacgym import gymapi
 
from math import pi
import random


# 所有 Gym API 函数都可以作为Gym启动时获取的单例对象的方法访问。
# 例如，要获取单例对象，可以使用gymapi.acquire_gym()函数。
# 然后，可以使用gym.create_sim()函数创建模拟环境。
# 要释放单例对象，可以使用gym.release()函数。'''
gym = gymapi.acquire_gym()

#SimParams类用于设置模拟参数，包括重力、时间步长、子步数等。
# 获得默认的模拟参数
sim_params = gymapi.SimParams()

 
# 设置模拟参数
sim_params.dt = 1 / 60
sim_params.substeps = 2
# 设定重力方向为Z轴向上
sim_params.up_axis = gymapi.UP_AXIS_Z
# 设定重力大小为-9.8m/s^2
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# 设置physx参数
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
 
# 第一个参数是计算设备序号，用于选择用于物理模拟的 GPU。
# 第二个参数是图形设备序号，用于选择用于渲染的 GPU。无需渲染时，设置为-1。
# 第三个参数是物理引擎类型， SIM_PHYSX专门做刚体模拟，SIM_FLEX专门做布料模拟。
# PhysX 后端提供强大的刚体和关节模拟，可在 CPU 或 GPU 上运行。它是目前唯一完全支持新张量 API 的后端。
# Flex 后端提供完全在 GPU 上运行的软体和刚体模拟，但它尚未完全支持张量 API。
# 第四个参数是模拟参数，包括重力、时间步长、子步数等。
sim = gym.create_sim(0,0,gymapi.SIM_PHYSX,sim_params)


# 创建一个平面参数对象，用于设置平面的属性。
plane_params = gymapi.PlaneParams()
#平面normal参数定义平面方向，取决于上轴的选择。z 向上使用 (0, 0, 1)，y 向上使用 (0, 1, 0)。
#可以指定非轴对齐的法向量以获得倾斜的地面。
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
# 该distance参数定义平面与原点的距离。
plane_params.distance = 0
# static_friction和dynamic_friction是静摩擦系数和动摩擦系数。
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
# 该restitution系数可用于控制与地面碰撞的弹性（反弹量）。
plane_params.restitution = 0

# 创建一个平面，并将其添加到模拟环境中。
gym.add_ground(sim, plane_params)



# Gym 目前支持加载 URDF 和 MJCF 文件格式。
# 加载资产文件会创建一个GymAsset对象，其中包含所有物体、碰撞形状、视觉附件、关节和自由度 (DOF) 的定义。
# 某些格式还支持软体和粒子。
asset_root = "assets"
asset_file = "urdf/cartpole.urdf"

# 该load_asset方法使用文件扩展名来确定资产文件格式。
# 支持的扩展名包括URDF 文件的.urdf和MJCF 文件的.xml 。
asset = gym.load_asset(sim, asset_root, asset_file)

# 每个环境都有自己的坐标空间，嵌入到全局模拟空间中。
# 创建环境时，我们指定环境的局部范围，这取决于环境实例之间的所需间距。
spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, 0 )
upper = gymapi.Vec3(spacing, spacing, spacing)
# 最后一个参数表示create_env每行要打包多少个环境。
# 随着新环境添加到模拟中，它们将一次一行地排列在 2D 网格中。
env = gym.create_env(sim, lower, upper, 8)

#每个演员都必须放置在一个环境中。您不能拥有不属于环境的演员。
# 演员姿势在环境局部坐标中使用位置向量p和方向四元数定义r。
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
#在下面的代码片段中，方向由四元数 (-0.707107, 0.0, 0.0, 0.707107) 指定。
# 构造函数按 ( , , , ) 顺序Quat接受参数，因此这个四元数表示绕 x 轴 -90 度旋转。
# 当将使用 z-up 约定定义的资产加载到使用 y-up 约定的模拟中时，这样的旋转是必要的。
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
#saac Gym 提供了一组方便的数学助手，包括四元数实用程序，因此四元数可以像这样以轴角形式定义：xyzw
pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * pi)


# 姿势后的参数create_actor是可选的。
# 您可以为演员指定一个名字，在本例中为“MyActor”。这样以后就可以按名字查找演员了。
# 如果您希望这样做，请确保为同一环境中的所有演员分配唯一的名称。

# 参数列表末尾的两个整数是collision_group和collision_filter。
# collision_group是一个整数，用于标识将为 Actor 的刚体分配碰撞组。
# 两个刚体只有属于同一个碰撞组时才会相互碰撞。
# 通常每个环境都有一个碰撞组，在这种情况下，组 ID 对应于环境索引。
# 这可以防止不同环境中的 Actor 相互进行物理交互。
# 在某些情况下，您可能希望为每个环境设置多个碰撞组，以实现更细粒度的控制。
# 值 -1 用于与所有其他组发生碰撞的特殊碰撞组。
# 这可用于创建可以与所有环境中的 Actor 进行物理交互的“共享”对象。

# collision_filter是一个位掩码，可用于过滤掉物体之间的碰撞。
# 如果两个物体的碰撞过滤器设置了共同的位，则它们不会发生碰撞。
# 此值可用于过滤掉多物体参与者中的自碰撞，或防止场景中某些类型的物体发生物理交互。
actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)


# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = random.uniform(1.0, 2.5)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.0, 0.0)

    actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 01)
    actor_handles.append(actor_handle)

# 这会弹出一个带有默认尺寸的窗口。
# 您可以通过自定义来设置不同的尺寸isaacgym.gymapi.CameraProperties。
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)  

# 如果您希望在查看器窗口关闭时终止模拟，则可以对query_viewer_has_closed方法进行循环调节，
# 该方法将在用户关闭窗口后返回 True。
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # 该step_graphics方法将模拟的视觉表示与物理状态同步。
    gym.step_graphics(sim)
    # 该draw_viewer方法在查看器中渲染最新快照。
    # 它们是独立的方法，因为step_graphics即使在没有查看器的情况下也可以使用，例如在渲染相机传感器时。
    gym.draw_viewer(viewer, sim, True)

    # 使用此代码，查看器窗口将尽快刷新。
    # 对于简单的模拟，这通常会比实时更快，因为每个dt增量的处理速度都比实时时间流逝的速度快。
    # 要将视觉更新频率与实时同步，您可以在循环迭代末尾添加此语句：
    # 这会将模拟速率降低到实时。如果模拟运行速度比实时慢，则此语句将不起作用。
    gym.sync_frame_time(sim)


# 查看器 GUI
# 创建查看器后，屏幕左侧将显示一个简单的图形用户界面。可以使用“Tab”键打开或关闭 GUI 显示。

# GUI 有 4 个独立的选项卡：Actors、Sim、Viewer和Perf。

# Actors选项卡提供了选择环境和该环境中的 Actor 的功能。当前选定的 Actor 有三个单独的子选项卡。

# 身体子选项卡提供有关活动角色刚体的信息。它还允许更改角色身体的显示颜色并切换身体轴的可视化。

# DOF子选项卡显示有关活动参与者自由度的信息。DOF 属性可使用用户界面进行编辑，但请注意，这是一项实验性功能。

# 姿势覆盖子选项卡可用于使用演员的自由度手动设置演员的姿势。启用此功能后，将使用滑块在用户界面中设置的值覆盖所选演员的姿势和驱动目标。它可以成为一种有用的工具，用于以交互方式探索或操纵演员的自由度。

# Sim选项卡显示物理模拟参数。参数因模拟类型（PhysX 或 Flex）而异，用户可以修改。

# 查看器选项卡允许自定义常见的可视化选项。值得注意的功能是能够在查看物体的图形表示和物理引擎使用的物理形状之间切换。这在调试物理行为时非常有用。

# Perf选项卡显示健身房内部测量的性能。顶部滑块“性能测量窗口”指定测量性能的帧数。帧速率报告上一个测量窗口的平均每秒帧数 (FPS)。其余性能指标报告为指定帧数的每帧平均值。

# 帧时间是从一个步骤开始到下一个步骤开始的总时间

# 物理模拟时间是物理求解器运行的时间。

# 物理数据复制是复制模拟结果所花费的时间。

# 空闲时间是处于空闲状态的时间，通常在之内gym.sync_frame_time(sim)。

# 查看器渲染时间是渲染和显示查看器所花费的时间

# 传感器图像复制时间是将传感器图像数据从 GPU 复制到 CPU 所花费的时间。

# 传感器图像渲染时间是将相机传感器（不包括查看器相机）渲染到 GPU 缓冲区所花费的时间。


# 要从查看器获取鼠标/键盘输入，可以订阅和查询动作事件。
# 有关如何执行此操作的示例，请查看examples/projectiles.py：
# gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "space_shoot")
# gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
# gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")
# ...
# while not gym.query_viewer_has_closed(viewer):
#     ...
#     for evt in gym.query_viewer_action_events(viewer):
#         ...

# 清理
# 退出时，模拟和查看器对象应按如下方式释放：

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)