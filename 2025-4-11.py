import numpy as np
from isaacgym import gymutil, gymapi
from isaacgym.terrain_utils import *
 
 
# initialize gym
gym = gymapi.acquire_gym()

# 转换自定义参数为 custom_parameters 所需格式
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 90, "help": "环境数量"}
]

# 调用 parse_arguments 函数
args = gymutil.parse_arguments(description="1. 创建复杂地形：平地+上下斜坡+\
                        离散地形+上下台阶+平地",custom_parameters=custom_parameters)
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
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

 
 
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()
 
# load ball asset
asset_root = "assets"
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# set up the env grid
num_envs = args.num_envs
num_per_row = 30
env_spacing = 0.5
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
pose = gymapi.Transform()
pose.r = gymapi.Quat(0, 0, 0, 1)
pose.p.z = 10.
pose.p.x = 0.

np.random.seed(0)
envs = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # ahandle = gym.create_actor(env, asset, pose, None, 0, 0)
    # gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# create terrain parameters
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
 
# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(-2.124770, -6.526505, 10.728952)
cam_target = gymapi.Vec3(6, 2.5, 2)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

#初始化GPU上环境变量
gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)


    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)