from isaacgym import gymutil, gymapi 

def draw_sphere(gym,viewer,env, pos=[0, 0, 0]):
    # 创建一个线框球体的几何参数对象
    # 球体半径为 0.1，纬度和经度的分段数均为 3，颜色设置为黄色 (1, 1, 0)
    sphere_params = gymutil.WireframeSphereGeometry(0.1, 3, 3, color=(1, 1, 0))
    # 创建一个变换对象，用于指定球体的位置
    # 从传入的 pos 列表中提取 x、y、z 坐标，旋转部分设置为 None
    pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
    # 使用 gymutil 模块的 draw_lines 函数，根据球体几何参数、gym 对象、查看器、环境和变换对象绘制球体
    gymutil.draw_lines(sphere_params, gym, viewer, env, pose)