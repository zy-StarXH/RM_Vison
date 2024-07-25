import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
g = 9.81
v0 = 10  # 初始速度
angle = np.deg2rad(30)  # 初始速度与水平面之间的夹角，转换为弧度
spin_factor = 0.2  # 旋转效应因子
initial_height = 0.5
table_height = 0.76  # 桌面高度
dz = 0.01  # 乒乓球与桌面接触的微小厚度
t_final = 2  # 模拟时间长度
n_steps = 1000  # 时间步数
dt = t_final / (n_steps - 1)  # 时间步长

# 计算初始速度在x轴和z轴的分量
vx0 = v0 * np.cos(angle)
vz0 = v0 * np.sin(angle)

# 初始化位置数组
x = np.zeros(n_steps)
z = np.zeros(n_steps)

# 设置初始位置
x[0] = 0
z[0] = initial_height

# 初始化速度
vx = vx0
vz = vz0

# 绘制图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制桌面
ax.plot([-10, 10], [0, 0], [table_height, table_height], color='k', lw=2)

# 迭代计算位置和速度
for t in range(n_steps - 1):
    # 更新x坐标
    x[t + 1] = x[t] + vx * dt
    # 更新z方向的速度（考虑重力）
    vz_new = vz - g * dt
    # 碰撞检测
    if z[t] - dz / 2 <= table_height and vz_new < 0:  # 假设dz是球与桌面表面的接触厚度
        # 更新碰撞后的z方向速度
        vz_new = -vz * (1 + spin_factor)
        # 设置碰撞后的z坐标
        z[t + 1] = table_height + dz / 2
    else:
        # 更新非碰撞时的z坐标
        z[t + 1] = z[t] + vz_new * dt
        # 保持x方向速度不变
    vx = vx
    # 更新z方向速度
    vz = vz_new

# 绘制轨迹
ax.plot(x, np.zeros_like(x), z, color='b', lw=2)

# 绘制初始位置和结束位置
ax.scatter(x[0], 0, z[0], color='b', s=100)  # 初始位置
ax.scatter(x[-1], 0, z[-1], color='g', s=100)  # 结束位置

# 设置坐标轴范围和视角
ax.set_xlim(min(x) - 2, max(x) + 2)
ax.set_ylim(0, 2)
ax.set_zlim(min(z) - 0.5, max(z) + 0.5)
ax.view_init(elev=10., azim=30)  # 设置视角
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'拍面角度：{np.rad2deg(angle)}°')  # 修改标题以显示角度

plt.show()
