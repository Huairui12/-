import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
import time

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置更好的图形风格
plt.style.use('dark_background')
mpl.rcParams['figure.figsize'] = [12, 8]

# 陈氏系统参数
a = 40
b = 3
c = 28

# 陈氏系统的微分方程
def chen_system(t, xyz):
    x, y, z = xyz
    dx_dt = a * (y - x)
    dy_dt = (c - a) * x - x * z + c * y
    dz_dt = x * y - b * z
    return [dx_dt, dy_dt, dz_dt]

# 时间范围和初始条件
t_span = (0, 20)
t_eval = np.linspace(*t_span, 2000)  # 使用2000帧以加快生成速度
initial_state = [1.0, 1.0, 1.0]

# 求解微分方程
print("正在求解陈氏吸引子微分方程...")
solution = solve_ivp(chen_system, t_span, initial_state, t_eval=t_eval)
x, y, z = solution.y
print("微分方程求解完成！")

# 创建图形
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# 设置图形属性
ax.set_xlabel('X轴', fontsize=12, labelpad=10)
ax.set_ylabel('Y轴', fontsize=12, labelpad=10)
ax.set_zlabel('Z轴', fontsize=12, labelpad=10)
ax.set_title('陈氏吸引子', fontsize=16, pad=20)

# 设置视角
ax.view_init(elev=30, azim=45)

# 创建文本区域显示微分方程
text_ax = fig.add_subplot(122)
text_ax.axis('off')
text_ax.set_facecolor('black')

# 添加微分方程文本
equations = (
    r"陈氏吸引子微分方程系统:" + "\n\n" +
    r"$\frac{dx}{dt} = a(y - x)$" + "\n\n" +
    r"$\frac{dy}{dt} = (c-a)x - xz + cy$" + "\n\n" +
    r"$\frac{dz}{dt} = xy - bz$" + "\n\n" +
    r"参数: $a = 40$, $b = 3$, $c = 28$"
)

text_ax.text(0.1, 0.5, equations, fontsize=16, color='white', 
             verticalalignment='center', horizontalalignment='left')

# 初始化空轨迹
line, = ax.plot([], [], [], lw=0.8, color='#00ffaa')  # 青绿色
point, = ax.plot([], [], [], 'o', color='yellow', markersize=6)

# 设置轴范围
ax.set_xlim(min(x)-1, max(x)+1)
ax.set_ylim(min(y)-1, max(y)+1)
ax.set_zlim(min(z)-1, max(z)+1)

# 初始化函数
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

# 动画更新函数
def update(frame):
    # 绘制轨迹
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    
    # 绘制当前点
    point.set_data([x[frame]], [y[frame]])
    point.set_3d_properties([z[frame]])
    
    # 调整视角，使图像旋转
    if frame % 50 == 0:  # 减少视角更新频率
        ax.view_init(elev=30, azim=frame/20)
    
    return line, point

# 创建动画
ani = FuncAnimation(fig, update, frames=len(t_eval), 
                    init_func=init, blit=True, interval=1, repeat=True)  # 间隔从2减少到1

# 保存为GIF
print("开始生成GIF动画...")
start_time = time.time()
# 定义进度回调函数
def progress_callback(current_frame, total_frames):
    progress = (current_frame + 1) / total_frames * 100
    print(f"\r生成GIF进度: {progress:.1f}%", end="")
    if current_frame == total_frames - 1:
        print("\nGIF动画已保存完成")
ani.save('chen_attractor.gif', writer='pillow', fps=480, dpi=60,  # fps从240增加到480
         progress_callback=progress_callback)
end_time = time.time()
print(f"总共用时: {end_time - start_time:.2f} 秒")

# 显示图像
plt.tight_layout()
plt.show()