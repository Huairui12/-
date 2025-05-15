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

# 洛伦兹系统参数
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# 洛伦兹系统的微分方程
def lorenz_system(t, xyz):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# 时间范围和初始条件
t_span = (0, 40)
# 减少帧数，从4000减少到2000，加快生成速度
t_eval = np.linspace(*t_span, 2000)
initial_state = [0.1, 0.1, 0.1]

# 求解微分方程
solution = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
x, y, z = solution.y

# 创建图形
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# 设置图形属性
ax.set_xlabel('X轴', fontsize=12, labelpad=10)
ax.set_ylabel('Y轴', fontsize=12, labelpad=10)
ax.set_zlabel('Z轴', fontsize=12, labelpad=10)
ax.set_title('洛伦兹吸引子', fontsize=16, pad=20)

# 设置视角
ax.view_init(elev=30, azim=45)

# 创建文本区域显示微分方程
text_ax = fig.add_subplot(122)
text_ax.axis('off')
text_ax.set_facecolor('black')

# 添加微分方程文本
equations = (
    r"洛伦兹吸引子微分方程系统:" + "\n\n" +
    r"$\frac{dx}{dt} = \sigma(y - x)$" + "\n\n" +
    r"$\frac{dy}{dt} = x(\rho - z) - y$" + "\n\n" +
    r"$\frac{dz}{dt} = xy - \beta z$" + "\n\n" +
    r"参数: $\sigma = 10$, $\rho = 28$, $\beta = \frac{8}{3}$"
)

text_ax.text(0.1, 0.5, equations, fontsize=16, color='white', 
             verticalalignment='center', horizontalalignment='left')

# 初始化空轨迹
line, = ax.plot([], [], [], lw=0.8, color='cyan')
point, = ax.plot([], [], [], 'o', color='yellow', markersize=6)

# 设置轴范围
ax.set_xlim(min(x)-5, max(x)+5)
ax.set_ylim(min(y)-5, max(y)+5)
ax.set_zlim(min(z)-5, max(z)+5)

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
ani.save('lorenz_attractor.gif', writer='pillow', fps=480, dpi=60,  # fps从240增加到480
         progress_callback=progress_callback)
end_time = time.time()
print(f"总共用时: {end_time - start_time:.2f} 秒")

# 显示图像
plt.tight_layout()
plt.show()