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

# 引力常数
G = 1.0

# 三体系统的微分方程
def three_body_system(t, state):
    # 状态向量包含三个天体的位置和速度
    # state = [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, x3, y3, z3, vx3, vy3, vz3]
    
    # 提取位置和速度
    r1 = state[0:3]
    v1 = state[3:6]
    r2 = state[6:9]
    v2 = state[9:12]
    r3 = state[12:15]
    v3 = state[15:18]
    
    # 计算天体间距离
    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)
    
    # 计算加速度
    a1 = G * m2 * (r2 - r1) / r12**3 + G * m3 * (r3 - r1) / r13**3
    a2 = G * m1 * (r1 - r2) / r12**3 + G * m3 * (r3 - r2) / r23**3
    a3 = G * m1 * (r1 - r3) / r13**3 + G * m2 * (r2 - r3) / r23**3
    
    # 返回导数
    return np.concatenate([v1, a1, v2, a2, v3, a3])

# 三体质量
m1 = 1.0
m2 = 1.0
m3 = 1.0

# 初始条件 - 使用一个有趣的三体构型（例如，三体共面旋转）
# 初始位置
r1_init = np.array([1.0, 0.0, 0.0])
r2_init = np.array([-0.5, 0.866, 0.0])
r3_init = np.array([-0.5, -0.866, 0.0])

# 初始速度 - 给予一些初始速度使系统产生混沌行为
v1_init = np.array([0.0, 0.1, 0.0])
v2_init = np.array([-0.0866, -0.05, 0.0])
v3_init = np.array([0.0866, -0.05, 0.0])

# 组合初始状态
initial_state = np.concatenate([r1_init, v1_init, r2_init, v2_init, r3_init, v3_init])

# 时间范围和求解参数
t_span = (0, 20)
# 减少帧数，从2000减少到1000，加快生成速度
t_eval = np.linspace(*t_span, 1000)

# 求解微分方程
print("正在求解三体问题微分方程...")
# 降低精度要求，加快计算速度
solution = solve_ivp(three_body_system, t_span, initial_state, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)
print("微分方程求解完成！")

# 提取结果
t = solution.t
states = solution.y
x1, y1, z1 = states[0], states[1], states[2]
x2, y2, z2 = states[6], states[7], states[8]
x3, y3, z3 = states[12], states[13], states[14]

# 创建图形
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# 设置图形属性
ax.set_xlabel('X轴', fontsize=12, labelpad=10)
ax.set_ylabel('Y轴', fontsize=12, labelpad=10)
ax.set_zlabel('Z轴', fontsize=12, labelpad=10)
ax.set_title('三体问题动力学模拟', fontsize=16, pad=20)

# 设置视角
ax.view_init(elev=30, azim=45)

# 创建文本区域显示方程
text_ax = fig.add_subplot(122)
text_ax.axis('off')
text_ax.set_facecolor('black')

# 添加方程文本
equations = (
    r"三体问题牛顿运动方程:" + "\n\n" +
    r"$m_1\frac{d^2\vec{r}_1}{dt^2} = G m_1 m_2 \frac{\vec{r}_2-\vec{r}_1}{|\vec{r}_2-\vec{r}_1|^3} + G m_1 m_3 \frac{\vec{r}_3-\vec{r}_1}{|\vec{r}_3-\vec{r}_1|^3}$" + "\n\n" +
    r"$m_2\frac{d^2\vec{r}_2}{dt^2} = G m_2 m_1 \frac{\vec{r}_1-\vec{r}_2}{|\vec{r}_1-\vec{r}_2|^3} + G m_2 m_3 \frac{\vec{r}_3-\vec{r}_2}{|\vec{r}_3-\vec{r}_2|^3}$" + "\n\n" +
    r"$m_3\frac{d^2\vec{r}_3}{dt^2} = G m_3 m_1 \frac{\vec{r}_1-\vec{r}_3}{|\vec{r}_1-\vec{r}_3|^3} + G m_3 m_2 \frac{\vec{r}_2-\vec{r}_3}{|\vec{r}_2-\vec{r}_3|^3}$" + "\n\n" +
    r"参数: $m_1 = m_2 = m_3 = 1.0$, $G = 1.0$"
)

text_ax.text(0.1, 0.5, equations, fontsize=14, color='white', 
             verticalalignment='center', horizontalalignment='left')

# 计算坐标轴范围
x_min = min(min(x1), min(x2), min(x3)) - 0.5
x_max = max(max(x1), max(x2), max(x3)) + 0.5
y_min = min(min(y1), min(y2), min(y3)) - 0.5
y_max = max(max(y1), max(y2), max(y3)) + 0.5
z_min = min(min(z1), min(z2), min(z3)) - 0.5
z_max = max(max(z1), max(z2), max(z3)) + 0.5

# 设置轴范围
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# 初始化轨迹线和点
line1, = ax.plot([], [], [], lw=0.8, color='#ff5555')  # 红色
line2, = ax.plot([], [], [], lw=0.8, color='#55ff55')  # 绿色
line3, = ax.plot([], [], [], lw=0.8, color='#5555ff')  # 蓝色

point1, = ax.plot([], [], [], 'o', color='#ff5555', markersize=8)
point2, = ax.plot([], [], [], 'o', color='#55ff55', markersize=8)
point3, = ax.plot([], [], [], 'o', color='#5555ff', markersize=8)

# 初始化函数
def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    line3.set_data([], [])
    line3.set_3d_properties([])
    
    point1.set_data([], [])
    point1.set_3d_properties([])
    point2.set_data([], [])
    point2.set_3d_properties([])
    point3.set_data([], [])
    point3.set_3d_properties([])
    
    return line1, line2, line3, point1, point2, point3

# 动画更新函数
def update(frame):
    # 绘制轨迹
    line1.set_data(x1[:frame], y1[:frame])
    line1.set_3d_properties(z1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    line2.set_3d_properties(z2[:frame])
    line3.set_data(x3[:frame], y3[:frame])
    line3.set_3d_properties(z3[:frame])
    
    # 绘制当前点
    point1.set_data([x1[frame]], [y1[frame]])
    point1.set_3d_properties([z1[frame]])
    point2.set_data([x2[frame]], [y2[frame]])
    point2.set_3d_properties([z2[frame]])
    point3.set_data([x3[frame]], [y3[frame]])
    point3.set_3d_properties([z3[frame]])
    
    # 调整视角，使图像旋转
    if frame % 100 == 0:  # 减少视角更新频率
        ax.view_init(elev=30, azim=frame/20)
    
    return line1, line2, line3, point1, point2, point3

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
ani.save('three_body_problem.gif', writer='pillow', fps=360, dpi=60,  # fps从120增加到360
         progress_callback=progress_callback)
end_time = time.time()
print(f"总共用时: {end_time - start_time:.2f} 秒")

# 显示图像
plt.tight_layout()
plt.show()