import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import time

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置更好的图形风格
plt.style.use('dark_background')
mpl.rcParams['figure.figsize'] = [12, 8]

# 立方映射函数
def cubic_map(x, r):
    """
    立方映射函数
    x_{n+1} = r * x_n * (1 - x_n^2)
    
    这是一种包含三次方项的混沌映射
    """
    return r * x * (1 - x**2)

# 生成立方映射序列
def generate_sequence(x0, r, n):
    """生成n个立方映射迭代点"""
    sequence = np.zeros(n)
    sequence[0] = x0
    for i in range(1, n):
        sequence[i] = cubic_map(sequence[i-1], r)
    return sequence

# 参数设置
r = 3.0  # 立方映射参数
x0 = 0.1  # 初始值
n_iterations = 1000  # 迭代次数
n_frames = 300  # 动画帧数

# 生成完整序列
print("正在生成立方映射序列...")
full_sequence = generate_sequence(x0, r, n_iterations)
print("序列生成完成！")

# 创建图形
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('black')

# 创建左上图形：立方映射函数和轨迹
ax1 = fig.add_subplot(221)
ax1.set_facecolor('black')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel('x_n', fontsize=12)
ax1.set_ylabel('x_{n+1}', fontsize=12)
ax1.set_title('立方映射函数和轨迹', fontsize=16)

# 绘制立方映射函数
x = np.linspace(-1.5, 1.5, 1000)
y = np.array([cubic_map(xi, r) for xi in x])
ax1.plot(x, y, 'w-', lw=1.5, alpha=0.7)

# 绘制对角线 y=x
ax1.plot(x, x, 'w--', lw=0.5, alpha=0.5)

# 初始化轨迹线
map_line, = ax1.plot([], [], 'r-', lw=0.5, alpha=0.7)
map_points, = ax1.plot([], [], 'yo', markersize=4)

# 创建右上图形：公式显示区域
ax_formula = fig.add_subplot(222)
ax_formula.axis('off')
ax_formula.set_facecolor('black')

# 添加立方映射公式
ax_formula.text(0.5, 0.8, "立方映射方程:", fontsize=14, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.6, r"$x_{n+1} = r \cdot x_n \cdot (1 - x_n^2)$", fontsize=14, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.4, r"当 $r > 2.3$ 时，系统开始呈现混沌行为", fontsize=12, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.2, r"参数: $r = " + f"{r}" + r"$", fontsize=14, color='white', 
         horizontalalignment='center')

# 创建左下图形：时间序列
ax2 = fig.add_subplot(223)
ax2.set_facecolor('black')
ax2.set_xlim(0, n_iterations)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('迭代次数 n', fontsize=12)
ax2.set_ylabel('x_n', fontsize=12)
ax2.set_title('立方映射时间序列', fontsize=16)

# 初始化时间序列线
time_line, = ax2.plot([], [], 'c-', lw=0.8)

# 创建右下图形：分岔图
ax3 = fig.add_subplot(224)
ax3.set_facecolor('black')
ax3.set_xlim(2.0, 3.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_xlabel('参数 r', fontsize=12)
ax3.set_ylabel('稳态值 x', fontsize=12)
ax3.set_title('立方映射分岔图', fontsize=16)

# 生成分岔图数据
print("正在生成分岔图数据...")
r_range = np.linspace(2.0, 3.5, 300)
bifurcation_x = []
bifurcation_y = []

for r_val in r_range:
    x_seq = generate_sequence(0.1, r_val, 200)
    # 只取后100个点作为稳态值
    for x_val in x_seq[-100:]:
        bifurcation_x.append(r_val)
        bifurcation_y.append(x_val)

ax3.scatter(bifurcation_x, bifurcation_y, s=0.1, c='magenta', alpha=0.5)
print("分岔图生成完成！")

# 初始化函数
def init():
    map_line.set_data([], [])
    map_points.set_data([], [])
    time_line.set_data([], [])
    return map_line, map_points, time_line

# 动画更新函数
def update(frame):
    # 计算当前帧对应的迭代次数
    n = int(frame * n_iterations / n_frames)
    if n < 2:
        n = 2  # 确保至少有两个点
    
    # 更新映射轨迹
    x_vals = full_sequence[:n-1]
    y_vals = full_sequence[1:n]
    
    # 绘制轨迹线
    map_line.set_data(x_vals, y_vals)
    
    # 绘制最后几个点
    last_n = min(20, n-1)
    map_points.set_data(x_vals[-last_n:], y_vals[-last_n:])
    
    # 更新时间序列
    time_line.set_data(range(n), full_sequence[:n])
    
    # 打印进度信息，帮助调试
    if frame % 30 == 0:
        print(f"处理帧: {frame}/{n_frames}, 迭代次数: {n}/{n_iterations}")
    
    return map_line, map_points, time_line

# 创建动画
ani = FuncAnimation(fig, update, frames=n_frames, 
                    init_func=init, blit=True, interval=5, repeat=True)  # 间隔从10减少到5

# 保存为GIF
print("开始生成GIF动画...")
start_time = time.time()
# 定义进度回调函数
def progress_callback(current_frame, total_frames):
    progress = (current_frame + 1) / total_frames * 100
    print(f"\r生成GIF进度: {progress:.1f}%", end="")
    if current_frame == total_frames - 1:
        print("\nGIF动画已保存完成")
try:
    ani.save('cubic_map.gif', writer='pillow', fps=80, dpi=60,  # fps从40增加到80
             progress_callback=progress_callback)
    print("GIF动画保存成功！")
except Exception as e:
    print(f"保存GIF时出错: {e}")
end_time = time.time()
print(f"总共用时: {end_time - start_time:.2f} 秒")

# 显示图像 - 确保动画能够运行
plt.tight_layout()
plt.show(block=True)  # 确保阻塞直到窗口关闭