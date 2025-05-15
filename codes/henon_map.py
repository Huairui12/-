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

# Hénon 映射函数
def henon_map(x, y, a=1.4, b=0.3):
    """
    Hénon 映射函数
    x_{n+1} = 1 - a*x_n^2 + y_n
    y_{n+1} = b*x_n
    """
    x_new = 1 - a * x**2 + y
    y_new = b * x
    return x_new, y_new

# 生成 Hénon 映射序列
def generate_sequence(x0, y0, a, b, n):
    """生成 n 个 Hénon 映射迭代点"""
    x_seq = np.zeros(n)
    y_seq = np.zeros(n)
    x_seq[0], y_seq[0] = x0, y0
    
    for i in range(1, n):
        x_seq[i], y_seq[i] = henon_map(x_seq[i-1], y_seq[i-1], a, b)
    
    return x_seq, y_seq

# 参数设置
a = 1.4  # 经典 Hénon 映射参数
b = 0.3
x0, y0 = 0, 0  # 初始点
n_iterations = 10000  # 迭代次数
n_frames = 500  # 动画帧数
n_skip = 100  # 跳过前 n_skip 个点（瞬态）

# 生成完整序列
print("正在生成 Hénon 映射序列...")
x_full, y_full = generate_sequence(x0, y0, a, b, n_iterations)
print("序列生成完成！")

# 创建图形
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('black')

# 创建左侧图形：Hénon 映射相空间
ax1 = fig.add_subplot(121)
ax1.set_facecolor('black')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.5, 0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Hénon 映射相空间', fontsize=16)

# 初始化相空间点
scatter = ax1.scatter([], [], s=0.5, c='cyan', alpha=0.7)

# 创建右上图形：公式显示区域
ax_formula = fig.add_subplot(222)
ax_formula.axis('off')
ax_formula.set_facecolor('black')

# 添加 Hénon 映射公式
ax_formula.text(0.5, 0.8, "Hénon 映射方程:", fontsize=14, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.6, r"$x_{n+1} = 1 - a \cdot x_n^2 + y_n$", fontsize=14, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.4, r"$y_{n+1} = b \cdot x_n$", fontsize=14, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.2, r"参数: $a = " + f"{a}" + r"$, $b = " + f"{b}" + r"$", fontsize=14, color='white', 
         horizontalalignment='center')

# 创建右下图形：Lyapunov 指数
ax3 = fig.add_subplot(224)
ax3.set_facecolor('black')
ax3.set_xlim(0, n_frames)
ax3.set_ylim(-0.5, 0.5)
ax3.set_xlabel('迭代次数 (×20)', fontsize=12)
ax3.set_ylabel('Lyapunov 指数估计', fontsize=12)
ax3.set_title('Hénon 映射 Lyapunov 指数', fontsize=16)

# 计算 Lyapunov 指数
def estimate_lyapunov(x_seq, y_seq, a, b, n_points):
    """估计 Hénon 映射的 Lyapunov 指数"""
    epsilon = 1e-10  # 微小扰动
    lyapunov_sum = 0
    
    # 使用有限差分法估计 Lyapunov 指数
    for i in range(n_points):
        if i >= len(x_seq) - 1:
            break
            
        # 当前点
        x, y = x_seq[i], y_seq[i]
        
        # 添加微小扰动
        x_perturbed = x + epsilon
        
        # 计算一步迭代后的差异
        next_x, next_y = henon_map(x, y, a, b)
        next_x_perturbed, next_y_perturbed = henon_map(x_perturbed, y, a, b)
        
        # 计算扰动的增长率
        d0 = epsilon
        d1 = np.sqrt((next_x_perturbed - next_x)**2 + (next_y_perturbed - next_y)**2)
        
        # 累加 ln(d1/d0)
        if d1 > 0:
            lyapunov_sum += np.log(d1/d0)
    
    # 返回平均值作为 Lyapunov 指数估计
    return lyapunov_sum / n_points

# 计算不同迭代次数下的 Lyapunov 指数
lyapunov_values = []
for i in range(n_frames):
    n_points = min(200, (i+1)*20)  # 逐渐增加计算点数
    lyapunov = estimate_lyapunov(x_full, y_full, a, b, n_points)
    lyapunov_values.append(lyapunov)

# 绘制 Lyapunov 指数曲线
lyapunov_line, = ax3.plot([], [], 'magenta', lw=1.5)

# 动画更新函数
def update(frame):
    # 计算当前帧对应的迭代次数
    n = int(frame * (n_iterations - n_skip) / n_frames) + n_skip
    
    # 更新相空间点
    scatter.set_offsets(np.column_stack((x_full[n_skip:n], y_full[n_skip:n])))
    
    # 更新 Lyapunov 指数曲线
    lyapunov_line.set_data(range(frame+1), lyapunov_values[:frame+1])
    
    return scatter, lyapunov_line

# 创建动画
ani = FuncAnimation(fig, update, frames=n_frames, 
                    interval=5, blit=True, repeat=True)  # 间隔从10减少到5

# 保存为GIF
print("开始生成GIF动画...")
start_time = time.time()
# 定义进度回调函数
def progress_callback(current_frame, total_frames):
    progress = (current_frame + 1) / total_frames * 100
    print(f"\r生成GIF进度: {progress:.1f}%", end="")
    if current_frame == total_frames - 1:
        print("\nGIF动画已保存完成")
ani.save('henon_map.gif', writer='pillow', fps=120, dpi=60,  # fps从60增加到120
         progress_callback=progress_callback)
end_time = time.time()
print(f"总共用时: {end_time - start_time:.2f} 秒")

# 显示图像
plt.tight_layout()
plt.show()