import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import time
from PIL import Image
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置更好的图形风格
plt.style.use('dark_background')
mpl.rcParams['figure.figsize'] = [12, 8]

# 猫映射函数
def cat_map(x, y, a=1, b=1):
    """
    阿诺德猫映射函数
    (x', y') = ((x + y) mod 1, (x + a*y) mod 1)
    """
    x_new = (x + y) % 1
    y_new = (a*x + (a*b+1)*y) % 1
    return x_new, y_new

# 创建一个简单的图像
def create_test_image(size=200):
    """创建一个测试图像，包含一些简单的几何形状"""
    img = np.zeros((size, size))
    
    # 添加一个圆形
    center = size // 2
    radius = size // 4
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    circle = dist_from_center <= radius
    img[circle] = 1
    
    # 添加一个矩形
    rect_size = size // 6
    rect_pos = size // 3
    img[rect_pos:rect_pos+rect_size, rect_pos:rect_pos+rect_size] = 0.7
    
    # 添加一个三角形
    for i in range(size//3):
        width = i // 2
        pos = size*2//3
        if pos-width >= 0 and pos+width < size and pos+i < size:
            img[pos+i, pos-width:pos+width] = 0.4
    
    return img

# 应用猫映射到图像
def apply_cat_map(image, iterations=1, a=1, b=1):
    """对图像应用猫映射变换"""
    h, w = image.shape
    result = np.copy(image)
    
    for _ in range(iterations):
        new_result = np.zeros_like(result)
        for y in range(h):
            for x in range(w):
                # 归一化坐标到[0,1)区间
                x_norm, y_norm = x/w, y/h
                # 应用猫映射
                x_new_norm, y_new_norm = cat_map(x_norm, y_norm, a, b)
                # 转换回图像坐标
                x_new, y_new = int(x_new_norm * w), int(y_new_norm * h)
                new_result[y_new, x_new] = result[y, x]
        result = new_result
    
    return result

# 参数设置
image_size = 200
max_iterations = 30
a, b = 1, 1  # 猫映射参数

# 创建或加载图像
try:
    # 尝试加载猫图像（如果存在）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cat_image_path = os.path.join(script_dir, "cat.jpg")
    
    if os.path.exists(cat_image_path):
        print("加载猫图像...")
        original_image = Image.open(cat_image_path).convert('L')
        original_image = original_image.resize((image_size, image_size))
        original_image = np.array(original_image) / 255.0
    else:
        print("未找到猫图像，创建测试图像...")
        original_image = create_test_image(image_size)
except Exception as e:
    print(f"加载图像出错: {e}，创建测试图像...")
    original_image = create_test_image(image_size)

# 预计算所有迭代的图像
print("正在计算猫映射变换序列...")
image_sequence = [original_image]
current_image = original_image

for i in range(max_iterations):
    current_image = apply_cat_map(current_image, 1, a, b)
    image_sequence.append(current_image)
    
print(f"已完成 {max_iterations} 次迭代计算")

# 创建图形
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('black')

# 创建左侧图形：显示当前迭代的图像
ax1 = fig.add_subplot(121)
ax1.set_facecolor('black')
ax1.set_title('猫映射迭代图像', fontsize=16)
ax1.axis('off')

# 创建右上图形：显示公式
ax_formula = fig.add_subplot(222)
ax_formula.axis('off')
ax_formula.set_facecolor('black')

# 添加猫映射公式 - 修复公式显示问题
ax_formula.text(0.5, 0.8, "猫映射方程:", fontsize=14, color='white', 
         horizontalalignment='center')

# 使用 \bmod 或简单地使用文本 "mod" 代替 \mod
ax_formula.text(0.5, 0.6, r"$x_{n+1} = (x_n + y_n) \; \mathrm{mod} \; 1$", fontsize=14, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.4, r"$y_{n+1} = (ax_n + (ab+1)y_n) \; \mathrm{mod} \; 1$", fontsize=14, color='white', 
         horizontalalignment='center')

ax_formula.text(0.5, 0.2, r"参数: $a = " + f"{a}" + r"$, $b = " + f"{b}" + r"$", fontsize=14, color='white', 
         horizontalalignment='center')

# 创建右下图形：显示迭代次数与混沌程度关系
ax3 = fig.add_subplot(224)
ax3.set_facecolor('black')
ax3.set_xlim(0, max_iterations)
ax3.set_ylim(0, 1)
ax3.set_xlabel('迭代次数', fontsize=12)
ax3.set_ylabel('混沌度', fontsize=12)
ax3.set_title('猫映射混沌度变化', fontsize=16)

# 计算每次迭代的混沌度（使用图像熵作为度量）
def calculate_entropy(image):
    """计算图像的熵作为混沌度的度量"""
    hist = np.histogram(image, bins=20, range=(0,1))[0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy / np.log2(20)  # 归一化到[0,1]

entropy_values = [calculate_entropy(img) for img in image_sequence]
ax3.plot(range(len(entropy_values)), entropy_values, 'c-', lw=1.5)

# 初始化图像显示
img_display = ax1.imshow(original_image, cmap='viridis', interpolation='nearest')
iteration_text = ax1.text(0.05, 0.95, "迭代次数: 0", transform=ax1.transAxes, 
                         color='white', fontsize=12)

# 动画更新函数
def update(frame):
    img_display.set_array(image_sequence[frame])
    iteration_text.set_text(f"迭代次数: {frame}")
    return [img_display, iteration_text]

# 创建动画
ani = FuncAnimation(fig, update, frames=len(image_sequence), 
                    interval=50, blit=True, repeat=True)  # 间隔从100减少到50

# 保存为GIF
print("开始生成GIF动画...")
start_time = time.time()
# 定义进度回调函数
def progress_callback(current_frame, total_frames):
    progress = (current_frame + 1) / total_frames * 100
    print(f"\r生成GIF进度: {progress:.1f}%", end="")
    if current_frame == total_frames - 1:
        print("\nGIF动画已保存完成")
        
ani.save('cat_map.gif', writer='pillow', fps=20, dpi=60,  # fps从10增加到20
         progress_callback=progress_callback)
end_time = time.time()
print(f"总共用时: {end_time - start_time:.2f} 秒")

# 显示图像
plt.tight_layout()
plt.show()