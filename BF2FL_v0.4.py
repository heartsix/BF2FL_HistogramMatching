import cv2
import numpy as np
import matplotlib.pyplot as plt
from color_matcher import ColorMatcher

# 读取图像
bf_img = cv2.imread('./20241128TB model 采集/微珠-8000/TB-微珠-1.bmp')  # Bright-field image
fl_img = cv2.imread('./20241128TB model 采集/微珠-8000/FL-微珠-1.bmp')  # Fluorescence image

if bf_img is None or fl_img is None:
    print("无法加载图像，请检查文件路径！")
    exit()

# 将图像从BGR转换为RGB 
bf_img_rgb = cv2.cvtColor(bf_img, cv2.COLOR_BGR2RGB)
fl_img_rgb = cv2.cvtColor(fl_img, cv2.COLOR_BGR2RGB)

# 使用 ColorMatcher 进行颜色匹配
cm = ColorMatcher()
matched_img = cm.transfer(src=bf_img_rgb, ref=fl_img_rgb, method='default')
matched_img = matched_img.astype(np.uint8)

# 计算灰度残差
bf_gray = cv2.cvtColor(matched_img, cv2.COLOR_RGB2GRAY)
fl_gray = cv2.cvtColor(fl_img, cv2.COLOR_BGR2GRAY)
residual_gray = cv2.absdiff(bf_gray, fl_gray)

# 计算并打印最大灰度差值
max_residual = np.max(residual_gray)
mean_residual = np.mean(residual_gray)
print(f"最大灰度差值: {max_residual}")
print(f"灰度差均值: {mean_residual}")

# 绘制图像和直方图
plt.figure(figsize=(15, 10), dpi=420)
plt.suptitle("Images and Histograms", fontsize=16)

# 子图1：原始 BF 图像
plt.subplot(2, 3, 1)
plt.imshow(bf_img_rgb)
plt.title("Original BF Image")
plt.axis('off')

# 子图2：原始 FL 图像
plt.subplot(2, 3, 2)
plt.imshow(fl_img_rgb)
plt.title("Original FL Image")
plt.axis('off')

# 子图3：匹配后的 BF 图像
plt.subplot(2, 3, 3)
plt.imshow(matched_img)
plt.title("Matched BF Image")
plt.axis('off')

# 子图4：原始 BF 图像直方图
plt.subplot(2, 3, 4)
for i, color in enumerate(['r', 'g', 'b']):
    hist = cv2.calcHist([bf_img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f"Channel {color.upper()}")
plt.title("Histogram of BF Image")
plt.legend()

# 子图5：原始 FL 图像直方图
plt.subplot(2, 3, 5)
for i, color in enumerate(['r', 'g', 'b']):
    hist = cv2.calcHist([fl_img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f"Channel {color.upper()}")
plt.title("Histogram of FL Image")
plt.legend()

# 子图6：匹配后的 BF 图像直方图
plt.subplot(2, 3, 6)
for i, color in enumerate(['r', 'g', 'b']):
    hist = cv2.calcHist([matched_img], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f"Channel {color.upper()}")
plt.title("Histogram of Matched Image")
plt.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# 单独绘制残差图和残差分布
plt.figure(figsize=(12, 6), dpi=150)

# 子图1：残差图
plt.subplot(1, 2, 1)
plt.title("Residual (Gray Difference)", fontsize=16)
im = plt.imshow(residual_gray, cmap='hot')
plt.colorbar(im)
plt.axis('off')

# 子图2：残差分布直方图
plt.subplot(1, 2, 2)
plt.hist(residual_gray.flatten(),range=(0,25), bins=25, color='gray', alpha=0.7)
plt.title("Residual Distribution", fontsize=16)
plt.xlabel("Residual Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# 计算每个通道的残差
residual_rgb = cv2.absdiff(matched_img, fl_img_rgb)

# 计算并打印每个通道的最大灰度差值
max_residual_r = np.max(residual_rgb[:, :, 0])
max_residual_g = np.max(residual_rgb[:, :, 1])
max_residual_b = np.max(residual_rgb[:, :, 2])
print(f"R通道最大灰度差值: {max_residual_r}")
print(f"G通道最大灰度差值: {max_residual_g}")
print(f"B通道最大灰度差值: {max_residual_b}")

# 绘制每个通道的残差图和残差分布
plt.figure(figsize=(15, 10), dpi=150)
colors = ['Reds', 'Greens', 'Blues']
titles = ['R Channel Residual', 'G Channel Residual', 'B Channel Residual']

for i in range(3):
    # 子图1-3：每个通道的残差图
    plt.subplot(3, 2, i * 2 + 1)
    plt.title(titles[i], fontsize=14)
    im = plt.imshow(residual_rgb[:, :, i], cmap=colors[i])
    plt.colorbar(im)
    plt.axis('off')

    # 子图2-4：每个通道的残差直方图
    plt.subplot(3, 2, i * 2 + 2)
    channel_data = residual_rgb[:, :, i].flatten()
    # 绘制直方图
    plt.hist(channel_data,range=(0,50), bins=50, color=colors[i][0].lower(), alpha=0.7)
    plt.title(f"{titles[i]} Distribution", fontsize=14)
    plt.xlabel("Residual Intensity")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
