from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 读取图像
bf_img = cv2.imread('TB-empty.png')  # Bright-field image
fl_img = cv2.imread('FL-empty.png')  # Fluorescence image

if bf_img is None or fl_img is None:
    print("无法加载图像，请检查文件路径！")
    exit()

# 将图像从 BGR 转换为 RGB
bf_img_rgb = cv2.cvtColor(bf_img, cv2.COLOR_BGR2RGB)
fl_img_rgb = cv2.cvtColor(fl_img, cv2.COLOR_BGR2RGB)

# 将图像拉平成 (N, 3) 的点云
bf_points = bf_img_rgb.reshape(-1, 3).astype(float)
fl_points = fl_img_rgb.reshape(-1, 3).astype(float)

# 基于最近邻的多维分布匹配
def scatter_matching(source_points, target_points):
    """
    基于散点图的多维分布匹配。
    source_points: 源点云 (N, 3)
    target_points: 目标点云 (M, 3)
    返回与目标分布匹配的源点云。
    """
    # 使用最近邻方法找到最接近的目标点
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
    distances, indices = nbrs.kneighbors(source_points)

    # 用目标点云替换源点云
    matched_points = target_points[indices.flatten()]
    return matched_points

# 进行多维分布匹配
matched_points = scatter_matching(bf_points, fl_points)

# 将匹配后的点云转换回图像
matched_img = matched_points.reshape(bf_img_rgb.shape).astype(np.uint8)

# 绘制散点图比较（仅采样部分点以加速绘图）
sample_indices = np.random.choice(len(bf_points), size=1000, replace=False)
fig = plt.figure(figsize=(18, 6))

# 原始 BF 图像散点
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(bf_points[sample_indices, 0],
            bf_points[sample_indices, 1],
            bf_points[sample_indices, 2], c=bf_points[sample_indices] / 255.0)
ax1.set_title("BF Image Scatter Plot")

# 匹配后的图像散点
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(matched_points[sample_indices, 0],
            matched_points[sample_indices, 1],
            matched_points[sample_indices, 2], c=matched_points[sample_indices] / 255.0)
ax2.set_title("Matched Image Scatter Plot")

# 原始 FL 图像散点
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(fl_points[sample_indices, 0],
            fl_points[sample_indices, 1],
            fl_points[sample_indices, 2], c=fl_points[sample_indices] / 255.0)
ax3.set_title("FL Image Scatter Plot")

plt.show()

# 计算残差
residual_img = cv2.absdiff(fl_img_rgb, matched_img)

# 显示原始图像和匹配结果
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(bf_img_rgb)
plt.title("Original BF Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(matched_img)
plt.title("Matched Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(fl_img_rgb)
plt.title("Original FL Image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(residual_img, cmap='hot')
plt.colorbar(label='Residual Intensity')
plt.title("Residual Map")
plt.axis('off')

plt.tight_layout()
plt.show()
