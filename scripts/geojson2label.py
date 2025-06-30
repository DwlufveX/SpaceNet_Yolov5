import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
# 输入文件配置
image_file = 'RGB-PanSharpen_AOI_5_Khartoum_img1.tif'
geojson_file = 'buildings_AOI_5_Khartoum_img1.geojson'

# 检查文件存在性
assert os.path.exists(image_file), f"图像文件不存在: {image_file}"
assert os.path.exists(geojson_file), f"GeoJSON文件不存在: {geojson_file}"

# --------------------------
# 1. 读取卫星图像
# --------------------------
with rasterio.open(image_file) as src:
    image_data = src.read()
    transform = src.transform
    crs = src.crs

    # 获取RGB通道
    if image_data.shape[0] >= 3:
        rgb_channels = image_data[[2, 1, 0], :, :]
        image_rgb = rgb_channels.transpose(1, 2, 0)
    else:
        image_rgb = image_data.transpose(1, 2, 0)

    # 归一化处理
    image_rgb = (image_rgb / np.max(image_rgb) * 255).astype(np.uint8)

height, width = image_rgb.shape[:2]

# --------------------------
# 2. 处理地理标签数据
# --------------------------
gdf = gpd.read_file(geojson_file).to_crs(crs)

# YOLO 标签保存路径
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)
yolo_label_file = os.path.join(output_dir, os.path.basename(image_file).replace('.tif', '.txt'))


# 写入 YOLO 格式标签（过滤面积过小的目标）
min_width_px = 0
min_height_px = 0

with rasterio.open(image_file) as src:
    image_data = src.read()
    transform = src.transform
    crs = src.crs
    height, width = image_data.shape[1:]

gdf = gpd.read_file(geojson_file).to_crs(crs)

# label_file = os.path.join(label_output_dir, image_file.replace('.tif', '.txt'))
with open(yolo_label_file, 'w') as f:
    for _, row in gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        x1, y1 = ~transform * (minx, miny)
        x2, y2 = ~transform * (maxx, maxy)

        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height

        box_w = abs(x2 - x1)
        box_h = abs(y2 - y1)
        # 过滤条件：小于阈值则跳过
        if box_w < min_width_px or box_h < min_height_px:
            continue

        box_width = box_w / width
        box_height = box_h / height


        # 仅写入合法框
        if 0 < box_width <= 1 and 0 < box_height <= 1:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print(f"YOLO 标签已保存至：{yolo_label_file}")

# --------------------------
# 3. 可视化标签和图像
# --------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(image_rgb)

# 绘制 YOLO 标签
with open(yolo_label_file, 'r') as f:
    for line in f.readlines():
        _, xc, yc, w, h = map(float, line.strip().split())
        xc, yc, w, h = xc * width, yc * height, w * width, h * height
        rect = plt.Rectangle((xc - w / 2, yc - h / 2), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

ax.set_title('YOLO HBB (水平边界框) 标签')
plt.show()
