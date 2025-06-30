import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

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
    # 读取图像数据并转换通道顺序
    image_data = src.read()
    transform = src.transform  # 地理坐标转像素坐标的变换矩阵
    crs = src.crs  # 图像的坐标系

    # 获取RGB通道（假设原始顺序为BGR）
    if image_data.shape[0] >= 3:
        rgb_channels = image_data[[2, 1, 0], :, :]  # BGR -> RGB
        image_rgb = rgb_channels.transpose(1, 2, 0)
    else:
        image_rgb = image_data.transpose(1, 2, 0)

    # 归一化处理
    image_rgb = (image_rgb / np.max(image_rgb) * 255).astype(np.uint8)

# --------------------------
# 2. 处理地理标签数据
# --------------------------
# 读取GeoJSON并转换坐标系
gdf = gpd.read_file(geojson_file).to_crs(crs)


def convert_coords(geometry):
    """将地理坐标转换为像素坐标（处理三维坐标）"""
    if geometry.geom_type == 'Polygon':
        return [~transform * (x, y) for x, y, *_ in geometry.exterior.coords]
    elif geometry.geom_type == 'MultiPolygon':
        return [[~transform * (x, y) for x, y, *_ in poly.exterior.coords]
                for poly in geometry.geoms]
    return []


# 添加像素坐标列
gdf['pixel_coords'] = gdf['geometry'].apply(convert_coords)

# --------------------------
# 3. 可视化设置
# --------------------------
plt.rcParams['figure.dpi'] = 150  # 提高显示分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存高分辨率图像

# --------------------------
# 4. 绘制原图（第一个图）
# --------------------------
fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.imshow(image_rgb)
ax1.set_title('Original Satellite Image', fontsize=12)
ax1.axis('off')
plt.tight_layout()
plt.show()

# --------------------------
# 5. 绘制标注图（第二个图）
# --------------------------
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.imshow(image_rgb)

# 绘制所有建筑物轮廓
for coords in gdf['pixel_coords']:
    if isinstance(coords[0], list):  # MultiPolygon
        for poly_coords in coords:
            x, y = zip(*poly_coords)
            ax2.plot(x, y, color='#FF0000', linewidth=1.2, alpha=0.8)
    else:  # Polygon
        x, y = zip(*coords)
        ax2.plot(x, y, color='#FF0000', linewidth=1.2, alpha=0.8)

ax2.set_title('Building Annotations Overlay', fontsize=12)
ax2.axis('off')
plt.tight_layout()
plt.show()

# --------------------------
# 6. 保存结果（可选）
# --------------------------
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# 保存原图
fig1.savefig(os.path.join(output_dir, 'original_image.jpg'),
             bbox_inches='tight', pad_inches=0.1)

# 保存标注图
fig2.savefig(os.path.join(output_dir, 'annotated_image.jpg'),
             bbox_inches='tight', pad_inches=0.1)

print(f"处理完成，结果已保存至：{os.path.abspath(output_dir)}")