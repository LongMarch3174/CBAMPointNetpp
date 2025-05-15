import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def visualize_point_cloud_txt(
    file_path: str,
    point_size: int = 12.5,
    colormap: str = "tab20"
):
    """
    使用 Open3D 可视化带标签的点云。

    参数：
        file_path (str): txt 文件路径，每行应为 'x y z label'；
        point_size (int): 点的大小（渲染时使用），默认 5；
        colormap (str): matplotlib 支持的 colormap 名称，用于标签上色，默认 'tab20'。

    示例：
        visualize_point_cloud_txt("area_scene_0.txt")
        visualize_point_cloud_txt("area_scene_0.txt", point_size=10, colormap="rainbow")
    """
    # 1. 读取数据
    data = np.loadtxt(file_path)  # shape = (N, ≥4)
    points = data[:, :3]
    labels = data[:, 3].astype(int)

    # 2. 生成颜色：根据标签映射到一个离散 colormap
    cmap = plt.get_cmap(colormap)
    num_classes = labels.max() + 1
    colors = cmap(labels / max(num_classes - 1, 1))[:, :3]  # 归一化到 [0,1]

    # 3. 构造 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 4. 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud", width=1200, height=900)
    vis.add_geometry(pcd)

    # 设置渲染选项：点大小
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.asarray([0, 0, 0])  # 可选：黑色背景

    # 运行并销毁窗口
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # 默认调用
    visualize_point_cloud_txt("results/segmentation_result.txt")
