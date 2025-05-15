import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PyQt5.QtCore import QObject, pyqtSignal

class VisualizeAndStatWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    chart_ready = pyqtSignal(object)  # matplotlib 图像 Figure
    vis_done = pyqtSignal()           # Open3D 视图完成后可触发（用于通知）

    def __init__(self, txt_file_path: str, num_classes: int = 12, class_names=None, colormap: str = "tab20", point_size: float = 5.0):
        super().__init__()
        self.txt_file_path = txt_file_path
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
        self.colormap = colormap
        self.point_size = point_size
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # 1. 加载数据
            data = np.loadtxt(self.txt_file_path)
            if data.shape[1] < 4:
                self.error.emit("数据格式应至少为 x y z label")
                return
            points = data[:, :3]
            labels = data[:, 3].astype(int)

            # 2. 可视化点云
            self._visualize_point_cloud(points, labels)

            # 3. 绘制饼状图
            self._draw_pie_chart(labels)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))

    def _visualize_point_cloud(self, points, labels):
        cmap = plt.get_cmap(self.colormap)
        colors = cmap(labels / max(self.num_classes - 1, 1))[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud", width=1200, height=900)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()

        self.vis_done.emit()

    def _draw_pie_chart(self, labels):
        counts = np.bincount(labels, minlength=self.num_classes)
        total = counts.sum()
        if total == 0:
            self.error.emit("标签数据为空")
            return
        ratios = counts / total
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.get_cmap(self.colormap)
        colors = cmap(np.linspace(0, 1, self.num_classes))

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=self.class_names,
            autopct=lambda pct: f"{pct:.1f}%",
            colors=colors,
            startangle=140
        )
        ax.axis('equal')
        ax.set_title("类别占比统计")
        self.chart_ready.emit(fig)
