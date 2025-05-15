import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PyQt5.QtCore import QObject, pyqtSignal

class VisualizeWorker(QObject):
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, txt_file: str, colormap: str = "tab20", point_size: float = 5.0):
        super().__init__()
        self.txt_file    = txt_file
        self.colormap    = colormap
        self.point_size  = point_size
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            if not os.path.isfile(self.txt_file):
                self.error.emit("文件不存在")
                return

            data = np.loadtxt(self.txt_file)
            if data.ndim != 2 or data.shape[1] < 4:
                self.error.emit("文件格式错误，应包含 x y z label")
                return

            points = data[:, :3]
            labels = data[:, 3].astype(int)

            # 颜色映射
            cmap = plt.get_cmap(self.colormap)
            normed = labels / labels.max() if labels.max() > 0 else labels
            colors = cmap(normed)[:, :3]

            # 构建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Point Cloud", width=800, height=600)
            vis.add_geometry(pcd)

            opt = vis.get_render_option()
            opt.point_size = self.point_size
            # 设置背景为黑色
            opt.background_color = np.asarray([0.0, 0.0, 0.0])

            vis.run()
            vis.destroy_window()

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
