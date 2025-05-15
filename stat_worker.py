import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QObject, pyqtSignal

# 全局配置：支持中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

class StatWorker(QObject):
    chart_ready = pyqtSignal(object)  # 发出 matplotlib Figure
    finished    = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(
        self,
        txt_file: str,
        num_classes: int = 12,
        class_names: list = None,
        colormap: str = "tab20"
    ):
        super().__init__()
        self.txt_file    = txt_file
        self.num_classes = num_classes
        self.class_names = class_names or [f"类别 {i}" for i in range(num_classes)]
        self.colormap    = colormap
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # 1. 文件检查
            if not os.path.isfile(self.txt_file):
                self.error.emit("结果文件不存在")
                return

            data = np.loadtxt(self.txt_file)
            if data.ndim != 2 or data.shape[1] < 4:
                self.error.emit("结果文件格式错误，应包含 x y z label 四列")
                return

            labels = data[:, 3].astype(int)
            if labels.min() < 0 or labels.max() >= self.num_classes:
                self.error.emit(f"标签值超出 [0, {self.num_classes-1}] 范围")
                return

            # 2. 统计各类数量
            counts = np.bincount(labels, minlength=self.num_classes)
            total  = counts.sum()
            if total == 0:
                self.error.emit("结果文件中没有标签数据")
                return

            # 3. 绘制饼图（不在扇区内标注）
            fig, ax = plt.subplots(figsize=(7, 6))
            cmap   = plt.get_cmap(self.colormap)
            colors = cmap(np.linspace(0, 1, self.num_classes))
            wedges, _ = ax.pie(
                counts,
                labels=[None]*self.num_classes,
                colors=colors,
                startangle=90,
                wedgeprops=dict(width=0.5)
            )
            ax.axis('equal')
            ax.set_title("类别占比统计")

            # 4. 在图例中显示类别名和百分比
            legend_labels = [
                f"{name}: {count/total:.1%}"
                for name, count in zip(self.class_names, counts)
            ]
            ax.legend(
                wedges,
                legend_labels,
                title="图例",
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
                fontsize='small'
            )

            # 5. 调整布局，给图例留出空间
            fig.subplots_adjust(right=0.75)

            # 6. 发出图形并结束
            self.chart_ready.emit(fig)
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
