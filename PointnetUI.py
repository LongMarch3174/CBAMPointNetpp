from PyQt5.QtWidgets import QApplication
import sys
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QApplication, QVBoxLayout
from PyQt5.QtCore import QThread
from PyQt5 import uic
import os
from visualize_and_stat_worker import VisualizeAndStatWorker
from preprocess_worker import PreprocessWorker
from segment_worker import SegmentWorker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main.ui", self)

        # UI 元素绑定
        self.btnSelectScenes.clicked.connect(self.select_scene_files)
        self.btnStartPreprocess.clicked.connect(self.start_preprocessing)
        self.btnSegment.clicked.connect(self.start_segmentation)

        # 目录路径占位
        self.scene_files = []
        self.scene_dir = ""
        self.output_dir = "results"
        self.model_path = "weights/pycbam_550.pth"

        self.clear_pie_chart()

    def select_scene_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择场景文件", "", "Text Files (*.txt)")
        if files:
            self.scene_files = files
            self.scene_dir = os.path.dirname(files[0])
            self.listScenes.clear()
            for f in files:
                item = QListWidgetItem(os.path.basename(f))
                self.listScenes.addItem(item)

    def start_preprocessing(self):
        if not self.scene_dir:
            return

        self.thread = QThread()
        self.worker = PreprocessWorker(self.scene_dir, self.scene_dir)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.pbarPreprocess.setValue)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(lambda: self.statusbar.showMessage("预处理完成", 3000))
        self.worker.error.connect(lambda msg: self.statusbar.showMessage(f"预处理错误: {msg}", 5000))
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def start_segmentation(self):
        self.thread = QThread()
        self.worker = SegmentWorker(
            model_path=self.model_path,
            data_dir=self.scene_dir,
            output_dir=self.output_dir
        )
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.pbarSegment.setValue)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.start_visualization_and_stat)
        self.worker.error.connect(lambda msg: self.statusbar.showMessage(f"分割错误: {msg}", 5000))
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def start_visualization_and_stat(self):
        result_file = os.path.join(self.output_dir, "segmentation_result.txt")
        self.thread = QThread()
        self.worker = VisualizeAndStatWorker(
            txt_file_path=result_file,
            class_names=["bridge", "building", "water", "tree", "veg", "low_veg", "light", "electric", "ground", "others", "vehicle", "non-vehicle"]
        )
        self.worker.moveToThread(self.thread)
        self.worker.chart_ready.connect(self.display_pie_chart)
        self.worker.error.connect(lambda msg: self.statusbar.showMessage(f"可视化错误: {msg}", 5000))
        self.worker.finished.connect(lambda: self.statusbar.showMessage("可视化完成", 3000))
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def display_pie_chart(self, fig):
        self.clear_pie_chart()
        canvas = FigureCanvas(fig)
        layout = self.pieChartWidget.layout()
        if layout is None:
            layout = QVBoxLayout(self.pieChartWidget)
            self.pieChartWidget.setLayout(layout)
        layout.addWidget(canvas)

    def clear_pie_chart(self):
        layout = self.pieChartWidget.layout()
        if layout:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
