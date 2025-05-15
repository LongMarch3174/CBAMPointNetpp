import os
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QMessageBox,
    QVBoxLayout
)
from PyQt5.QtCore import QThread
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from preprocess_worker import PreprocessWorker
from segment_worker import SegmentWorker
from stat_worker import StatWorker
from visualize_worker import VisualizeWorker  # 新增导入

MODEL_PATH = "weights/pycbam_550.pth"
OUTPUT_DIR = "results"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main.ui", self)

        # 功能区 1
        self.btnSelectSceneFile.clicked.connect(self.select_scene_file)
        self.btnStartPreprocess.clicked.connect(self.start_preprocess)

        # 功能区 2
        self.btnSelectDataset.clicked.connect(self.select_dataset)
        self.btnSegment.clicked.connect(self.start_segment)

        # 功能区 3 + 4
        self.btnSelectResultFile.clicked.connect(self.select_result_file)

        # 功能区 5：可视化按钮
        self.btnVisualize.clicked.connect(self.start_visualize)

        # 清空饼图显示
        self.clear_pie_chart()

        # 持有线程与 worker 引用，防止 gc
        self.preprocess_thread = None
        self.preprocess_worker = None
        self.segment_thread = None
        self.segment_worker = None
        self.stat_thread = None
        self.stat_worker = None
        self.vis_thread = None
        self.vis_worker = None

    # —— 功能区 1 —— #
    def select_scene_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择场景描述文件", "", "Text Files (*.txt)"
        )
        if path:
            self.lineEditSceneFile.setText(path)

    def start_preprocess(self):
        scene_txt = self.lineEditSceneFile.text().strip()
        if not scene_txt:
            QMessageBox.warning(self, "提示", "请先选择场景描述文件")
            return

        self.preprocess_thread = QThread(self)
        self.preprocess_worker = PreprocessWorker(
            input_file=scene_txt,
            output_dir=OUTPUT_DIR,
            num_points=4096
        )
        self.preprocess_worker.moveToThread(self.preprocess_thread)

        self.preprocess_worker.progress.connect(self.pbarPreprocess.setValue)
        self.preprocess_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "预处理出错", msg)
        )
        self.preprocess_worker.finished.connect(
            lambda: self.statusbar.showMessage("预处理完成", 3000)
        )
        self.preprocess_worker.finished.connect(self.preprocess_thread.quit)
        self.preprocess_thread.finished.connect(self.preprocess_worker.deleteLater)
        self.preprocess_thread.finished.connect(self.preprocess_thread.deleteLater)

        self.preprocess_thread.started.connect(self.preprocess_worker.run)
        self.preprocess_thread.start()

    # —— 功能区 2 —— #
    def select_dataset(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择数据和标签文件",
            "",
            "NumPy Files (*_data.npy *_label.npy)"
        )
        if len(files) != 2:
            QMessageBox.warning(self, "选择错误", "请同时选择一个 *_data.npy 和 一个 *_label.npy")
            return
        data_file = next(f for f in files if f.endswith("_data.npy"))
        label_file = next(f for f in files if f.endswith("_label.npy"))
        self.lineEditDataFile.setText(data_file)
        self.lineEditLabelFile.setText(label_file)

    def start_segment(self):
        data_file = self.lineEditDataFile.text().strip()
        label_file = self.lineEditLabelFile.text().strip()
        if not data_file or not label_file:
            QMessageBox.warning(self, "提示", "请先选择数据和标签文件")
            return

        self.segment_thread = QThread(self)
        self.segment_worker = SegmentWorker(
            model_path=MODEL_PATH,
            data_file=data_file,
            label_file=label_file,
            output_dir=OUTPUT_DIR,
            batch_size=4,
            gpu_id=0
        )
        self.segment_worker.moveToThread(self.segment_thread)

        self.segment_worker.progress.connect(self.pbarSegment.setValue)
        self.segment_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "分割出错", msg)
        )
        self.segment_worker.finished.connect(self.on_segment_finished)
        self.segment_worker.finished.connect(self.segment_thread.quit)
        self.segment_thread.finished.connect(self.segment_worker.deleteLater)
        self.segment_thread.finished.connect(self.segment_thread.deleteLater)

        self.segment_thread.started.connect(self.segment_worker.run)
        self.segment_thread.start()

    def on_segment_finished(self, result_path: str):
        self.lineEditResultFile.setText(result_path)
        self.statusbar.showMessage("分割完成", 3000)
        # 自动触发统计
        self.start_stat()

    # —— 功能区 3 + 4 —— #
    def select_result_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择分割结果文件", "", "Text Files (*.txt)"
        )
        if path:
            self.lineEditResultFile.setText(path)
            self.start_stat()

    def start_stat(self):
        txt_file = self.lineEditResultFile.text().strip()
        if not txt_file:
            QMessageBox.warning(self, "提示", "请先选择结果文件")
            return

        self.stat_thread = QThread(self)
        self.stat_worker = StatWorker(
            txt_file=txt_file,
            num_classes=12,
            class_names=[
                "bridge","building","water","tree","veg","low_veg",
                "light","electric","ground","others","vehicle","non-vehicle"
            ],
            colormap="tab20"
        )
        self.stat_worker.moveToThread(self.stat_thread)

        self.stat_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "统计出错", msg)
        )
        self.stat_worker.chart_ready.connect(self.display_pie_chart)
        self.stat_worker.finished.connect(
            lambda: self.statusbar.showMessage("统计完成", 3000)
        )
        self.stat_worker.finished.connect(self.stat_thread.quit)
        self.stat_thread.finished.connect(self.stat_worker.deleteLater)
        self.stat_thread.finished.connect(self.stat_thread.deleteLater)

        self.stat_thread.started.connect(self.stat_worker.run)
        self.stat_thread.start()

    # —— 功能区 5：可视化 —— #
    def start_visualize(self):
        txt = self.lineEditResultFile.text().strip()
        if not txt:
            QMessageBox.warning(self, "提示", "请先选择结果文件")
            return

        self.vis_thread = QThread(self)
        self.vis_worker = VisualizeWorker(
            txt_file=txt,
            colormap="tab20",
            point_size=7.5
        )
        self.vis_worker.moveToThread(self.vis_thread)

        self.vis_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "可视化错误", msg)
        )
        self.vis_worker.finished.connect(
            lambda: self.statusbar.showMessage("可视化完成", 3000)
        )
        self.vis_worker.finished.connect(self.vis_thread.quit)
        self.vis_thread.finished.connect(self.vis_worker.deleteLater)
        self.vis_thread.finished.connect(self.vis_thread.deleteLater)

        self.vis_thread.started.connect(self.vis_worker.run)
        self.vis_thread.start()

    # —— 绘制饼图到 UI —— #
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
