#!/usr/bin/env python3
# Seg.py
# PyQt5 UI script for scene segmentation & statistics

import sys
import os
import shutil
import subprocess
import numpy as np

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
)
from visualize import visualize_point_cloud_txt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Load the .ui file (assumed saved as 'SegUI.ui' in same dir)
        uic.loadUi('SegUI.ui', self)

        # Internal state
        self.selected_txts = []    # list of selected .txt paths
        self.npy_dir = ''          # directory where .txt → .npy conversion lives
        self.seg_results = []      # list of segmentation result file paths

        # Connect signals
        self.btnSelectScenes.clicked.connect(self.select_scenes)
        self.btnStartPreprocess.clicked.connect(self.preprocess)
        self.btnSegment.clicked.connect(self.segment)
        self.listSegFiles.itemDoubleClicked.connect(self.visualize_and_stat)

    def select_scenes(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            '选择场景文件',
            '',
            'Text Files (*.txt)'
        )
        if not files:
            return
        self.selected_txts = files
        self.listScenes.clear()
        for f in files:
            self.listScenes.addItem(QListWidgetItem(os.path.basename(f)))

    def preprocess(self):
        if not self.selected_txts:
            QMessageBox.warning(self, '警告', '请先选择场景文件')
            return

        # Create (or clean) a temp directory for preprocessing
        self.npy_dir = os.path.join(os.getcwd(), 'preprocess_tmp')
        if os.path.exists(self.npy_dir):
            shutil.rmtree(self.npy_dir)
        os.makedirs(self.npy_dir)

        # Copy selected .txt files into that directory
        for f in self.selected_txts:
            shutil.copy(f, self.npy_dir)

        # Run data_process.py on that directory
        self.pbarPreprocess.setRange(0, 0)  # indeterminate
        cmd = ['python3', 'data_process.py', self.npy_dir]
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                cwd=os.getcwd())
        out, err = proc.communicate()
        self.pbarPreprocess.setRange(0, 1)
        self.pbarPreprocess.setValue(1)

        if proc.returncode != 0:
            QMessageBox.critical(
                self,
                '错误',
                f'预处理失败:\n{err.decode("utf-8", "ignore")}'
            )
            return

        # Move the generated test_data.npy / test_label.npy into our temp dir
        for fname in ('test_data.npy', 'test_label.npy'):
            if os.path.exists(fname):
                shutil.move(fname, os.path.join(self.npy_dir, fname))

        # Populate the .npy list
        self.listNpyFiles.clear()
        self.listNpyFiles.addItem(QListWidgetItem('test_data.npy'))

    def segment(self):
        npy_path = os.path.join(self.npy_dir, 'test_data.npy')
        if not os.path.exists(npy_path):
            QMessageBox.warning(self, '警告', '请先完成预处理')
            return

        # Run the segmentation script
        self.pbarSegment.setRange(0, 0)
        cmd = [
            'python3', 'testSeg_ISPRS_class.py',
            '--gpu', '0',
            '--num_points', '4096'
        ]
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                cwd=os.getcwd())
        out, err = proc.communicate()
        self.pbarSegment.setRange(0, 1)
        self.pbarSegment.setValue(1)

        if proc.returncode != 0:
            QMessageBox.critical(
                self,
                '错误',
                f'分割失败:\n{err.decode("utf-8", "ignore")}'
            )
            return

        # Assume the segmentation script writes "area_cbam.txt" to a known location
        result_src = '/root/PointNet++/test/area_cbam.txt'
        if not os.path.exists(result_src):
            QMessageBox.critical(self, '错误', '未找到分割结果文件')
            return

        dst = os.path.join(self.npy_dir, 'seg_result.txt')
        shutil.copy(result_src, dst)
        self.seg_results = [dst]

        # Populate the segmentation results list
        self.listSegFiles.clear()
        self.listSegFiles.addItem(QListWidgetItem(os.path.basename(dst)))

    def visualize_and_stat(self, item):
        # Full path of selected result
        file_path = os.path.join(self.npy_dir, item.text())

        # 1) Visualize with Open3D
        visualize_point_cloud_txt(file_path)

        # 2) Compute per-class counts
        data = np.loadtxt(file_path)
        labels = data[:, 3].astype(int)
        unique, counts = np.unique(labels, return_counts=True)

        # 3) Plot a pie chart into the pieChartWidget
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.pie(
            counts,
            labels=[str(u) for u in unique],
            autopct='%1.1f%%'
        )
        canvas = FigureCanvas(fig)

        # Clear previous chart
        layout = self.pieChartWidget.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(self.pieChartWidget)
            self.pieChartWidget.setLayout(layout)
        else:
            # remove old widgets
            while layout.count():
                w = layout.takeAt(0).widget()
                if w:
                    w.setParent(None)

        layout.addWidget(canvas)
        canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
