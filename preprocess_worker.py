import os
import math
import numpy as np
import torch
from PyQt5.QtCore import QObject, pyqtSignal
from proceed_utils import farthest_point_sample, coordinate_normalize2

class PreprocessWorker(QObject):
    progress = pyqtSignal(int)   # 发出 [0..100]
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_file: str, output_dir: str, num_points: int = 4096):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.num_points = num_points
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        """处理单个场景描述 TXT 文件，按网格分块采样并保存"""
        try:
            # —— 1. 读取并解析文件 —— #
            if not os.path.isfile(self.input_file):
                self.error.emit("输入文件不存在")
                return

            lines = []
            with open(self.input_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            lines.append([float(x) for x in parts[:5]])
                        except:
                            self.error.emit("数值转换错误")
                            return
            if not lines:
                self.error.emit("文件中无有效点数据")
                return

            data_np = np.array(lines)  # shape = (N,5)
            X, Y = data_np[:,0], data_np[:,1]

            # —— 2. 网格参数 —— #
            Xmin, Xmax = X.min(), X.max()
            Ymin, Ymax = Y.min(), Y.max()
            nx = math.ceil((Xmax - Xmin) / 100) or 1
            ny = math.ceil((Ymax - Ymin) / 100) or 1
            total_cells = nx * ny

            # —— 3. 计算每点归属 cell_index —— #
            rows = np.floor((Y - Ymin) / 100).astype(int)
            cols = np.floor((X - Xmin) / 100).astype(int)
            rows = np.clip(rows, 0, ny-1)
            cols = np.clip(cols, 0, nx-1)
            cell_indices = rows * nx + cols

            # —— 4. 分桶收集索引 —— #
            buckets = [[] for _ in range(total_cells)]
            for idx_pt, c in enumerate(cell_indices):
                buckets[c].append(idx_pt)

            # —— 5. 对每个 cell 做 FPS + 归一化 —— #
            data_list = []
            label_list = []
            xyz_all_cpu = data_np[:, :3]  # 全局点集 (CPU)
            for ci, idx_list in enumerate(buckets):
                if not self._is_running:
                    return
                if len(idx_list) < self.num_points:
                    # 不够点时直接跳过，仍然更新进度
                    self.progress.emit(int((ci+1)/total_cells*100))
                    continue

                # 在 GPU 上做 FPS 和归一化
                block = data_np[idx_list]                   # (M,5) CPU
                block_t = torch.from_numpy(block).cuda()    # (M,5) GPU
                xyz_all = torch.from_numpy(xyz_all_cpu).cuda()

                # FPS 采样
                fps_idx = farthest_point_sample(
                    block_t[:, :3].unsqueeze(0),
                    self.num_points
                )                                           # (1, num_points)
                sampled = block_t[fps_idx.squeeze(0)][:, :5]# (num_points,5)

                # 归一化
                new_xyz = coordinate_normalize2(
                    xyz_all, sampled[:, :3]
                )                                           # (num_points,3)

                # 标签
                lbl = sampled[:, 4].unsqueeze(1)            # (num_points,1)

                data_list.append(new_xyz.unsqueeze(0))      # (1,num_points,3)
                label_list.append(lbl.unsqueeze(0))         # (1,num_points,1)

                # 更新进度
                self.progress.emit(int((ci+1)/total_cells*100))

            # —— 6. 保存输出 —— #
            if not data_list:
                self.error.emit("所有网格块点数不足，未生成任何输出")
                return

            out_data = torch.cat(data_list, dim=0).cpu().numpy()
            out_label = torch.cat(label_list, dim=0).squeeze(-1).cpu().numpy()
            base = os.path.splitext(os.path.basename(self.input_file))[0]
            os.makedirs(self.output_dir, exist_ok=True)
            np.save(os.path.join(self.output_dir, f"{base}_data.npy"), out_data)
            np.save(os.path.join(self.output_dir, f"{base}_label.npy"), out_label)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
