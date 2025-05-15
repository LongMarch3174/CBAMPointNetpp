import os
import math
import numpy as np
import torch
from PyQt5.QtCore import QObject, pyqtSignal
from proceed_utils import farthest_point_sample, coordinate_normalize2

class PreprocessWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_dir: str, output_dir: str, num_points=4096):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_points = num_points
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            all_data = []
            all_label = []
            file_list = sorted(os.listdir(self.input_dir))
            total_files = len(file_list)

            for i, filename in enumerate(file_list):
                if not self._is_running:
                    break

                file_path = os.path.join(self.input_dir, filename)
                if not os.path.isfile(file_path):
                    continue

                # 1. 读入并构造 Nx5 numpy 数组
                lines = []
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                lines.append([float(x) for x in parts[:5]])
                            except:
                                self.error.emit(f"数值转换错误：{filename}")
                                return
                if len(lines) == 0:
                    self.error.emit(f"没有有效行：{filename}")
                    return
                input_data = np.array(lines)  # shape = (N,5)

                # 2. 计算网格划分参数
                X = input_data[:, 0]
                Y = input_data[:, 1]
                Xmin, Xmax = X.min(), X.max()
                Ymin, Ymax = Y.min(), Y.max()
                # 每 100 单位一个网格
                a = math.ceil((Xmax - Xmin) / 100) or 1
                b = math.ceil((Ymax - Ymin) / 100) or 1
                allblock = a * b

                # 3. 向量化：计算每个点所属的行/列索引
                #    floor((coord - min) / 100)
                rows = np.floor((Y - Ymin) / 100).astype(int)
                cols = np.floor((X - Xmin) / 100).astype(int)
                # 越界 clamp
                rows = np.clip(rows, 0, b - 1)
                cols = np.clip(cols, 0, a - 1)
                cell_indices = rows * a + cols  # shape = (N,)

                # 4. 根据 cell_indices 构建 list2
                list2 = [[] for _ in range(allblock)]
                for idx_point, cell in enumerate(cell_indices):
                    list2[cell].append(idx_point)

                # 5. 对每个网格执行 FPS 采样 + 归一化
                data_list = []
                label_list = []
                for id_list in list2:
                    if len(id_list) < self.num_points:
                        continue

                    block = input_data[id_list]                        # (M,5) on CPU
                    block_t = torch.from_numpy(block).cuda()           # (M,5) -> GPU
                    xyz_all = torch.from_numpy(input_data[:, :3]).cuda()

                    # FPS 采样
                    xyz = block_t[:, :3].unsqueeze(0)                  # (1, M, 3)
                    fps_idx = farthest_point_sample(xyz, self.num_points)  # (1, num_points)
                    sampled = block_t[fps_idx.squeeze(0)][:,:5]        # (num_points,5)

                    # 归一化
                    coords = sampled[:, :3]
                    new_xyz = coordinate_normalize2(xyz_all, coords)  # (num_points,3)

                    # label
                    lbl = sampled[:, 4].unsqueeze(1)                   # (num_points,1)

                    data_list.append(new_xyz.unsqueeze(0))             # (1,num_points,3)
                    label_list.append(lbl.unsqueeze(0))                # (1,num_points,1)

                # 收集本文件的所有子块
                if data_list:
                    all_data.append(torch.cat(data_list, dim=0))       # (K,num_points,3)
                    # 从 (K,num_points,1) -> (K,num_points)
                    all_label.append(torch.cat(label_list, dim=0).squeeze(-1))

                # 进度
                self.progress.emit(int((i + 1) / total_files * 100))

            # 6. 保存
            if all_data:
                out_data = torch.cat(all_data, dim=0).cpu().numpy()
                out_label = torch.cat(all_label, dim=0).cpu().numpy()
                os.makedirs(self.output_dir, exist_ok=True)
                np.save(os.path.join(self.output_dir, 'test_data.npy'), out_data)
                np.save(os.path.join(self.output_dir, 'test_label.npy'), out_label)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
