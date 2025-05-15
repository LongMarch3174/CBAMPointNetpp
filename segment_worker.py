import os
import torch
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from torch.utils.data import Dataset, DataLoader
from MsgSeg import PointNet2MSGSeg

class NpyDataset(Dataset):
    """Inference dataset loading a single *_data.npy file."""
    def __init__(self, data_file: str):
        self.data = np.load(data_file)  # shape = (K, num_points, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return (num_points, 3) array
        return torch.from_numpy(self.data[idx]).float()

class SegmentWorker(QObject):
    progress = pyqtSignal(int)       # 0–100%
    finished = pyqtSignal(str)       # emit save_path when done
    error = pyqtSignal(str)

    def __init__(
        self,
        model_path: str,
        data_file: str,
        label_file: str,
        output_dir: str,
        batch_size: int = 4,
        gpu_id: int = 0
    ):
        super().__init__()
        self.model_path = model_path
        self.data_file = data_file
        self.label_file = label_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # 1. 校验文件配对逻辑
            base_data = os.path.splitext(os.path.basename(self.data_file))[0]
            base_label = os.path.splitext(os.path.basename(self.label_file))[0]
            if not base_data.endswith("_data") or not base_label.endswith("_label"):
                self.error.emit("文件名必须分别以 '_data.npy' 和 '_label.npy' 结尾")
                return
            if base_data[:-5] != base_label[:-6]:
                self.error.emit("数据文件和标签文件前缀不匹配")
                return

            # 2. 加载模型
            net = PointNet2MSGSeg(12)
            net.load_state_dict(
                torch.load(self.model_path, map_location=f"cuda:{self.gpu_id}")
            )
            net.cuda(self.gpu_id)
            net.eval()

            # 3. 准备数据集 & DataLoader
            dataset = NpyDataset(self.data_file)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )
            total_batches = len(loader)
            all_results = []

            # 4. 推理循环
            for i, data in enumerate(loader):
                if not self._is_running:
                    return
                data = data.cuda(self.gpu_id)                    # (B, N, 3)
                with torch.no_grad():
                    out, _ = net(data)                            # (B, C, N)
                    preds = out.max(dim=1)[1].cpu().numpy()       # (B, N)

                coords = data.cpu().numpy().reshape(-1, 3)        # (B*N, 3)
                labels = preds.reshape(-1, 1)                     # (B*N, 1)
                batch_res = np.hstack((coords, labels))          # (B*N, 4)
                all_results.append(batch_res)

                self.progress.emit(int((i + 1) / total_batches * 100))

            # 5. 保存结果
            result = np.vstack(all_results)  # (K*N, 4)
            os.makedirs(self.output_dir, exist_ok=True)
            save_name = f"{base_data}_seg.txt"
            save_path = os.path.join(self.output_dir, save_name)
            np.savetxt(save_path, result, fmt="%.6f", delimiter=" ")

            self.finished.emit(save_path)

        except Exception as e:
            self.error.emit(str(e))
