import os
import torch
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from torch.utils.data import DataLoader
from mydataset import MyDataset
from MsgSeg import PointNet2MSGSeg  # 确保导入路径正确

class SegmentWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_path: str, data_dir: str, output_dir: str, num_points: int = 4096, batch_size: int = 4, gpu_id: int = 0):
        super().__init__()
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_points = num_points
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            model = PointNet2MSGSeg(12)
            model.load_state_dict(torch.load(self.model_path, map_location=f"cuda:{self.gpu_id}"))
            model.cuda(self.gpu_id)
            model.eval()

            test_dataset = MyDataset(self.num_points, self.data_dir)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

            total_batches = len(test_loader)
            all_outputs = []

            for i, (data, _) in enumerate(test_loader):
                if not self._is_running:
                    break
                data = data.cuda(self.gpu_id)
                with torch.no_grad():
                    out, _ = model(data)
                    preds = out.max(dim=1)[1].cpu().numpy()

                points = data[:, :, :3].cpu().numpy().reshape(-1, 3)
                labels = preds.reshape(-1, 1)
                result = np.hstack((points, labels))
                all_outputs.append(result)

                self.progress.emit(int((i + 1) / total_batches * 100))

            final_output = np.vstack(all_outputs)
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, "segmentation_result.txt")
            np.savetxt(save_path, final_output, fmt="%.6f", delimiter=" ")

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
