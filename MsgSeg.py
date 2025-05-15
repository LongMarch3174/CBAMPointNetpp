import torch.nn as nn
import PointNetSetAbstractionMsgPyCBAM as SAMSG
import PointNetFeaturePropagation as FP
import torch.nn.functional as F


class PointNet2MSGSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2MSGSeg, self).__init__()
        self.sa1 = SAMSG.PointNetSetAbstractionMsg(
            npoint=1024, radius=[0.05, 0.1],
            nsample=[16, 32], in_channel=0, mlp=[[6, 16, 16, 32], [6, 32, 32, 64]])    # 0+3 to
        self.sa2 = SAMSG.PointNetSetAbstractionMsg(
            npoint=256, radius=[0.1, 0.2],
            nsample=[16, 32], in_channel=96, mlp=[[99, 64, 64, 128], [99, 64, 96, 128]])
        self.sa3 = SAMSG.PointNetSetAbstractionMsg(
            npoint=64, radius=[0.2, 0.4],
            nsample=[16, 32], in_channel=256, mlp=[[259, 128, 196, 256], [259, 128, 196, 256]])
        self.sa4 = SAMSG.PointNetSetAbstractionMsg(
            npoint=16, radius=[0.4, 0.8],
            nsample=[16, 32], in_channel=512, mlp=[[515, 256, 256, 512], [515, 256, 384, 512]])
        self.fp3 = FP.PointNetFeaturePropagation(in_channel=256, mlp=[128, 128])    # 256+3 to
        self.fp2 = FP.PointNetFeaturePropagation(in_channel=512 + 96, mlp=[256, 256])
        self.fp1 = FP.PointNetFeaturePropagation(in_channel=512 + 256, mlp=[512, 512])
        self.fp0 = FP.PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        # Set Abstraction layers
        xyz, features = self._break_up_pc(data)
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # Feature Propagation layers
        l3_points = self.fp0(l3_xyz.transpose(2, 1), l4_xyz.transpose(2, 1), l3_points, l4_points)
        l2_points = self.fp1(l2_xyz.transpose(2, 1), l3_xyz.transpose(2, 1), l2_points, l3_points)
        l1_points = self.fp2(l1_xyz.transpose(2, 1), l2_xyz.transpose(2, 1), l1_points, l2_points)
        l0_points = self.fp3(xyz.transpose(2, 1), l1_xyz.transpose(2, 1), features, l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        return x, feat


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim
    import pointnet.data.Indoor3DSemSegLoader as Indoor3DSemSegLoader
    from torch.utils.data import DataLoader

    train_set = Indoor3DSemSegLoader.Indoor3DSemSeg(16)
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
    )

    model = PointNet2MSGSeg(num_classes=6)  # 13 to 6
    model.cuda()
