import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample

        # 多尺度 MLP 卷积块
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks   = nn.ModuleList()
        for out_list in mlp:
            convs, bns = nn.ModuleList(), nn.ModuleList()
            c_in = in_channel + 3
            for c_out in out_list:
                convs.append(nn.Conv2d(c_in, c_out, 1))
                bns.append(nn.BatchNorm2d(c_out))
                c_in = c_out
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        # 计算拼接后特征维度
        out_channels = sum(branch[-1] for branch in mlp)
        # 实例化标准 CBAM
        self.cbam = PyramidCBAMLayer3D(num_channels=out_channels, reduction=16)

    def forward(self, xyz, points=None):
        new_points_list = []
        for i, r in enumerate(self.radius):
            if self.npoint is not None:
                new_xyz, grouped = utils.sample_and_group(r, self.npoint,
                                                         self.nsample[i],
                                                         xyz, points)
            else:
                new_xyz, grouped = utils.sample_and_group_all(xyz, points)
            # grouped: [B, D+3, nsample, npoint]
            grouped = grouped.permute(0, 3, 2, 1)  # → [B, npoint, nsample, D+3]
            for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                grouped = F.relu(bn(conv(grouped)))
            # → [B, out_c, npoint]
            new_points = torch.max(grouped, dim=2)[0]
            new_points_list.append(new_points)

        # 拼接多尺度输出 -> [B, sum(out_c), npoint]
        new_points_concat = torch.cat(new_points_list, dim=1)

        # 调用标准 CBAM 注意力
        new_points_concat = self.cbam(new_points_concat)

        return new_xyz, new_points_concat
    

class PyramidCBAMLayer3D(nn.Module):
    def __init__(self, num_channels, reduction=16, mlp_kernel_sizes=(1,3,5)):
        super(PyramidCBAMLayer3D, self).__init__()

        # === 池化操作 ===
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # === 金字塔 MLP 分支定义 ===
        # 每个分支都是 Conv1d→ReLU→Conv1d，但 kernel_size 不同
        def make_mlp_branch(k):
            return nn.Sequential(
                nn.Conv1d(num_channels, num_channels//reduction,
                          kernel_size=k, padding=k//2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(num_channels//reduction, num_channels,
                          kernel_size=k, padding=k//2, bias=False)
            )

        # 为 avg, max, min 池化各自创建一个 ModuleList 分支组
        self.mlp_avg = nn.ModuleList([make_mlp_branch(k) for k in mlp_kernel_sizes])
        self.mlp_max = nn.ModuleList([make_mlp_branch(k) for k in mlp_kernel_sizes])

        # 最后汇总各尺度分支的 sigmoid
        self.sigmoid = nn.Sigmoid()

        # 空间注意力依旧用单一路径（或同样也可做金字塔）
        self.conv_spatial = nn.Conv1d(3, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x: [B, C, N]
        # —— 通道注意力 —— 
        # 全局池化
        a = self.avg_pool(x)   # [B, C, 1]
        m = self.max_pool(x)   # [B, C, 1]

        # 对每个尺度分支做前向，然后相加
        def pyramid_mlp(pool, mlp_branches):
            out = 0
            for branch in mlp_branches:
                out = out + branch(pool)    # [B, C, 1]
            return out

        avg_out = pyramid_mlp(a, self.mlp_avg)
        max_out = pyramid_mlp(m, self.mlp_max)

        # 合并三种池化 + Sigmoid 得到通道注意力权重
        channel_att = self.sigmoid(avg_out + max_out)  # [B, C, 1]
        x_ca = x * channel_att

        # —— 空间注意力 ——
        # 在通道维度上做统计 (此处也可扩展成多尺度)
        max_c, _ = torch.max(x_ca, dim=1, keepdim=True)  # [B,1,N]
        avg_c    = torch.mean(x_ca, dim=1, keepdim=True)  # [B,1,N]
        min_c, _ = torch.min(x_ca, dim=1, keepdim=True)  # [B,1,N]
        spa_feat = torch.cat([max_c, avg_c, min_c], dim=1)  # [B,3,N]

        spatial_att = self.sigmoid(self.conv_spatial(spa_feat))  # [B,1,N]
        x_sa = x_ca * spatial_att

        return x_sa
