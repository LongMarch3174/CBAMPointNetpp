import torch
import torch.nn as nn
import utils
import torch.nn.functional as F


class PointNetSetAbstractionMsg(nn.Module):
    r"""Input:
    npoint: Number of point for FPS sampling
    radius: Radius for ball query
    nsample: Number of point for each ball query
    mlp: A list for mlp input-output channel, such as [64, 64, 128]
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            channel = in_channel + 3
            for out_channel in mlp[i]:
                convs.append(nn.Conv2d(channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        # 获取最终输出的规格 例如[32, 32, 64]就获取后面的64
        #self.last = mlp[len(mlp) - 1]
        #self.cbam = CBAMLayer3D(num_channels=self.last)  # mlp [256, 512, 1024]


    def forward(self, xyz, points=None):
        r"""Input:
               xyz : torch.Tensor
                   (B, N, C) tensor of the xyz coordinates of the features
               points : torch.Tensor
                   (B, N, D) tensor of the descriptors of the the features
               Returns:
               new_xyz : torch.Tensor
                   (B, npoint, 3) sampled points
               new_points : torch.Tensor
                   (B, C' , npoint) sample points feature data
               """
        new_points_list = []
        for i in range(len(self.radius)):
            radius = self.radius[i]
            nsample = self.nsample[i]
            if self.npoint is not None:
                new_xyz, new_points = utils.sample_and_group(radius, self.npoint, nsample, xyz, points)
                grouped_points = new_points
            else:
                new_xyz, new_points = utils.sample_and_group_all(xyz, points)
                grouped_points = new_points
            grouped_points = grouped_points.permute(0, 3, 2, 1)   # [B, D, nsample, npoint]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]   # [B, npoint, D]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)

        #print("new_points:",new_points.shape)
        #new_points = self.cbam(new_points)

        return new_xyz, new_points_concat


class CBAMLayer3D(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super(CBAMLayer3D, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 添加最大池化层

        # 定义两个不同的MLP来处理最大池化和均值池化的结果
        self.mlp_avg = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // reduction, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // reduction, num_channels, 3,1,1)
        )
        self.mlp_max = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // reduction, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // reduction, num_channels, 3,1,1)
        )

        # Spatial Attention
        self.conv = nn.Conv1d(num_channels * 2, 1, kernel_size=3, padding=1)
        # self.conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.mlp_avg(self.avg_pool(x))
        max_out = self.mlp_max(self.max_pool(x))
        channel_out = torch.sigmoid(avg_out + max_out)  # 组合最大池化和均值池化结果
        # print("channel_out size:", channel_out.size())
        # print("x size:", x.size())
        x = channel_out * x

        # Spatial Attention
        max_out = torch.max(x, dim=2, keepdim=True)[0]
        avg_out = torch.mean(x, dim=2, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x
