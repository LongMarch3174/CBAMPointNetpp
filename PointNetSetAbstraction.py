import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        '''
    Input:
    npoint: Number of point for FPS sampling
    radius: Radius for ball query
    nsample: Number of point for each ball query
    in_channel: the dimention of channel
    mlp: A list for mlp input-output channel, such as [64, 64, 128]
    group_all: bool type for group_all or not
        '''
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        # print('xyz',xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = utils.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = utils.sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        # 利用1x1的2d的卷积相当于把每个group当成一个通道，共npoint个通道，对[C+D, nsample]的维度上做逐像素的卷积，结果相当于对单个C+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # 对每个group做一个max pooling得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
