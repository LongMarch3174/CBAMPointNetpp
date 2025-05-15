import torch
import numpy as np
from networkx import radius


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N, dtype=torch.float64).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    centroids[:, 0] = farthest
    return centroids


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def coordinate_normalize(xyz_all, input_xyz):
    """
    normalize the xyz coordinate value
    :param input_xyz: the original coordinate value , [N, 3]
    :param max_xyz: the max xyz coordinate value for whole data , [1, 3]
    :param min_xyz: the min xyz coordinate value for whole data , [1, 3]
    :return: output_xyz: normalized xyz coordinate value , [N, 3]
    """
    device = input_xyz.device
    dtype = input_xyz.dtype
    N, C = input_xyz.shape
    gap_matrix = torch.zeros((1, 3), dtype=dtype).to(device)
    max_xyz, _ = torch.max(xyz_all, 0)
    min_xyz, _ = torch.min(xyz_all, 0)
    gap_matrix[0, 0] = max_xyz[0] - min_xyz[0]
    gap_matrix[0, 1] = max_xyz[1] - min_xyz[1]
    gap_matrix[0, 2] = max_xyz[2] - min_xyz[2]
    min_matrix = min_xyz.unsqueeze(0).repeat(N, 1)
    gap_xyz = input_xyz - min_matrix
    output_xyz = torch.div(gap_xyz, gap_matrix)
    return output_xyz


def coordinate_normalize3(input_xyz):
    max_xyz = torch.max(input_xyz, dim=0)[0]
    min_xyz = torch.min(input_xyz, dim=0)[0]
    mean_xyz = (max_xyz + min_xyz)/2
    div = torch.max(input_xyz - mean_xyz, dim=0)[0]
    new_xyz = (input_xyz - mean_xyz) / torch.max(div, dim=0)[0]
    return new_xyz


def coordinate_normalize2(xyz_all, input_xyz):
    max_xyz = torch.max(xyz_all, dim=0)[0]
    min_xyz = torch.min(xyz_all, dim=0)[0]
    mean_xyz = (max_xyz + min_xyz)/2
    div = torch.max(xyz_all - mean_xyz, dim=0)[0]
    new_xyz = (input_xyz - mean_xyz) / torch.max(div, dim=0)[0]
    return new_xyz


if __name__ == "__main__":
    data = [[639345.24015625, 4878040.25005859, 302.38000023, 212, 180, 161, 90.000000],
            [639333.76017578, 4878036.19000000, 306.23999893, 70, 112, 161, 8.000000],
            [639414.28996094, 4878305.65997070, 314.23000061, 119, 112, 161, 31.000000],
            [639734.41984375, 4878430.35992188, 303.68999969, 70, 112, 161, 13.000000]]
    data = np.array(data)
    data = torch.from_numpy(data)
    xyz = data[:, 0:3]
    other = data[:, 3:7]
    print(xyz)
    print(other)
    max, _ = torch.max(data, 0)
    print(max[0:3])
    min, _ = torch.min(data, 0)
    print(min[0:3])
    out = coordinate_normalize(xyz, max[0:3], min[0:3])
    print(out)
    out = torch.cat((out, other), 1)
    print(out)