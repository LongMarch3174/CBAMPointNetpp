from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
#
classes = ['bridge', 'building', 'water', 'tree', '	veg','low_veg','light','electric','ground','others','vehicle'
           'non vehicle']
class2label = {cls: i for i, cls in enumerate(classes)}


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data

        return pc


class PointcloudRotatebyAngle(object):
    def __init__(self, rotation_angle=0.0):
        self.rotation_angle = rotation_angle

    def __call__(self, pc):
        normals = pc.size(2) > 3
        bsize = pc.size()[0]
        for i in range(bsize):
            cosval = np.cos(self.rotation_angle)
            sinval = np.sin(self.rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            rotation_matrix = torch.from_numpy(rotation_matrix).float().cuda()

            cur_pc = pc[i, :, :]
            if not normals:
                cur_pc = cur_pc @ rotation_matrix
            else:
                pc_xyz = cur_pc[:, 0:3]
                pc_normals = cur_pc[:, 3:]
                cur_pc[:, 0:3] = pc_xyz @ rotation_matrix
                cur_pc[:, 3:] = pc_normals @ rotation_matrix

            pc[i, :, :] = cur_pc

        return pc


class MyDataset(Dataset):
    def __init__(self, num_points, path, train=True):
        self.train = train
        self.path = path
        self.num_points = num_points
        if self.train:
            self.data = np.load(self.path + 'train_data.npy')
            self.label = np.load(self.path + 'train_label.npy')

        else:
            self.data = np.load(self.path + 'test_data.npy')
            self.label = np.load(self.path + 'test_label.npy')
        self.label = np.where(self.label == 200000, 0, self.label)
        self.label = np.where(self.label == 200101, 1, self.label)
        self.label = np.where(self.label == 200200, 2, self.label)
        self.label = np.where(self.label == 200201, 2, self.label)
        self.label = np.where(self.label == 200301, 3, self.label)
        self.label = np.where(self.label == 200400, 4, self.label)
        self.label = np.where(self.label == 200500, 5, self.label)
        self.label = np.where(self.label == 200601, 6, self.label)
        self.label = np.where(self.label == 200700, 7, self.label)
        self.label = np.where(self.label == 200800, 8, self.label)
        self.label = np.where(self.label == 200900, 9, self.label)
        self.label = np.where(self.label == 100500, 10, self.label)
        self.label = np.where(self.label == 100600, 11, self.label)
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        current_points = self.data[index, pt_idxs]
        current_labels = torch.from_numpy(self.label[index, pt_idxs].copy()).type(
            torch.LongTensor
        )

        current_points = torch.from_numpy(current_points).type(
            torch.FloatTensor
        )
        return current_points, current_labels


if __name__ == '__main__':
    path = 'E:\数据集\Torch_data_muti\\block_train\\test_data.npy'
    path_1 = 'E:\数据集\Torch_data_muti\\block_train\\test_label.npy'
    data = np.load(path)
    print(data.shape)
    # dataset = MyDataset(4096, path, train=True)
    # for point, label in dataset:
    #     print(point.size(1))
    # loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=3)
    # for point, label in loader:
    #     print(point)
