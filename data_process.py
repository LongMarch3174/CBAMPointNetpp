import math
import os
import numpy as np
import torch
from proceed_utils import farthest_point_sample
from proceed_utils import coordinate_normalize, coordinate_normalize2

path = r'F:\guoqin\PointNet++new\train'  # D:\pointnet\experiment\test
list1 = []
pathDir = os.listdir(path)
all_data = []
all_label = []
for s in pathDir:
    newDir = os.path.join(path, s)
    print(newDir)
    with open(newDir, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            read_data = [float(x) for x in eachline[0:5]]
            list1.append(read_data)
            line = f.readline()
        input_data = np.array(list1)
    print(input_data.shape)
    list1.clear()
    Xmax = np.amax(input_data, axis=0)[0]
    Xmin = np.amin(input_data, axis=0)[0]
    Ymax = np.amax(input_data, axis=0)[1]
    Ymin = np.amin(input_data, axis=0)[1]

    xyz_all = input_data[:, 0:3]
    xyz_all = torch.from_numpy(xyz_all).cuda()
    # 这里修改格网数量
    a = math.ceil((Xmax - Xmin) / 100)
    b = math.ceil((Ymax - Ymin) / 100)
    allblock = a * b
    print(allblock)
    list2 = list(map(lambda x: [], range(allblock)))
    for i, k in enumerate(input_data):
        for m in range(b):
            for n in range(a):
                if m <= (k[1] - Ymin) / 100 <= m + 1 and n <= (k[0] - Xmin) / 100 <= n + 1:
                    list2[m * a + n].append(i)
    data_list = []
    label_list = []
    for j in range(allblock):
        id_list = list2[j]
        num_point = 4096
        if len(id_list) < num_point:
            continue
        else:
            index = np.array(id_list)
            block = input_data[index]
            block = torch.from_numpy(block).cuda()
            xyz = block[:, 0:3]
            xyz = xyz.unsqueeze(0)
            fps_idx = farthest_point_sample(xyz, num_point)
            data = block[fps_idx.squeeze(0)][:, 0:5]
            xyz = data[:, 0:3]
            new_xyz = coordinate_normalize2(xyz_all, xyz)   # -1~1
            label = block[fps_idx.squeeze(0)][:, 4:5]
            data_list.append(new_xyz.unsqueeze(0))
            label_list.append(label.unsqueeze(0))
    All_data = torch.cat(data_list, dim=0)
    All_label = torch.cat(label_list, dim=0).squeeze(-1)
    print(All_data.shape)
    print(All_label.shape)
    all_data.append(All_data)
    all_label.append(All_label)
if os.path.basename(path) == 'train':
    test_data = torch.cat(all_data, dim=0)
    test_label = torch.cat(all_label, dim=0)
    print('test_data', test_data.shape)
    print('test_label', test_label.shape)
    np.save('test_data', test_data.cpu().numpy())
    np.save('test_label', test_label.cpu().numpy())
# elif os.path.basename(path) == 'train':
#     train_data = torch.cat(all_data, dim=0)
#     train_label = torch.cat(all_label, dim=0)
#     print('train_data', train_data.shape)
#     print('train_label', train_label.shape)nishi
#     np.save('train_data', train_data.cpu().numpy())
#     np.save('train_label', train_label.cpu().numpy())
