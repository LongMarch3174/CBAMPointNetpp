import torch
import torch.nn as nn
import os
import argparse
import datetime

from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append('../data')
sys.path.append('../Model')
from MsgSeg import PointNet2MSGSeg
from mydataset import MyDataset
# file_path = 'root/PointNet++/data/mydataset.py'
# from file_path import MyDataset
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, cohen_kappa_score
import logging  # 引入logging模块
import visdom
import torch.nn.functional as F

# 获取当前时间
now = datetime.datetime.now()
# 格式化时间字符串为 "月-日-时-分"
log_filename = now.strftime("%m-%d-%H-%M") + ".log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='/root/PointNet++/train/Log/'+ log_filename,  # 指定日志文件的路径
                    filemode='a')# 'a' 表示追加模式，'w' 表示写模式（覆盖）

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for seg training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument('--gpu', type=int, default=0, help='specify gpu device')
    parser.add_argument("-num_points", type=int, default=4096, help="Number of points to train with")
    parser.add_argument("-num_epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("-weight_reg", type=float, default=1e-5, help="L2 regularization coeff")
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training [default: 0.001 for Adam, 0.01 for SGD]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay for Adam')
    # parser.add_argument('--optimizer', type=str, default='SGD', help='type of optimizer')
    parser.add_argument('--optimizer', type=str, default='NAG', help='type of optimizer')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument("-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum")
    parser.add_argument("-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma")
    parser.add_argument("-checkpoint", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument("-test_freq", type=int, default=1)
    return parser.parse_args()


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(12)  # 6 to 9
    U_all = np.zeros(12)  # 6 to 9
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(12):  # 6 to 9
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def calculate_metrics_per_class(confusion_mat, y_true, y_pred):
    precision_per_class = precision_score(y_true, y_pred, labels=range(12), average=None)
    recall_per_class = recall_score(y_true, y_pred, labels=range(12), average=None)

    IoU_per_class = np.diag(confusion_mat) / (
            np.sum(confusion_mat, axis=1) + np.sum(confusion_mat, axis=0) - np.diag(confusion_mat))
    mIoU = np.nanmean(IoU_per_class)

    return precision_per_class, recall_per_class, IoU_per_class, mIoU



def train():
    args = parse_args()

    logging.info('Loading Dataset...')

    path =r'/root/PointNet++/train'  # 修改npy文件的路径
    train_dataset = MyDataset(args.num_points, path)
    test_dataset = MyDataset(4096, path, train=False)
    logging.info('train_dataset: {}'.format(len(train_dataset)))
    logging.info('test_dataset: {}'.format(len(test_dataset)))
    logging.info('Done...\n')

    logging.info('Creating DataLoader...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    logging.info('Done...\n')

    logging.info('Creating Model...')
    model = PointNet2MSGSeg(12)  # 6 to 9
    model = model.cuda(args.gpu)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), path + '/model_weights/origin', '4096_999.pth')), strict=False)  # 修改保存名称
    # optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'NAG':
        # optimizer = torch.optim.SGDM(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)

    # 学习率衰减
    # schedular = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    def adjust_learning_rate(optimizer, step):
        """Sets the learning rate to the initial LR decayed by 30 every 20000 steps"""
        lr = args.learning_rate * (0.3 ** (step // 20000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 加载预训练模型
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))
    # CrossEntropy Loss
    criterion = nn.CrossEntropyLoss()
    logging.info('Done...\n')

    logging.info('Start training...')
    acc_container = torch.zeros((args.num_epochs, 2))
    step = 0
    best_test_iou = 0
    best_test_acc = 0
    # viz = visdom.Visdom(env='3_PointNet++_ISPRS_4096_cel_unlessY')  # 修改名称
    for epoch in range(args.num_epochs):
        logging.info("--------Epoch {}--------".format(epoch))

        tqdm_batch = tqdm(train_loader, desc='Epoch-{} training'.format(epoch))
        # train
        model.train()
        train_loss = 0.0
        count = 0.0
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        for batch_idx, (data, label) in enumerate(tqdm_batch):
            data, label = data.cuda(args.gpu), label.cuda(args.gpu)
            out, _ = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step = step + 1
            adjust_learning_rate(optimizer, step)
            batch_size = data.size()[0]
            count += batch_size
            train_loss += loss.item() * batch_size
            preds = out.max(dim=1)[1]
            label = label.cpu().numpy()
            preds = preds.detach().cpu().numpy()
            train_true_cls.append(label.reshape(-1))
            train_pred_cls.append(preds.reshape(-1))
            train_true_seg.append(label)
            train_pred_seg.append(preds)
        tqdm_batch.close()
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        # avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        train_loss = train_loss * 1.0 / count
        logging.info('Train Loss: {:.4f} Train acc:{:.4f}  Train mIOU:{:.4f}'.format(
            train_loss,
            train_acc,
            np.mean(train_ious)))

        if epoch % args.test_freq == 0:
            tqdm_batch = tqdm(test_loader, desc='Epoch-{} testing'.format(epoch))
            # test
            model.eval()
            test_loss = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for batch_idx, (data, label) in enumerate(tqdm_batch):
                data, label = data.cuda(args.gpu), label.cuda(args.gpu)
                out, _ = model(data)
                loss = criterion(out, label)
                batch_size = data.size()[0]
                preds = out.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                label = label.cpu().numpy()
                preds = preds.detach().cpu().numpy()
                test_true_cls.append(label.reshape(-1))
                test_pred_cls.append(preds.reshape(-1))
                test_true_seg.append(label)
                test_pred_seg.append(preds)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            # avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            test_loss = test_loss * 1.0 / count
            '''detailed_classification_report = metrics.classification_report(test_true_cls, test_pred_cls, digits=4,
                                                                           output_dict=False)
            logging.info('\nDetailed Classification Report:\n{}'.format(detailed_classification_report))'''

        acc_container[epoch, 0] = test_acc
        # acc_container[epoch, 1] = avg_per_class_acc
        logging.info(
            'Test Loss: {:.4f} Test acc:{:.4f}  Test mIOU:{:.4f}'.format(test_loss,
                                                                         test_acc,
                                                                         np.mean(test_ious)))
        print("\nMax Test accuracy is:" + str(torch.max(acc_container[:, 0]).data.cpu().numpy()))
        print("Max Test average accuracy is:" + str(torch.max(acc_container[:, 1]).data.cpu().numpy()))
        tqdm_batch.close()
        # 如果精度高于阈值且为当前最高精度，则保存模型
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            Path = os.path.join(os.getcwd(), path+'/weights', '4096_{}.pth'.format(epoch))
            torch.save(model.state_dict(), Path)
            logging.info('model saved in {}'.format(Path))
            a = 1
        elif test_acc >= best_test_acc:
            test_acc = best_test_acc
            if a == 1:
                a = 0
            else:
                Path = os.path.join(os.getcwd(), path+'/weights', '4096_{}.pth'.format(epoch))
                torch.save(model.state_dict(), Path)
                logging.info('model saved in {}'.format(Path))
                a = 0
    logging.info('Done...\n')


def test():
    # for i in range(50):
        args = parse_args()
        path = r'/root/PointNet++/train'  # 修改路径
        test_loader = torch.utils.data.DataLoader(MyDataset(4096, path, train=False), batch_size=4,
                                                  shuffle=True, drop_last=False)  # 4096 to 512
        # Try to load models
        model = PointNet2MSGSeg(12).cuda(args.gpu)  # 6 to 9

        model.load_state_dict(torch.load(os.path.join(os.getcwd(), path + '/weights', '4096_999.pth')))  # 修改保存名称
        model = model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        points_list = []
        pred_list = []
        for data, label in test_loader:
            data, label = data.cuda(args.gpu), label.cuda(args.gpu)
            out, _ = model(data)
            batch_size = data.size()[0]
            preds = out.max(dim=1)[1]
            label = label.cpu().numpy()
            preds = preds.detach().cpu().numpy()
            test_true_cls.append(label.reshape(-1))
            test_pred_cls.append(preds.reshape(-1))
            test_true_seg.append(label)
            test_pred_seg.append(preds)

            pred = F.log_softmax(out, dim=1).permute(0, 2, 1)

            for j in range(pred.size(0)):
                batch_pred = pred[j]
                batch_target = label[j]
                batch_choice = batch_pred.data.max(1)[1]
                pred_list.append(batch_choice.unsqueeze(0).reshape(-1, 1))
            a = torch.cat(pred_list, dim=0).squeeze(0).cpu().data.numpy()
            points_list.append(data[:, :, 0:3].reshape(-1, 3))

        all_points = torch.cat(points_list, dim=0).cpu().data.numpy()
        out = np.hstack((all_points, a))

        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        f1_score = metrics.f1_score(test_true_cls, test_pred_cls, average='micro')
        kappa = metrics.cohen_kappa_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)

        # 在测试循环结束后，计算混淆矩阵和Kappa系数
        y_true = test_true_cls
        y_pred = test_pred_cls
        confusion_mat = confusion_matrix(y_true, y_pred, labels=range(12))

        precision_per_class, recall_per_class, IoU_per_class, mIoU = calculate_metrics_per_class(confusion_mat, y_true, y_pred)
        # kappa = cohen_kappa_score(y_true, y_pred)

        for cls_index, iou in enumerate(IoU_per_class):
            logging.info('Class {}: IoU: {:.4f}'.format(cls_index, iou))
        np.savetxt("/root/PointNet++/test/area_cbam.txt", out, fmt='%f', delimiter=' ')  # 修改保存名称
        logging.info('Test acc:{:.4f} Test avg acc:{:.4f} Test mIOU:{:.4f} F1_score:{:.4f} Kappa:{:.4f}'
                     .format(test_acc, avg_per_class_acc, np.mean(test_ious), f1_score, kappa))

        for cls in range(12):
            logging.info(
                f"Class {cls}: Precision: {precision_per_class[cls]}, Recall: {recall_per_class[cls]}, IoU: {IoU_per_class[cls]}")
        logging.info(f"Mean IoU: {mIoU}, Kappa: {kappa}")
    

if __name__ == '__main__':
    train()
    # test()