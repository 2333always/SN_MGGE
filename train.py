import os
import sys

from model_seg import knn
from util import *
import argparse
import torch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from manager import IouTable, get_miou, get_macc
from ShapeNetPart import ShapeNetDataset, get_valid_labels
from importlib import import_module
from visualize import visualize
import torch.nn as nn

from scipy.spatial import cKDTree  # Import KD-tree from scipy
import torch

TRAIN_NAME = __file__.split('.')[0]

# 设定三元组损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative)
        losses = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
triplet_criterion = TripletLoss()

class PartSegConfig():
    ####################
    # Dataset parameters
    ####################

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [False, False, False]
    normal_scale = True
    augment_shift = None
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.25
    augment_noise = 0.002
    augment_noise_clip = 0.05
    augment_occlusion = 'none'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='model', metavar='N',
                        help='Model to use')
    parser.add_argument('--gpu_idx', type=int, default=[0, 1], nargs='+',
                        help='set < 0 to use CPU')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--Tmax', type=int, default=100, metavar='N',
                        help='Max iteration number of scheduler. ')
    parser.add_argument('--mode', default='train', help='[train/test]')
    parser.add_argument('--epoch', type=int, default=200, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset', type=str, default='data/',
                        help="Path to dataset")
    parser.add_argument('--load', help='Path to load model')
    parser.add_argument('--record', type=str, default='record.log', help='Record file name (e.g. record.log)')
    parser.add_argument('--interval', type=int, default=100, help='Record interval within an epoch')
    parser.add_argument('--checkpoint_gap', type=int, default=10, help='Save checkpoints every n epochs')
    parser.add_argument('--point', type=int, default=2048, help='Point number per object')
    parser.add_argument('--output', help='Folder for visualization images')
    args = parser.parse_args()

    if args.name == '':
        args.name = TRAIN_NAME

    config = PartSegConfig()

    # Create Network
    MODEL = import_module(args.model)
    model = MODEL.Net(args=args, class_num=50, cat_num=16)
    manager = Manager(model, args)

    ################
    # Start Training
    ################

    if args.mode == "train":
        print("Training ...")
        train_data = ShapeNetDataset(root=args.dataset, config=config, num_points=args.point, split='trainval')
        train_loader = DataLoader(train_data, shuffle=True, batch_size=args.bs, drop_last=True)
        test_data = ShapeNetDataset(root=args.dataset, config=config, num_points=args.point, split='test')
        test_loader = DataLoader(test_data, shuffle=False, batch_size=args.bs, drop_last=False)
        manager.train(train_loader, test_loader)

    elif args.mode == "test":
        print("Testing ...")
        test_data = ShapeNetDataset(root=args.dataset, config=config, num_points=args.point, split='test')
        test_loader = DataLoader(test_data, shuffle=False, batch_size=args.bs, drop_last=False)

        test_loss, test_table_str = manager.test(test_loader, args.output)
        print(test_table_str)


class Manager():
    def __init__(self, model, args):

        ############
        # Parameters
        ############

        self.args_info = args.__str__()
        self.device = torch.device('cpu' if len(args.gpu_idx) == 0 else 'cuda:{}'.format(args.gpu_idx[0]))
        self.model = model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=args.gpu_idx)
        print('Now use {} GPUs: {}'.format(len(args.gpu_idx), args.gpu_idx))
        if args.load:
            self.model.load_state_dict(torch.load(args.load))

        self.epoch = args.epoch
        self.Tmax = args.Tmax
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.Tmax, eta_min=args.lr)
        self.loss_function = nn.CrossEntropyLoss()

        self.save = os.path.join('models', args.name, 'checkpoints')
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        self.record_interval = args.interval
        self.record_file = None
        if args.record:
            self.record_file = open(os.path.join('models', args.name, args.record), 'w')
        self.checkpoint_gap = args.checkpoint_gap
        self.out_dir = args.output

    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write(info + '\n')
            self.record_file.flush()





    def calculate_save_mious(self, iou_table, category_names, labels, predictions):
        for i in range(len(category_names)):
            category = category_names[i]
            pred = predictions[i]
            label = labels[i]
            valid_labels = get_valid_labels(category)
            miou = get_miou(pred, label, valid_labels)
            macc = get_macc(pred, label, valid_labels)
            iou_table.add_obj_miou(category, miou)
            iou_table.add_obj_macc(category, macc)

    def save_visualizations(self, dir, category_names, object_ids, points, labels, predictions):
        for i in range(len(category_names)):
            cat = category_names[i]
            valid_labels = get_valid_labels(cat)
            shift = min(valid_labels) * (-1)
            obj_id = object_ids[i]
            point = points[i].to("cpu")
            label = labels[i].to("cpu") + shift
            pred = predictions[i].to("cpu") + shift

            cat_dir = os.path.join(dir, cat)
            if not os.path.isdir(cat_dir):
                os.mkdir(cat_dir)
            gt_fig_name = os.path.join(cat_dir, "{}_gt.png".format(obj_id))
            pred_fig_name = os.path.join(cat_dir, "{}_pred.png".format(obj_id))
            visualize(point, label, gt_fig_name)
            visualize(point, pred, pred_fig_name)

    def train(self, train_data, test_data):
        self.record("*****************************************")
        self.record("Hyper-parameters: {}".format(self.args_info))
        self.record("Model parameter number: {}".format(parameter_number(self.model)))
        self.record("Model structure: \n{}".format(self.model.__str__()))
        self.record("*****************************************")

        def get_triplet_features(x, k=20, idx=None, dim6=False):

            num_epochs = 10
            for epoch in range(num_epochs):
                batch_size = x.size(0)
                num_points = x.size(2)
                x = x.view(batch_size, -1, num_points)

                if idx is None:
                    if dim6 == False:
                        idx = knn(x, k=k)  # (batch_size, num_points, k)
                    else:
                        idx = knn(x[:, 0:3], k=k)
                    device = x.device
                    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
                    idx = idx + idx_base
                    idx = idx.view(-1)

                _, num_dims, _ = x.size()

                x = x.transpose(2, 1).contiguous()
                feature = x.view(batch_size * num_points, -1)[idx, :]
                feature = feature.view(batch_size, num_points, k, num_dims)
                x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

                norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
                x_tanh_norm = torch.tanh(norm_x) / norm_x
                x_anchor = x_tanh_norm * x  # 锚点

                norm_f = torch.norm(feature, p=2, dim=1, keepdim=True)
                f_tanh_norm = torch.tanh(norm_f) / norm_f
                feature_positive = f_tanh_norm * feature  # 正样本
                # 生成负样本，这里简单地随机选择，实际中应该使用更复杂的方式
                negative_idx = torch.randperm(x_anchor.size(0))
                negative = feature_positive[negative_idx]
                # 计算损失
                triploss = triplet_criterion(x_anchor, feature_positive, negative)
                loss = triploss

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



        for epoch in range(self.epoch):
            self.model.train()
            train_loss = 0
            train_iou_table = IouTable()
            learning_rate = self.optimizer.param_groups[0]['lr']
            for i, (cat_name, obj_ids, points, labels, mask, onehot) in enumerate(train_data):
                points = points.to(self.device)
                labels = labels.to(self.device)
                onehot = onehot.to(self.device)
                out = self.model(points, onehot)

                self.optimizer.zero_grad()
                loss = self.loss_function(out.reshape(-1, out.size(-1)), labels.view(-1, ))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                out[mask == 0] = out.min()
                pred = torch.max(out, 2)[1]
                self.calculate_save_mious(train_iou_table, cat_name, labels, pred)

                # record within epoch
                if self.record_interval and ((i + 1) % self.record_interval == 0):
                    c_miou = train_iou_table.get_mean_category_miou()
                    i_miou = train_iou_table.get_mean_instance_miou()
                    self.record(
                        ' epoch {:3} step {:5} | avg loss: {:.3f} | miou(c): {:.3f} | miou(i): {:.3f}'.format(epoch + 1,  i + 1,
                                                                                                              train_loss / (
                                                                                                                          i + 1),
                                                                                                              c_miou,
                                                                                                              i_miou))

            train_loss /= (i + 1)
            train_table_str = train_iou_table.get_string()
            test_loss, test_table_str = self.test(test_data, self.out_dir)
            if epoch < self.Tmax:
                self.lr_scheduler.step()
            elif epoch == self.Tmax:
                for group in self.optimizer.param_groups:
                    group['lr'] = 0.0001

            # save checkpoints
            if self.save:
                torch.save(self.model.state_dict(), os.path.join(self.save, 'model.pkl'))
                # Save checkpoints occasionally
                if (epoch + 1) % self.checkpoint_gap == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save, 'epoch_{:03d}.pkl'.format(epoch)))

            # Record IoU
            self.record("==== Epoch {:3} ====".format(epoch + 1))
            self.record("Training mIoU:")
            self.record(train_table_str)
            self.record("Testing mIoU:")
            self.record(test_table_str)

    def test(self, test_data, out_dir=None):
        if out_dir:
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

        self.model.eval()
        test_loss = 0
        test_iou_table = IouTable()

        for i, (cat_name, obj_ids, points, labels, mask, onehot) in enumerate(test_data):
            points = points.to(self.device)
            labels = labels.to(self.device)
            onehot = onehot.to(self.device)
            with torch.no_grad():
                out = self.model(points, onehot)
            loss = self.loss_function(out.reshape(-1, out.size(-1)), labels.view(-1, ))
            test_loss += loss.item()

            out[mask == 0] = out.min()
            pred = torch.max(out, 2)[1]
            self.calculate_save_mious(test_iou_table, cat_name, labels, pred)
            if out_dir:
                self.save_visualizations(out_dir, cat_name, obj_ids, points, labels, pred)

        test_loss /= (i + 1)
        c_miou = test_iou_table.get_mean_category_miou()
        i_miou = test_iou_table.get_mean_instance_miou()
        test_table_str = test_iou_table.get_string()

        if out_dir:
            miou_file = open(os.path.join(out_dir, "miou.txt"), "w")
            miou_file.write(test_table_str)

        return test_loss, test_table_str


if __name__ == '__main__':
    main()
