import os
import sys
from util import *
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def get_miou(pred: "tensor (point_num, )", target: "tensor (point_num, )", valid_labels: list):
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    part_ious = []
    for part_id in valid_labels:
        pred_part = (pred == part_id)
        target_part = (target == part_id)
        I = np.sum(np.logical_and(pred_part, target_part))
        U = np.sum(np.logical_or(pred_part, target_part))
        if U == 0:
            part_ious.append(1)
        else:
            part_ious.append(I / U)
    miou = np.mean(part_ious)
    return miou

def get_macc(pred: "tensor (point_num, )", target: "tensor (point_num, )", valid_labels: list):
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    accuracy=0.0
    correct_count = 0
    total_count = 0
    for part_id in valid_labels:
        pred_part = (pred == part_id)
        target_part = (target == part_id)
        correct_count += np.sum(np.logical_and(pred_part, target_part))
        total_count += np.sum(target_part)

    if total_count == 0:
        total_count = 1.0  # 处理目标标签中没有有效标签的情况
    else:
        accuracy = correct_count / total_count

    return accuracy




class IouTable():
    def __init__(self):
        self.obj_miou = {}
        self.obj_macc = {}

    def add_obj_miou(self, category: str, miou: float):
        if category not in self.obj_miou:
            self.obj_miou[category] = [miou]
        else:
            self.obj_miou[category].append(miou)

    def add_obj_macc(self, category: str, macc: float):
        if category not in self.obj_macc:
            self.obj_macc[category] = [macc]
        else:
            self.obj_macc[category].append(macc)

    def get_category_miou(self):
        """
        Return: moiu table of each category
        """
        category_miou = {}
        for c, mious in self.obj_miou.items():
            category_miou[c] = np.mean(mious)
        return category_miou

    def get_category_macc(self):
        """
        Return: moiu table of each category
        """
        category_macc = {}
        for c, macc in self.obj_macc.items():
            category_macc[c] = np.mean(macc)
        return category_macc

    def get_mean_category_miou(self):
        category_miou = []
        for c, mious in self.obj_miou.items():
            c_miou = np.mean(mious)
            category_miou.append(c_miou)
        return np.mean(category_miou)

    def get_mean_category_macc(self):
        category_macc = []
        for c, macc in self.obj_macc.items():
            c_macc = np.mean(macc)
            category_macc.append(c_macc)
        return np.mean(category_macc)

    def get_mean_instance_miou(self):
        object_miou = []
        for c, mious in self.obj_miou.items():
            object_miou += mious
        return np.mean(object_miou)

    def get_mean_instance_macc(self):
        object_macc = []
        for c, macc in self.obj_macc.items():
            object_macc += macc
        return np.mean(object_macc)

    def get_string(self):
        mean_c_miou = self.get_mean_category_miou()
        mean_i_miou = self.get_mean_instance_miou()
        mean_c_macc = self.get_mean_category_macc()
        mean_i_macc = self.get_mean_instance_macc()
        first_row = "| {:5} | {:5} ||".format("Avg_c", "Avg_i")
        second_row = "| {:.3f} | {:.3f} ||".format(mean_c_miou, mean_i_miou)
        third_row = "| {:.3f} | {:.3f} ||".format(mean_c_macc, mean_i_macc)
        categories = list(self.obj_miou.keys())
        categories.sort()
        cate_miou = self.get_category_miou()
        for c in categories:
            miou = cate_miou[c]

            first_row += " {:5} |".format(c[:3])
            second_row += " {:.3f} |".format(miou)
        string = first_row + "\n" + second_row
        acategories = list(self.obj_macc.keys())
        acategories.sort()
        cate_macc = self.get_category_macc()
        for c in acategories:
            macc = cate_macc[c]
            first_row += " {:5} |".format(c[:3])
            second_row += " {:.3f} |".format(macc)
        string2 = first_row + "\n" + second_row
        result_string = string2 + "\n" 
        return result_string


def test():
    pass


if __name__ == "__main__":
    test()

