from os import listdir
from os.path import join
import random
from sklearn import preprocessing
from torchvision import transforms

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from util.Nii_utils import NiiDataRead
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from skimage import transform
import os
import torchvision.transforms as transforms
from util.util import *
def randomcrop_Npatch(crop_size, crop_Npatch, mri1, ct, ct_mask):
    this_frame = crop_size
    img = mri1

    non_zero_z, non_zero_x, non_zero_y = np.where(ct_mask == 1)
    non_zero_num = non_zero_x.shape[0]

    patch_index = random.sample(range(0, non_zero_num), crop_Npatch)
    patch_mri1 = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_ct = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_mask = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(int)

    for idx in range(crop_Npatch):
        z_med = non_zero_z[patch_index[idx]]
        x_med = non_zero_x[patch_index[idx]]
        y_med = non_zero_y[patch_index[idx]]
        z_frame_size = int(this_frame[0] / 2)
        x_frame_size = int(this_frame[1] / 2)
        y_frame_size = int(this_frame[2] / 2)

        # 计算裁剪区域的边界
        z_this_min = max(0, z_med - z_frame_size)
        z_this_max = min(img.shape[0], z_med + z_frame_size)
        x_this_min = max(0, x_med - x_frame_size)
        x_this_max = min(img.shape[1], x_med + x_frame_size)
        y_this_min = max(0, y_med - y_frame_size)
        y_this_max = min(img.shape[2], y_med + y_frame_size)

        # 裁剪区域
        mri1_crop = mri1[z_this_min:z_this_max, x_this_min:x_this_max, y_this_min:y_this_max]
        ct_crop = ct[z_this_min:z_this_max, x_this_min:x_this_max, y_this_min:y_this_max]
        ct_mask_crop = ct_mask[z_this_min:z_this_max, x_this_min:x_this_max, y_this_min:y_this_max]

        # 计算需要填充的尺寸
        pad_width = []
        for dim in range(3):
            current_size = mri1_crop.shape[dim]
            target_size = this_frame[dim]
            pad_total = max(target_size - current_size, 0)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_width.append((pad_before, pad_after))

        # 进行零填充
        mri1_padded = np.pad(mri1_crop, pad_width, mode='constant', constant_values=0)
        ct_padded = np.pad(ct_crop, pad_width, mode='constant', constant_values=0)
        ct_mask_padded = np.pad(ct_mask_crop, pad_width, mode='constant', constant_values=0)

        # 确保填充后的尺寸正确
        patch_mri1[idx] = mri1_padded
        patch_ct[idx] = ct_padded
        patch_mask[idx] = ct_mask_padded

    return (
        np.ascontiguousarray(patch_mri1),
        np.ascontiguousarray(patch_ct),
        np.ascontiguousarray(patch_mask)
    )

class DatasetFromFolder_train(data.Dataset):
    def __init__(self, opt, region):
        self.image_dir = opt.image_dir
        self.Max_CT = opt.Max_CT

        if region=='Brain':
            self.train_txt = os.path.join(opt.code_dir, 'data', 'brain_train.txt')
        elif region=='All':
            self.train_txt = os.path.join(opt.code_dir, 'data', 'all_train.txt')
        with open(self.train_txt, 'r') as f:
            name_list = f.readlines()
        self.image_filenames = [n.strip('\n') for n in name_list]
        self.crop_size = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.crop_Npatch = opt.Npatch
        self.all_patch_num = self.crop_Npatch * len(self.image_filenames)

    def __getitem__(self, index):
        this_index = int(index // self.crop_Npatch)
        self.ran_num = 1
        patient_name = self.image_filenames[this_index]

        if 'Pelvis' in patient_name:
            a_mri1, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'mr.nii.gz'))
            b_ct, spacing1, origin1, direction1 = NiiDataRead(
                join(self.image_dir, patient_name, 'ct.nii.gz'))
            b_mask, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'mask.nii.gz'))
            ct_max = self.Max_CT
            label = 1
        else:
            a_mri1, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'mr.nii.gz'))
            b_ct, spacing1, origin1, direction1 = NiiDataRead(
                join(self.image_dir, patient_name, 'ct.nii.gz'))
            b_mask, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'mask.nii.gz'))
            ct_max = self.Max_CT
            label = 0

        a_mri1 = normalization(a_mri1, 0, 255)
        ct_min = -1000

        b_ct[b_ct < ct_min] = ct_min
        b_ct[b_ct > ct_max] = ct_max
        b_ct[b_mask == 0] = ct_min
        b_ct = normalization(b_ct, ct_min, ct_max)

        a_patch_mri1, b_patch_ct, b_patch_mask = randomcrop_Npatch(self.crop_size, self.ran_num, a_mri1, b_ct, b_mask)

        a = torch.tensor(a_patch_mri1).float()
        b = torch.tensor(b_patch_ct).float()
        mask = torch.tensor(b_patch_mask).float()

        p1 = np.random.choice([0, 1])
        p2 = np.random.choice([0, 1])
        self.trans = transforms.Compose([
                                  transforms.RandomHorizontalFlip(p1),
                                  transforms.RandomVerticalFlip(p2),
                                       ])
        a = self.trans(a)
        b = self.trans(b)
        mask = self.trans(mask)
        label = torch.tensor(int(label)).long()


        return {
            'A': a,
            'B': b,
            'mask': mask,
            'label':label
        }

    def __len__(self):
        return self.all_patch_num




