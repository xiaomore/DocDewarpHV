# -*- coding: utf-8 -*-
# @Time : 2024/11/21 18:49
# @Author : Heng LI
# @FileName: doc_dewarp_hv_read.py
# @Software: PyCharm

"""
Code references:
# DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks (https://github.com/cvlab-stonybrook/DewarpNet)
# Revisiting Document Image Dewarping by Grid Regularization ()
"""

import collections
import os
from time import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
from torch.utils.data import Dataset, DataLoader
import hdf5storage as h5
import random
from PIL import Image


class DocDewarpHVData(Dataset):
    def __init__(self, input_size, data_root, split, is_aug=False):
        assert os.path.exists(data_root), 'Images folder does not exist'
        self.split = split
        self.is_aug = is_aug
        self.files = collections.defaultdict(list)
        self.data_root = data_root
        path = os.path.join(self.data_root, split + '.txt')
        file_list = tuple(open(path, 'r').read().splitlines())
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list

        self.img_size = input_size

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        try:
            img_filename = self.files[self.split][index]
            data_root = self.data_root

            im_path = os.path.join(data_root, 'warp_img', img_filename + '.png')
            bm_path = os.path.join(data_root, 'bm', img_filename.split('.')[0][:-4] + '_ann0001.mat')
            wc_path = os.path.join(data_root, 'wc', img_filename.split('.')[0][:-4] + '_ann0001.exr')
            uv_path = os.path.join(data_root, 'uvmat', img_filename.split('.')[0][:-4] + '_ann0001.mat')

            wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            h, w, c = wc.shape
            h_line_path = os.path.join(data_root, 'alb_h', img_filename.split('.')[0][:-4] + '_ann0001.png')
            v_line_path = os.path.join(data_root, 'alb_v', img_filename.split('.')[0][:-4] + '_ann0001.png')

            img = np.array(Image.open(im_path).convert("RGB"))
            h, w, c = img.shape

            bm = h5.loadmat(bm_path)['bm']
            h, w, c = bm.shape

            h_line = cv2.imread(h_line_path)
            h, w, c = h_line.shape
            v_line = cv2.imread(v_line_path)
            h, w, c = v_line.shape

            uv = h5.loadmat(uv_path)['uv']

            img, lbl, h_line, v_line, wc, uv = self.transform_new(wc, bm, img, h_line, v_line, uv, img_filename.split('/')[-1])

            lbl = lbl.permute((2, 0, 1))  # HWC -> CHW
        except Exception as e:
            print(f"Failed to read: {self.files[self.split][index]}")
            return self[index + 1]
        return img, lbl, wc, uv, h_line, v_line, img_filename.split('/')[-1]

    def tight_crop(self, wc, img, h_line, v_line, uv):
        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8)
        size = msk.shape
        [y, x] = (msk).nonzero()
        crop_random = random.random()
        if crop_random > 0.55:
            minx = min(x) // 2
            maxx = size[1] - (size[1] - max(x)) // 2
            miny = min(y) // 2
            maxy = size[0] - (size[0] - max(y)) // 2

            s = 0
        else:
            minx = min(x)
            maxx = max(x)
            miny = min(y)
            maxy = max(y)
            s = random.randint(7, 25)

        wc = wc[miny: maxy + 1, minx: maxx + 1, :]
        img = img[miny: maxy + 1, minx: maxx + 1, :]
        h_line = h_line[miny: maxy + 1, minx: maxx + 1, :]
        v_line = v_line[miny: maxy + 1, minx: maxx + 1, :]
        uv = uv[miny: maxy + 1, minx: maxx + 1, :]

        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')
        img = np.pad(img, ((s, s), (s, s), (0, 0)), 'constant')
        h_line = np.pad(h_line, ((s, s), (s, s), (0, 0)), 'constant')
        v_line = np.pad(v_line, ((s, s), (s, s), (0, 0)), 'constant')
        uv = np.pad(uv, ((s, s), (s, s), (0, 0)), 'constant')

        t = miny - s# + cy1
        b = size[0] - maxy - s# + cy2
        l = minx - s# + cx1
        r = size[1] - maxx - s# + cx2

        return wc, img, h_line, v_line, uv, t, b, l, r

    def transform_new(self, wc, bm, img, h_line, v_line, uv, img_name):
        if self.is_aug:
            if random.random() > 0.55:
                img = color_line(img, bm)
        if random.random() > 0.1:
            wc, img, h_line, v_line, uv, t, b, l, r = self.tight_crop(wc, img, h_line, v_line, uv)
        else:
            t, b, l, r = 0, 0, 0, 0

        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8) * 255

        bm = bm.astype(float)
        bm[:, :, 1] = bm[:, :, 1] - t
        bm[:, :, 0] = bm[:, :, 0] - l
        bm = bm / np.array([512. - l - r, 512. - t - b])  # to [0, 1]

        if self.is_aug:
            if random.random() > 0.8:
                img = color_jitter(img, 0.2, 0.2, 0.6, 0.6)

        bm0 = cv2.resize(bm[:, :, 0], (self.img_size, self.img_size))
        bm1 = cv2.resize(bm[:, :, 1], (self.img_size, self.img_size))
        bm = np.stack([bm0, bm1], axis=-1)

        # bm = bm * (self.img_size)
        lbl = torch.from_numpy(bm).float()

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float64)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        img = img.astype(float) / 255.0  # 使用normalize归一化
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW
        img = torch.from_numpy(img).float()

        h_line = cv2.resize(h_line, (self.img_size, self.img_size))
        h_line = h_line.astype(float) / 255.
        h_line = h_line.transpose(2, 0, 1)  # NHWC -> NCHW
        h_line = h_line[[0], :, :]
        h_line = torch.from_numpy(h_line).float()

        v_line = cv2.resize(v_line, (self.img_size, self.img_size))
        v_line = v_line.astype(float) / 255.
        v_line = v_line.transpose(2, 0, 1)  # NHWC -> NCHW
        v_line = v_line[[0], :, :]
        v_line = torch.from_numpy(v_line).float()

        uv = cv2.resize(uv, (self.img_size, self.img_size))
        uv = uv.astype(float)
        uv = uv.transpose(2, 0, 1)  # NHWC -> NCHW
        uv = torch.from_numpy(uv).float()

        zmn = np.min(wc[:, :, 0])
        zmx = np.max(wc[:, :, 0])
        ymn = np.min(wc[:, :, 1])
        ymx = np.max(wc[:, :, 1])
        xmn = np.min(wc[:, :, 2])
        xmx = np.max(wc[:, :, 2])

        wc[:, :, 0] = (wc[:, :, 0] - zmn) / (zmx - zmn)
        wc[:, :, 1] = (wc[:, :, 1] - ymn) / (ymx - ymn)
        wc[:, :, 2] = (wc[:, :, 2] - xmn) / (xmx - xmn)

        wc = cv2.bitwise_and(wc, wc, mask=msk)
        wc = cv2.resize(wc, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        wc = wc.astype(float)
        wc = wc.transpose(2, 0, 1)  # NHWC -> NCHW
        wc = torch.from_numpy(wc).float()

        return img, lbl, h_line, v_line, wc, uv


def color_line(im, bm):
    # im = im# * 255
    # bm = bm * 448
    chance = random.random()
    if chance < 0.8:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[random.randint(2, 18):random.randint(20, 40), :, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)
        if random.random() > 0.7:
            c = np.array([1, 1, 1]) * 255.0
            t = bm[:random.randint(1, 10), :, :].reshape([-1, 2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)
    elif chance > 0.3 and chance < 0.6:
        cc = random.randint(2, 18)
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, random.randint(2, 18):random.randint(20, 40), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

        if random.random() > 0.7:
            c = np.array([1, 1, 1]) * 255.0
            t = bm[:, :random.randint(1, 10), :].reshape([-1, 2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if chance < 0.8:
        c = np.array([0, 0, 0]) * 255.0
        t = bm[25:random.randint(30, 40), :random.randint(112, 224), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, :10, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, [255, 255, 255], thickness=1)

    chance = random.random()
    if chance < 0.1:
        im[:, :, 0] = random.random() * 255.0
    elif chance < 0.2 and chance > 0.1:
        im[:, :, 1] = random.random() * 255.0

    elif chance < 0.6 and chance > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:random.randint(20, 45), :, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)
    elif chance < 0.8 and chance > 0.6:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, :random.randint(20, 45), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if random.random() > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, :random.randint(1, 20), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if random.random() > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:random.randint(1, 20), :, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if chance > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        num = int(random.random() * 20)
        cc = random.randint(10, 15)
        for m in range(30):
            t = bm[num:num + 20, 50 + cc * m:cc * m + 57, :].reshape([-1, 2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if chance > 0.9:
        im = 255 - im
    elif chance > 0.85:
        im[:, :, 0] = 255
    elif chance > 0.8:
        im[:, :, 0] = 255
    elif chance > 0.75:
        im[:, :, 0] = 0
    elif chance > 0.7:
        im[:, :, 1] = 255
    elif chance > 0.65:
        im[:, :, 1] = 0
    elif chance > 0.6:
        im[:, :, 2] = 255
    elif chance > 0.55:
        im[:, :, 2] = 0

    return im# / 255


def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    im = im / 255.
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)

    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)

    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    f = random.uniform(-hue, hue)
    hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)

    f = random.uniform(-saturation, saturation)
    hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    im = np.clip(im, 0., 1.)
    return im * 255.


if __name__ == '__main__':
    data_path = 'YOUR_DATA_PATH'
    dataset_train = DocDewarpHVData(448, data_path, 'DocDewarpHV', is_aug=False)
    train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=1)
    for img, lbl, wc, uv, h_line, v_line, img_filename in train_loader:
        start = time()
        print("-" * 50)
        print("img.shape: ", img.shape)
        print("lbl.shape: ", lbl.shape)
        print("wc.shape: ", wc.max())
        print("uv.shape: ", uv.shape)
        print("h_line.shape: ", h_line.shape)
        print("v_line.shape: ", v_line.shape)

        print("---time: ", time() - start)
        print("-" * 50)