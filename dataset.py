import math
import os

import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2

from scipy.stats import norm

import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

img_dir = "static/png"
pos_dir = "static/txt"
train_images = "data/train_images"
train_masks = "data/train_masks"


def ellipse_around_point(xc, yc, a, d, r1, r2):
    ind = np.zeros((2, d, d), dtype=np.int)
    m = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        ind[0, :, i] = range(-yc, d - yc)
    for i in range(d):
        ind[1, i, :] = range(-xc, d - xc)
    rs1 = np.arange(r1, 0, -float(r1) / r2)
    rs2 = np.arange(r2, 0, -1.0)
    s = math.sin(a)
    c = math.cos(a)

    pdf0 = norm.pdf(0)
    for i in range(len(rs1)):
        i1 = rs1[i]
        i2 = rs2[i]
        v = norm.pdf(float(len(rs1) - i) / len(rs1)) / pdf0
        # rotated ellipse
        m[((ind[0, :, :] * s + ind[1, :, :] * c) ** 2 / i1 ** 2 + (
                ind[1, :, :] * s - ind[0, :, :] * c) ** 2 / i2 ** 2) <= 1] = v

    return m


def generate_segm_labels(img, pos, w=10, r1=7, r2=12, FR_D=512):
    res = np.zeros((4, FR_D, FR_D), dtype=np.float32)  # data,labels_segm, labels_angle, weight
    # res[0] = img
    res[2] = -1

    for i in range(pos.shape[0]):
        x, y, obj_class, a = tuple(pos[i, :])

        obj_class += 1

        if obj_class == 2:
            a = 2 * math.pi
        else:
            a = math.radians(float(a))

        if obj_class == 1:
            m = ellipse_around_point(x, y, a, FR_D, r1, r2)
        elif obj_class == 3:
            m = ellipse_around_point(x, y, a, FR_D, r1, r2 * 2)
        else:
            m = ellipse_around_point(x, y, a, FR_D, r1, r1)

        mask = (m != 0)
        res[1][mask] = obj_class
        res[2][mask] = a / (2 * math.pi)
        res[3][mask] = m[mask]

    res[3] = res[3] * (w - 1) + 1
    return res


def read_img(fr, path):
    image = Image.open(os.path.join(path, "%06d.png" % fr)).convert('RGB')
    image = np.asarray(image)
    #plt.imshow(image,interpolation='nearest')
    #plt.show()
    return image


def create_from_frames_mask(frame_nbs, img_dir, pos_dir, mask_dir='static/masks'):
    FR_D = 512

    res = np.zeros((len(frame_nbs), 4, FR_D, FR_D), dtype=np.float32)
    for i, frame_nb in enumerate(frame_nbs):
        print("frame %i.." % frame_nb)
        img = read_img(frame_nb, img_dir)
        #pos = np.loadtxt(os.path.join(pos_dir, "%06d.txt" % frame_nb), delimiter=",", dtype=np.int)
        #res[i] = generate_segm_labels(img, pos)
        #im = Image.fromarray(res[i][1]).convert('RGB')
        #im.save(os.path.join(mask_dir, "mask_%06d.png" % frame_nb))


        #im = Image.fromarray(res[i][0]).convert('RGB')
        #cv2.imshow("asd",img)
        #cv2.waitKey()
        img = Image.fromarray(img)
        img.save(os.path.join('static/images', "%06d.png" % frame_nb))

####### DATA-SET ############
class LyftUdacity(Dataset):
    def __init__(self, img_dir='static', transform=None):
        self.transforms = transform
        image_paths = [img_dir+'/images']
        seg_paths = [img_dir+'/masks']
        self.images, self.masks = [], []
        for i in image_paths:
            imgs = os.listdir(i)
            self.images.extend([i + '/' + img for img in imgs])
        for i in seg_paths:
            masks = os.listdir(i)
            self.masks.extend([i + '/' + mask for mask in masks])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))

        if self.transforms is not None:

            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = torch.max(mask, dim=2)[0]

        return img, mask


def get_images(image_dir, transform=None, batch_size=1, shuffle=True, pin_memory=True):
    data = LyftUdacity(image_dir, transform=t1)
    train_size = int(0.8 * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                             pin_memory=pin_memory)
    return train_batch, test_batch


t1 = A.Compose([
    A.Resize(256, 256),
    A.augmentations.transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    ToTensorV2()
])
train_batch,test_batch = get_images(image_dir='static',transform =t1,batch_size=8)
if __name__ == "__main__":
    train_batch,test_batch = get_images(image_dir='static',transform =t1,batch_size=8)
    for img, mask in train_batch:
        img1 = np.transpose(img[0, :, :, :], (1, 2, 0))
        mask1 = np.array(mask[0, :, :])
        img2 = np.transpose(img[1, :, :, :], (1, 2, 0))
        mask2 = np.array(mask[1, :, :])
        img3 = np.transpose(img[2, :, :, :], (1, 2, 0))
        mask3 = np.array(mask[2, :, :])


        fig, ax = plt.subplots(3, 2, figsize=(18, 18))
        ax[0][0].imshow(img1)
        ax[0][1].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(mask3)
        plt.show()

        break