import os
from PIL import Image
import torch
from torch.utils import data
import transforms as trans
import random
from parameter import *


class CoData_Train(data.Dataset):

    def __init__(self, img_root, gt_root, img_root_coco, gt_root_coco, img_size, transform, t_transform, label_32_transform, label_64_transform, label_128_transform, max_num):
        # Path Pool
        self.img_root = img_root  # root
        self.gt_root = gt_root
        self.dirs = os.listdir(img_root)  # all dir
        self.img_dir_paths = list(  # [img_root+dir1, ..., img_root+dir2]
            map(lambda x: os.path.join(img_root, x), self.dirs))
        self.gt_dir_paths = list(  # [gt_root+dir1, ..., gt_root+dir2]
            map(lambda x: os.path.join(gt_root, x), self.dirs))
        self.img_name_list = [os.listdir(idir) for idir in self.img_dir_paths
                              ]  # [[name00,..., 0N],..., [M0,..., MN]]
        self.gt_name_list = [
            map(lambda x: x[:-3] + 'png', iname_list)
            for iname_list in self.img_name_list
            ]
        self.img_path_list = [
            list(
                map(lambda x: os.path.join(self.img_dir_paths[idx], x),
                    self.img_name_list[idx]))
            for idx in range(len(self.img_dir_paths))
            ]  # [[impath00,..., 0N],..., [M0,..., MN]]
        self.gt_path_list = [
            list(
                map(lambda x: os.path.join(self.gt_dir_paths[idx], x),
                    self.gt_name_list[idx]))
            for idx in range(len(self.gt_dir_paths))
            ]  # [[gtpath00,..., 0N],..., [M0,..., MN]]
        self.nclass = len(self.dirs)

        # CoCo Path Pool
        self.img_root_coco = img_root_coco  # root
        self.gt_root_coco = gt_root_coco
        self.dirs_coco = os.listdir(img_root_coco)  # all dir
        self.img_dir_paths_coco = list(  # [img_root+dir1, ..., img_root+dir2]
            map(lambda x: os.path.join(img_root_coco, x), self.dirs_coco))
        self.gt_dir_paths_coco = list(  # [gt_root+dir1, ..., gt_root+dir2]
            map(lambda x: os.path.join(gt_root_coco, x), self.dirs_coco))
        self.img_name_list_coco = [os.listdir(idir) for idir in self.img_dir_paths_coco
                              ]  # [[name00,..., 0N],..., [M0,..., MN]]
        self.gt_name_list_coco = [
            map(lambda x: x[:-3] + 'png', iname_list)
            for iname_list in self.img_name_list_coco
            ]
        self.img_path_list_coco = [
            list(
                map(lambda x: os.path.join(self.img_dir_paths_coco[idx], x),
                    self.img_name_list_coco[idx]))
            for idx in range(len(self.img_dir_paths_coco))
            ]  # [[impath00,..., 0N],..., [M0,..., MN]]
        self.gt_path_list_coco = [
            list(
                map(lambda x: os.path.join(self.gt_dir_paths_coco[idx], x),
                    self.gt_name_list_coco[idx]))
            for idx in range(len(self.gt_dir_paths_coco))
            ]  # [[gtpath00,..., 0N],..., [M0,..., MN]]
        self.nclass_coco = len(self.dirs_coco)

        # Other Hyperparameters
        self.size = img_size
        self.cat_size = int(img_size * 2)
        self.sizes = [img_size, img_size]
        self.transform = transform
        self.t_transform = t_transform
        self.label_32_transform = label_32_transform
        self.label_64_transform = label_64_transform
        self.label_128_transform = label_128_transform
        self.max_num = max_num

    def __getitem__(self, item):

        if random.random() < 0.5:
            # select coco data
            flag = False
            sel = item%(self.nclass_coco-1)
            img_paths = self.img_path_list_coco[sel]
            gt_paths = self.gt_path_list_coco[sel]
        else:
            # select from our dataset
            flag = True
            img_paths = self.img_path_list[item]
            gt_paths = self.gt_path_list[item]

        num = len(img_paths)
        if num > self.max_num:
            sampled_list = random.sample(range(num), self.max_num)
            new_img_paths = [img_paths[i] for i in sampled_list]
            img_paths = new_img_paths
            new_gt_paths = [gt_paths[i] for i in sampled_list]
            gt_paths = new_gt_paths
            num = self.max_num

        imgs = torch.Tensor(num, 3, self.sizes[0], self.sizes[1])
        gts = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])
        gts_32 = torch.Tensor(num, 1, img_size//8, img_size//8)
        gts_64 = torch.Tensor(num, 1, img_size//4, img_size//4)
        gts_128 = torch.Tensor(num, 1, img_size//2, img_size//2)

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            if flag:
                # data from our dataset
                # random replace to syn img or do not replace

                select_num = random.randint(1, 5)
                if select_num == 4:
                    # select original img
                    img_path = img_paths[idx]
                    gt_path = gt_paths[idx]
                if 1 <= select_num <= 3:
                    # select syn img
                    img_path = img_paths[idx].split('.jpg')[0] + '_syn' + str(select_num) + '.png'
                    img_path = img_syn_root + img_path.split('/img/')[1]
                    gt_path = gt_paths[idx]
                if select_num == 5:
                    # select reverse syn img
                    select_reverse_num = random.randint(1, 3)
                    img_path = img_paths[idx].split('.jpg')[0] + '_ReverseSyn' + str(select_reverse_num) + '.png'
                    tmp = img_path.split('/img/')[1]
                    img_path = img_ReverseSyn_root + tmp
                    gt_path = gt_ReverseSyn_root + tmp
            else:
                # data from coco
                img_path = img_paths[idx]
                gt_path = gt_paths[idx]

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')

            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))

            random_size = scale_size
            new_img = trans.Scale((random_size, random_size))(img)
            new_gt = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(gt)

            # random crop
            w, h = new_img.size
            if w != img_size and h != img_size:
                x1 = random.randint(0, w - img_size)
                y1 = random.randint(0, h - img_size)
                new_img = new_img.crop((x1, y1, x1 + img_size, y1 + img_size))
                new_gt = new_gt.crop((x1, y1, x1 + img_size, y1 + img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_gt = new_gt.transpose(Image.FLIP_LEFT_RIGHT)

            new_img = self.transform(new_img)
            gt_256 = self.t_transform(new_gt)
            gt_32 = self.label_32_transform(new_gt)
            gt_64 = self.label_64_transform(new_gt)
            gt_128 = self.label_128_transform(new_gt)

            imgs[idx] = new_img
            gts[idx] = gt_256
            gts_128[idx] = gt_128
            gts_64[idx] = gt_64
            gts_32[idx] = gt_32

        return imgs, gts, gts_128, gts_64, gts_32, subpaths, ori_sizes

    def __len__(self):
        return len(self.dirs)


class CoData_Test(data.Dataset):
    def __init__(self, img_root, img_size, transform):

        class_list = os.listdir(img_root)
        self.sizes = [img_size, img_size]
        self.transform = transform
        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))

    def __getitem__(self, item):

        names = os.listdir(self.img_dirs[item])
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))

        imgs = torch.Tensor(num, 3, self.sizes[0], self.sizes[1])

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))
            img = self.transform(img)
            imgs[idx] = img

        return imgs, subpaths, ori_sizes

    def __len__(self):
        return len(self.img_dirs)


def get_loader(img_root, img_size, batch_size, gt_root=None, max_num=float('inf'), mode='train', num_thread=1, pin=False):
    shuffle = False
    mean = torch.Tensor(3, img_size, img_size)
    mean[0, :, :] = 122.675  # R
    mean[1, :, :] = 116.669  # G
    mean[2, :, :] = 104.008  # B

    mean_bgr = torch.Tensor(3, img_size, img_size)
    mean_bgr[0, :, :] = 104.008  # B
    mean_bgr[1, :, :] = 116.669  # G
    mean_bgr[2, :, :] = 122.675  # R

    if mode == 'train':
        transform = trans.Compose([
            # trans.ToTensor  image -> [0,255]
            trans.ToTensor_BGR(),
            trans.Lambda(lambda x: x - mean_bgr)
        ])

        t_transform = trans.Compose([
            # transform.ToTensor  label -> [0,1]
            trans.ToTensor(),
        ])
        label_32_transform = trans.Compose([
            trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])
        label_64_transform = trans.Compose([
            trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])
        label_128_transform = trans.Compose([
            trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])
        shuffle = True
    else:

        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            trans.ToTensor_BGR(),
            trans.Lambda(lambda x: x - mean_bgr)
        ])

        t_transform = trans.Compose([
            trans.Scale((img_size, img_size), interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])
    if mode == 'train':
        dataset = CoData_Train(img_root, gt_root, img_root_coco, gt_root_coco, img_size, transform, t_transform,
                               label_32_transform, label_64_transform, label_128_transform, max_num)

    else:
        dataset = CoData_Test(img_root, img_size, transform)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread, pin_memory=pin)
    return data_loader

