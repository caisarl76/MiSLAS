import os

import numpy as np
import pandas as pd
import torch.utils.data
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from custum_data.randaugment import rand_augment_transform

class Cub2011(Dataset):
    base_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, imb_type='exp', imb_factor=1.0, transform=None, rand_number=0, random_seed=False, ra_params=None):
        self.random_seed = random_seed
        np.random.seed(rand_number)
        self.ra_params = ra_params
        self.root = os.path.expanduser(root)
        self.root = os.path.join(os.getcwd(), self.root, 'cub')
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.img_num_list = self.get_img_num_per_cls(200, imb_type, imb_factor)
        self.gen_imbalanced_data()
        self.hard_aug = False

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = 30
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(
            os.path.join(self.root, 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            train = self.data[self.data.is_training_img == 1]
            temp = train[train.target == 1].iloc[np.random.choice(30, self.img_num_list[0]), :]
            self.data = temp
            self.targets = list(temp.target.array - 1)
            for i in range(2, 201):
                idx = np.random.choice(len(train[train.target == i]), self.img_num_list[i - 1])
                temp = train[train.target == i].iloc[idx, :]
                self.data = self.data.append(temp)
                self.targets += list(temp.target - 1)
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def get_cls_num_list(self):
        return self.img_num_list

    def set_raparam(self, ra_params):
        self.ra_params = ra_params
        self.hard_aug = True
        self.cls_num_list = self.get_cls_num_list()
        self.factor = np.array(self.cls_num_list) / self.cls_num_list[0]

    def set_step(self, step):
        self.step = step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.hard_aug:
            classwise = 1 - (0.5 * self.factor[target])
            n = (int)(self.step * classwise)
            if type(self.transform) == list:
                samples = []
                for i, resol in enumerate([480, 224, 128]):
                    ra_param = self.ra_params[i]
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(resol, scale=(0.08, 1.)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
                        ], p=1.0),
                        rand_augment_transform('rand-n{}-m{}-mstd{}'.format(n, 2, 0.5), ra_param),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4707, 0.4601, 0.4550], std=[0.2667, 0.2658, 0.2706]),
                    ])
                    samples.append(transform(img))
                return samples, target
        if self.transform is not None:
            if type(self.transform) == list:
                samples = []
                for transform in self.transform:
                    sample = transform(img)
                    samples.append(sample)
                return samples, target
            else:
                img = self.transform(img)
                return img, target



if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = Cub2011(root='/media/hd/jihun/data/CUB_200_2011', train=True, transform=train_transform)
    test_dataset = Cub2011(root='/media/hd/jihun/data/CUB_200_2011', train=False, transform=train_transform)
    print(len(train_dataset))
    print(len(test_dataset))
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, (image, target) in enumerate(val_loader):
        print(image.shape, target.shape)
        break