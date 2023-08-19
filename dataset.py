import glob
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable


# 设置全局的随机种子，提高可重复性
def set_same_seeds(seed):
    # Python内置随机模组
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 定义x域数据集，这里指的是真实人像
class XDomainDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        if transforms is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


# 定义y域数据集，这里指的是卡通动漫头像
class YDomainDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        if transforms is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


# 对原有图像进行transform操作
def transform_dataset(root_x, root_y):
    fnames_x = glob.glob(os.path.join(root_x, '*'))
    fnames_y = glob.glob(os.path.join(root_y, '*'))

    compose = [
        transforms.ToPILImage(),
        # transforms.Resize((256, 256)),  # 如果你的尺寸大小不一致，可以在这里修改大小
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose(compose)
    xset = XDomainDataset(fnames_x, transform)
    yset = YDomainDataset(fnames_y, transform)
    return xset, yset


# 先前生成的样本的缓冲区
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []  # 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  # 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  # 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))