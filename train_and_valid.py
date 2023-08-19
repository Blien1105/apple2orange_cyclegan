import logging
import os

import numpy as np
import torch
import torch_fidelity
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import itertools
import dataset
import network


# 定义学习率降低函数
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# 定义训练器
class Trainer(object):

    def __init__(self, device):

        self.device = device
        self.iterations = 0
        self.x_shape = [3, 256, 256]  # x、y域图片的真实尺寸，也就是放在你数据集里面的尺寸
        self.y_shape = [3, 256, 256]

        # 设置超参数
        self.n_critic = 2  # G 训练次数和 D 训练次数 之间的比值
        self.n_epoch = 60
        self.decay_epoch = 25  # 学习率在第几代后降低
        self.start_epoch = 0
        self.n_gen_pics = 1000  # 生成图片的个数
        self.n_res_blocks = 9  # 生成器残差模块个数
        self.lamda_cy = 1  # cycle loss 的权重
        self.lamda_id = 1  # identity loss 的权重
        self.batchsize = 1
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999

        # 设置各项路径
        self.workspace_dir = './'
        self.ckpt_dir = os.path.join(self.workspace_dir, 'checkpoints')
        # self.ckpt_dir = '../../../root/autodl-tmp'
        self.log_dir = os.path.join(self.workspace_dir, 'logs')
        self.gen_dir = os.path.join(self.workspace_dir, 'generated_pictures')
        # self.dataset_dir = '../../../root/autodl-tmp/apple2orange'  # 我用的是autodl上面租的服务器，此处看情况改
        self.dataset_dir = '../dataset___apple2orange'

        # 设置计算方法
        self.get_gan_loss = nn.MSELoss().to(device)
        self.get_cy_loss = nn.L1Loss().to(device)
        self.get_id_loss = nn.L1Loss().to(device)

        # 设置网络架构
        self.G = network.Generator(self.x_shape, self.n_res_blocks).to(self.device)     # 从x域中采样，生成y域的图片
        self.F = network.Generator(self.y_shape, self.n_res_blocks).to(self.device)     # 从y域中采样，生成x域的图片
        self.D_x = network.Discriminator().to(self.device)   # 鉴别x与F(y),并返回分数
        self.D_y = network.Discriminator().to(self.device)  # 鉴别y与G(x),并返回分数

        # 初始化
        self.xset = None
        self.yset = None
        self.xsetloader = None
        self.ysetloader = None

        self.x_hat_buffer = None
        self.y_hat_buffer = None

        self.optimizer_Gen = None
        self.optimizer_Dx = None
        self.optimizer_Dy = None

        self.lr_scheduler_G = None
        self.lr_scheduler_Dx = None
        self.lr_scheduler_Dy = None

    # 进行优化器、数据集的配置
    def prepare_environment(self):

        # 封装数据集
        self.xset, self.yset = dataset.transform_dataset(os.path.join(self.dataset_dir, 'trainA'),
                                               os.path.join(self.dataset_dir, 'trainB'))

        self.xsetloader = DataLoader(self.xset, batch_size=self.batchsize, shuffle=True)
        self.ysetloader = DataLoader(self.yset, batch_size=self.batchsize, shuffle=True)

        # 设置优化器
        self.optimizer_Gen = torch.optim.Adam(
            itertools.chain(self.G.parameters(), self.F.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_Dx = torch.optim.Adam(self.D_x.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_Dy = torch.optim.Adam(self.D_y.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # 设置图片缓存区
        self.x_hat_buffer = dataset.ReplayBuffer()
        self.y_hat_buffer = dataset.ReplayBuffer()

        # 设置学习率降低函数
        self.lr_scheduler_G = lr_scheduler.LambdaLR(self.optimizer_Gen,
                                                    lr_lambda=LambdaLR(self.n_epoch, self.start_epoch, self.decay_epoch).step)
        self.lr_scheduler_Dx = lr_scheduler.LambdaLR(self.optimizer_Dx,
                                                     lr_lambda=LambdaLR(self.n_epoch, self.start_epoch, self.decay_epoch).step)
        self.lr_scheduler_Dy = lr_scheduler.LambdaLR(self.optimizer_Dy,
                                                     lr_lambda=LambdaLR(self.n_epoch, self.start_epoch, self.decay_epoch).step)

        # 创建文件夹
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # 计算GAN loss
    def cal_GAN_loss(self, bs, data, D):
        data_logit = D(data)

        data_dim = data_logit.shape[1]

        data_label = Variable(torch.ones(bs * data_dim).view(-1, data_dim), requires_grad=False).to(self.device)

        gan_loss = self.get_gan_loss(data_logit, data_label)
        return gan_loss

    def cal_D_loss(self, bs, data, data_hat, D):
        data_logit = D(data)

        data_dim = data_logit.shape[1]

        data_label = Variable(torch.ones(bs * data_dim).view(-1, data_dim), requires_grad=False).to(self.device)
        data_hat_label = Variable(torch.zeros(bs * data_dim).view(-1, data_dim), requires_grad=False).to(self.device)

        if D == self.D_y:
            data_hat_from_buffer = self.y_hat_buffer.push_and_pop(data_hat)
            data_hat_logit = D(data_hat_from_buffer.detach())

            dataloss = self.get_gan_loss(data_logit, data_label)
            data_hat_loss = self.get_gan_loss(data_hat_logit, data_hat_label)
            d_gan_loss = dataloss + data_hat_loss

        else:
            data_hat_from_buffer = self.x_hat_buffer.push_and_pop(data_hat)
            data_hat_logit = D(data_hat_from_buffer.detach())

            dataloss = self.get_gan_loss(data_logit, data_label)
            data_hat_loss = self.get_gan_loss(data_hat_logit, data_hat_label)
            d_gan_loss = dataloss + data_hat_loss

        return d_gan_loss * 0.5

    # 计算cycle loss 即 cycle consistence
    def cal_cycle_loss(self, data, recov_data):
        cycle_loss = self.get_cy_loss(data, recov_data)
        return cycle_loss

    # 计算identity loss 其作用是保持色调
    def cal_identity_loss(self, data, G):
        identity_loss = self.get_id_loss(G(data), data)
        return identity_loss

    # 训练结束后，生成一些图片，以用于测试
    def generate_pics(self, type='xtype'):
        self.G.eval()
        self.F.eval()
        # 设置采样器，从x_set中随机抽取n_gen_pics个样本
        sampler = torch.utils.data.sampler.SubsetRandomSampler(
            np.random.choice(range(len(self.xset)), self.n_gen_pics))
        # 图片生成以及命名、保存
        if type == 'xtype':
            for i, indice in enumerate(sampler):
                x = self.xset[indice]
                x = x.view(1, 3, 256, 256).to(self.device)
                gen_pics = self.G(x).data
                filename = os.path.join(self.gen_dir, f'generated_pic_{i + 1:03d}.jpg')
                torchvision.utils.save_image(gen_pics, filename, nrow=1)

        else:
            for i, indice in enumerate(sampler):
                y = self.yset[indice]
                y = y.view(1, 3, 256, 256).to(self.device)
                gen_pics = self.F(y).data
                filename = os.path.join(self.gen_dir, f'generated_pic_{i + 1:03d}.jpg')
                torchvision.utils.save_image(gen_pics, filename, nrow=1)
    pass

    # 训练过程
    def train(self):
        # 取x集中的一张作为日志生成的原图
        x_sample = self.xset[1].view(1, 3, 256, 256).to(self.device)

        x_iterator = iter(self.xsetloader)
        for e, epoch in enumerate(range(self.n_epoch)):
            progress_bar = tqdm(self.ysetloader)
            progress_bar.set_description(f"Epoch {e + 1}")
            for steps, data in enumerate(progress_bar):

                imgs = data.to(self.device)
                bs = imgs.size(0)

                # 末尾剩余数据去除
                if bs != self.batchsize:
                    break

                # 此法可同时采样x，y域的图像，且不会产生内存泄漏
                try:
                    x = Variable(next(x_iterator)).to(self.device)
                except StopIteration:
                    x_iterator = iter(self.xsetloader)
                    x = Variable(next(x_iterator)).to(self.device)
                y = Variable(imgs).to(self.device)

                # *********************
                # *      训练生成器     *
                # *********************
                self.optimizer_Gen.zero_grad()

                self.G.train()
                self.F.train()

                # 计算GAN loss
                y_hat = self.G(x).to(self.device)
                x_hat = self.F(y).to(self.device)

                loss_GAN = 0.5 * (self.cal_GAN_loss(bs, y_hat, self.D_y) +
                            self.cal_GAN_loss(bs, x_hat, self.D_x))

                # 计算cycle loss
                recov_x = self.F(y_hat).to(self.device)
                recov_y = self.G(x_hat).to(self.device)

                loss_cycle = 0.5 * (self.cal_cycle_loss(x, recov_x) +
                              self.cal_cycle_loss(y, recov_y))

                # 计算identity loss
                loss_identity = 0.5 * (self.cal_identity_loss(y, self.G) +
                                 self.cal_identity_loss(x, self.F))

                loss_Gen = loss_GAN + loss_cycle*self.lamda_cy + loss_identity*self.lamda_id

                loss_Gen.backward()
                self.optimizer_Gen.step()

                if steps % self.n_critic == 0:

                    # *********************
                    # *     训练鉴别器      *
                    # *********************

                    self.optimizer_Dx.zero_grad()
                    self.optimizer_Dy.zero_grad()

                    # 训练D_y
                    loss_GAN_x2y = self.cal_D_loss(bs, y, y_hat, self.D_y)
                    loss_GAN_x2y.backward()
                    self.optimizer_Dy.step()

                    # 训练D_x
                    loss_GAN_y2x = self.cal_D_loss(bs, x, x_hat, self.D_x)
                    loss_GAN_y2x.backward()
                    self.optimizer_Dx.step()

                if steps % 2 == 0:
                    progress_bar.set_postfix(loss_Dy=loss_GAN_x2y.item(), loss_Dx=loss_GAN_y2x.item(), loss_Gen=loss_Gen.item(), loss_gan=loss_GAN.item(), loss_id=loss_identity.item(), loss_cy=loss_cycle.item())

                if steps % 60 == 0:
                    self.G.eval()
                    imgs_sample = self.G(x_sample).data
                    filename = os.path.join(self.log_dir, f'Epoch_{epoch + 1:02d}_Steps_{steps:03d}.jpg')
                    torchvision.utils.save_image(imgs_sample, filename, nrow=8)
                    logging.info(f'Save some samples to {filename}.')

            # 学习率改变
            self.lr_scheduler_G.step()
            self.lr_scheduler_Dx.step()
            self.lr_scheduler_Dy.step()

            # 每五个epoch存档一次
            if e % 5 == 0 or epoch == 0:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.F.state_dict(), os.path.join(self.ckpt_dir, f'F_{e}.pth'))
                torch.save(self.D_x.state_dict(), os.path.join(self.ckpt_dir, f'Dx_{e}.pth'))
                torch.save(self.D_y.state_dict(), os.path.join(self.ckpt_dir, f'Dy_{e}.pth'))
        pass

    # 用来加载之前训练过的模型,这里主要加载生成器模型，对鉴别器有需求可以自行加上
    def load_trained_model(self):

        self.G.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'G_55.pth')))  # 这里选择你上面训练好的心仪的一代
        self.F.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'F_55.pth')))

        self.G.eval()
        self.F.eval()

    # 生成评估指标is, fid, kid，这里直接用软件包torch_fidelity
    def validation(self):
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=os.path.join(self.workspace_dir, self.gen_dir),
            input2=os.path.join(self.dataset_dir, 'trainB'),
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        print(metrics_dict)
    pass
