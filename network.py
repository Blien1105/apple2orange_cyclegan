import torch.nn as nn
import torch.nn.functional as F

# 定义残差模块，这是Generator的组成部分
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_shape, n_res_blocks=9):
        super(Generator, self).__init__()

        channels = input_shape[0]  # 输入特征数
        out_features = 64  # 输出特征数

        # 初始化net
        net = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # 下采样，循环2次
        for _ in range(2):
            out_features *= 2
            net += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 残差模块，循环9次
        for _ in range(n_res_blocks):
            net += [ResidualBlock(out_features)]

        # 上采样，循环2次
        for _ in range(2):
            out_features //= 2
            net += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 网络输出层
        net += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.generator = nn.Sequential(*net)
        self.weight_init()

    def forward(self, x):
        out = self.generator(x)
        return out

    def weight_init(self):
        for m in self.generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
    pass


# 定义鉴别器，采用PathGAN架构,即Markov Disciminat0r
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # 卷积层
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True)]

        # 全连接层
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # 平均池展开
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

