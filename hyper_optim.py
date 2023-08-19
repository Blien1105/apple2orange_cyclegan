import itertools
import os

import numpy as np
import optuna
import torch
import torch_fidelity
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import dataset
import network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_dir = '../../../root/autodl-tmp/apple2orange'  # 我这里租了一个autodl的服务器
dataset_dir = '../dataset___apple2orange'
workspace_dir = './'
gen_dir = os.path.join(workspace_dir, 'generated_pictures')
save_dir = '../../../root/autodl-tmp'

batchsize = 1
b1 = 0.5
b2 = 0.999
n_res_blocks = 6
n_gen_pics = 100
n_trials = 100
n_epoch = 3
lr = 0.0003

get_gan_loss = nn.MSELoss().to(device)
get_cy_loss = nn.L1Loss().to(device)
get_id_loss = nn.L1Loss().to(device)
x_shape = [3, 256, 256]
y_shape = [3, 256, 256]
best_score = 99999


# 生成一些图片，用于fid评定（下面很多内容都是隔壁搬来的）
def generate_pics(G):
    G.eval()
    # 设置采样器，从x_set中随机抽取n_gen_pics个样本
    sampler = torch.utils.data.sampler.SubsetRandomSampler(
        np.random.choice(range(len(xset)), n_gen_pics))
    # 图片生成以及命名、保存
    for i, indice in enumerate(sampler):
        x = xset[indice]
        x = x.view(1, 3, 256, 256).to(device)
        gen_faces = G(x).data
        filename = os.path.join(gen_dir, f'generated_pic_{i + 1:03d}.jpg')
        torchvision.utils.save_image(gen_faces, filename, nrow=1)


# 计算GAN loss
def cal_GAN_loss(bs, data, data_hat, D):
    data_logit = D(data)
    data_hat_logit = D(data_hat)

    data_dim = data_logit.shape[1]

    b_label = Variable(torch.ones(bs * data_dim).view(-1, data_dim), requires_grad=False).to(device)
    b_hat_label = Variable(torch.zeros(data_dim).view(-1, data_dim), requires_grad=False).to(device)

    dataloss = get_gan_loss(data_logit, b_label)
    data_hat_loss = get_gan_loss(data_hat_logit, b_hat_label)
    loss_GAN_a2b = (dataloss + data_hat_loss)
    return loss_GAN_a2b

def cal_D_loss(bs, data, data_hat, D, direction):
    data_logit = D(data)

    data_dim = data_logit.shape[1]

    data_label = Variable(torch.ones(bs * data_dim).view(-1, data_dim), requires_grad=False).to(device)
    data_hat_label = Variable(torch.zeros(data_dim).view(-1, data_dim), requires_grad=False).to(device)

    if direction == 'y':
        data_hat_from_buffer = y_hat_buffer.push_and_pop(data_hat)
        data_hat_logit = D(data_hat_from_buffer.detach())

        dataloss = get_gan_loss(data_logit, data_label)
        data_hat_loss = get_gan_loss(data_hat_logit, data_hat_label)
        d_gan_loss = dataloss + data_hat_loss

    else:
        data_hat_from_buffer = x_hat_buffer.push_and_pop(data_hat)
        data_hat_logit = D(data_hat_from_buffer.detach())

        dataloss = get_gan_loss(data_logit, data_label)
        data_hat_loss = get_gan_loss(data_hat_logit, data_hat_label)
        d_gan_loss = dataloss + data_hat_loss

    return d_gan_loss


# 计算cycle loss 即 cycle consistence
def cal_cycle_loss(data, data_recov):
    loss_cycle = get_cy_loss(data, data_recov)
    return loss_cycle


# 计算identity loss 其作用是保持色调
def cal_identity_loss(data, G):
    identity_loss = get_id_loss(G(data), data)
    return identity_loss


# 超参数优化的目标函数
def objective(trial):

    global best_score

    # 超参数设置
    n_critic = trial.suggest_int("n_critic", 2, 4, log=True)
    lamda_cy = trial.suggest_int("lamda_cy", 3, 15, log=True)
    lamda_id = trial.suggest_float("lamda_id", 0.1, 0.6, log=True)

    # ******************
    # *      初始化     *
    # ******************

    # 建立一个文件夹用来存放生成的图片
    os.makedirs(gen_dir, exist_ok=True)

    # 初始化得分
    score = []

    # 实例化生成器和鉴别器
    G = network.Generator(x_shape, n_res_blocks).to(device)     # 从x域中采样，生成y域的图片
    F = network.Generator(y_shape, n_res_blocks).to(device)     # 从y域中采样，生成x域的图片
    D_x = network.Discriminator().to(device)   # 鉴别x与F(y),并返回分数
    D_y = network.Discriminator().to(device)  # 鉴别y与G(x),并返回分数
    
    # 设置优化器
    optimizer_Gen = torch.optim.Adam(
        itertools.chain(G.parameters(), F.parameters()), lr=lr, betas=(b1, b2))
    optimizer_Dx = torch.optim.Adam(D_x.parameters(), lr=lr, betas=(b1, b2))
    optimizer_Dy = torch.optim.Adam(D_y.parameters(), lr=lr, betas=(b1, b2))

    # ******************
    # *     训练部分     *
    # ******************

    x_iterator = iter(xsetloader)
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = tqdm(ysetloader)
        progress_bar.set_description(f"Epoch {e + 1}")
        for steps, data in enumerate(progress_bar):

            G.train()
            imgs = data.to(device)
            bs = imgs.size(0)

            # 末尾剩余数据去除
            if bs != batchsize:
                break

            # 此法可同时采样x，y域的图像，且不会产生内存泄漏
            try:
                x = next(x_iterator).to(device)
            except StopIteration:
                x_iterator = iter(xsetloader)
                x = next(x_iterator).to(device)
            y = Variable(imgs).to(device)

            # *********************
            # *      训练生成器     *
            # *********************

            # 计算GAN loss
            y_hat = G(x).to(device)
            x_hat = F(y).to(device)

            loss_GAN = 0.5 * (cal_GAN_loss(bs, y, y_hat, D_y) +
                               cal_GAN_loss(bs, x, x_hat, D_x))

            # 计算cycle loss
            recov_x = F(y_hat).to(device)
            recov_y = G(x_hat).to(device)

            loss_cycle = 0.5 * (cal_cycle_loss(x, recov_x) +
                                 cal_cycle_loss(y, recov_y))

            # 计算identity loss
            loss_identity = 0.5 * (cal_identity_loss(y, G) +
                                    cal_identity_loss(x, F))

            loss_Gen = loss_GAN + loss_cycle*lamda_cy + loss_identity*lamda_id

            optimizer_Gen.zero_grad()
            loss_Gen.backward()
            optimizer_Gen.step()

            if steps % n_critic == 0:

                # *********************
                # *     训练鉴别器      *
                # *********************

                optimizer_Dx.zero_grad()
                optimizer_Dy.zero_grad()

                # 训练D_y
                loss_GAN_x2y = cal_D_loss(bs, y, y_hat, D_y, 'y')
                loss_GAN_x2y.backward()
                optimizer_Dy.step()

                # 训练D_x
                loss_GAN_y2x = cal_D_loss(bs, x, x_hat, D_x, 'x')
                loss_GAN_y2x.backward()
                optimizer_Dx.step()

            if steps % 10 == 0:
                progress_bar.set_postfix(loss_Dy=loss_GAN_x2y.item(), loss_Dx=loss_GAN_y2x.item(), loss_Gen=loss_Gen.item(), loss_gan=loss_GAN.item(), loss_id=loss_identity.item(), loss_cy=loss_GAN.item())

        generate_pics(G)

        # 每个epoch进行一次评估
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=os.path.join(workspace_dir, gen_dir),
            input2=os.path.join(dataset_dir, 'trainB'),
            cuda=True,
            isc=False,  # 因为isc有多样性的评判标准，我们这里只用到fid
            fid=True,
            kid=False,
            verbose=False,
        )
        # metrics_dict是字典结构，靠key访问其中的值
        fid_score = metrics_dict[torch_fidelity.KEY_METRIC_FID]
        score.append(fid_score)

    # 计算最终分数，此处权重可调
    final_score = 0.1*score[0] + 0.3*score[1] + 0.6*score[2]

    # 保存当前最优，用于后续训练
    if final_score < best_score:

        best_score = final_score

        torch.save(G.state_dict(), os.path.join(save_dir, f'G.pth'))
        torch.save(F.state_dict(), os.path.join(save_dir, f'F.pth'))
        torch.save(D_x.state_dict(), os.path.join(save_dir, f'Dx.pth'))
        torch.save(D_y.state_dict(), os.path.join(save_dir, f'Dy.pth'))

    return final_score


# main函数
if __name__ == "__main__":

    # 初始化数据集
    xset, yset = dataset.transform_dataset(os.path.join(dataset_dir, 'trainA'),
                                           os.path.join(dataset_dir, 'trainB'))

    xsetloader = DataLoader(xset, batch_size=batchsize, shuffle=True)
    ysetloader = DataLoader(yset, batch_size=batchsize, shuffle=True)

    # 设置图片缓存区
    x_hat_buffer = dataset.ReplayBuffer()
    y_hat_buffer = dataset.ReplayBuffer()

    # 实例化study，这是optuna中管定义理优化，决定优化的方式
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=800)

    # 优化架构，可以根据自己的需求改动
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    pass
