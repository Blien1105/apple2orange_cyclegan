import torch
import dataset
import train_and_valid
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_is_train_ = False

# 主函数
if __name__ == '__main__':
    trainer = train_and_valid.Trainer(device)  # 实例化训练器
    if _is_train_:
        dataset.set_same_seeds(2333)
        trainer.prepare_environment()
        trainer.train()
        pass
    else:
        trainer.prepare_environment()
        trainer.load_trained_model()
        trainer.generate_pics('xtype')
        trainer.validation()

    os.system("/usr/bin/shutdown")
    pass
