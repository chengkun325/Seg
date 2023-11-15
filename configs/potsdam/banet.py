from torch.utils.data import DataLoader
from models.BANet import BANet
from datasets.potsdam_dataset import *
from losses.banet_loss import BANetLoss
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training Hyper-parameters
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
num_classes = len(CLASSES)
classes = CLASSES
lr = 3e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1

monitor = 'val_OA'
monitor_mode = 'max'
weights_name = "banet-resT"
weights_path = "model_weights/potsdam/{}".format(weights_name)
log_name = 'potsdam/{}'.format(weights_name)

pretrained_ckpt_path = None
resume_ckpt_path = None
use_aux_loss = False


# define the network
net = BANet(num_classes=num_classes)


# define the loss
loss = BANetLoss(ignore_index=ignore_index)

# define the dataloader
train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=8,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
#余弦退火和热重启（warm restarts）的学习率调整策略。
#T_0 一个整数，表示第一个重启周期的长度（即余弦周期的长度）。
#T_mult 一个整数，表示每个重启周期后周期长度的倍数增加。
#例如，如果T_0=15，T_mult=2，则第二个周期的长度将是第一个周期长度的两倍。
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

