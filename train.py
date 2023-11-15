import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from utils.cfg import py2cfg
import os
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
from utils.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    """
    argparse 是 Python 标准库中的一个命令行解析工具，用于解析命令行参数。
    argparse.ArgumentParser()创建一个 ArgumentParser 对象，它用于定义命令行参数的选项和参数，解析它们并生成使用帮助信息。
    ArgumentParser 对象可以通过调用 add_argument() 方法来添加命令行选项和参数。
    在脚本中使用 argparse 可以使命令行工具的参数处理变得简单明了，方便用户操作。
    :return:
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.automatic_optimization = False

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        """
        在 PyTorch-Lightning 框架中，training_step函数定义了每个训练步骤的操作。
        函数接受两个参数 batch 和 batch_idx ，分别表示一个批次的输入数据和批次的索引。
        在函数中，从输入批次中获取图像数据img和掩膜（或标签）数据mask。
        接下来，将图像数据 img 输入到模型 self.net 中，得到预测结果prediction。
        然后，使用定义的损失函数 self.loss 计算预测结果 prediction 和真实标签 mask 之间的损失值 loss。
        如果配置中指定了使用辅助损失（use_aux_loss=True），则从预测结果中取出第一个通道，并使用 nn.Softmax(dim=1)进行软最大化（Softmax）处理。
        否则，直接对预测结果进行软最大化处理。
        接着，对软最大化处理后的预测结果进行维度上的操作，使用argmax(dim=1)获得每个像素点最可能的类别索引。
        然后，通过循环遍历批次中的每个样本，将真实标签和预测结果的类别索引传递给训练过程中的评估指标 self.metrics_train 进行更新。
        在监督训练阶段（supervision stage）中，获取优化器 opt，通过调用self.optimizers(use_pl_optimizer=False)进行获取。
        然后，使用self.manual_backward(loss)进行反向传播计算梯度。
        如果达到了累积梯度的步数（accumulate_n）的倍数（(batch_idx + 1) % self.config.accumulate_n == 0），则调用opt.step()执行优化器的梯度更新，
        并使用opt.zero_grad()清零梯度。
        接下来，获取学习率调度器sch，通过调用self.lr_schedulers()进行获取。
        如果当前批次是最后一个批次，并且当前训练的轮数（current_epoch）是1的倍数（(self.trainer.current_epoch + 1) % 1 == 0），则调用sch.step()更新学习率。
        最后，返回一个包含损失值的字典，形如{"loss": loss}，供训练过程进行统计和日志记录。
        :param batch:
        :param batch_idx:
        :return:
        """
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        # Batch, ClassNum, H, W
        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # supervision stage
        opt = self.optimizers(use_pl_optimizer=False)
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.accumulate_n == 0:
            opt.step()
            opt.zero_grad()

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """
        这段代码是训练过程中的一个训练周期结束函数training_epoch_end，用于在一个训练周期结束时进行一些计算和日志记录。

        函数接受一个参数outputs，表示在一个训练周期中所有训练步骤的输出。

        在函数中，根据配置中的log_name的不同，计算不同数据集的指标。
        根据数据集名称，分别计算mIoU（平均交并比）、F1和OA（整体精度）指标。这些指标是通过调用训练过程中的评估指标对象self.metrics_train的相应方法来计算的。

        接下来，根据不同数据集的名称，将计算得到的指标值存储在相应的变量 mIoU、F1 和 OA 中。

        然后，使用np.nanmean函数计算交并比、F1和OA的平均值，并将其存储在eval_value字典中。

        通过print语句打印出计算得到的指标值。

        接下来，构建一个字典iou_value，将每个类别的交并比指标存储在其中。

        通过调用self.metrics_train.reset()重置评估指标对象，以便在下一个训练周期中重新计算指标。

        接下来，计算所有训练步骤的损失值的平均值，并将其存储在loss变量中。

        构建一个日志字典log_dict，包含训练损失、mIoU、F1和OA的值。

        最后，通过self.log_dict方法将日志字典log_dict中的值记录到训练过程的日志中，并使用prog_bar=True显示在进度条中。
        :param outputs:
        :return:
        """
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log_dict = {"train_loss": loss, 'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """
        用于在验证集上执行模型的前向传播和评估。

        函数接受两个参数batch和batch_idx，分别表示一个批次的输入数据和批次的索引。

        在函数中，从输入批次中获取图像数据img和掩膜（或标签）数据mask。

        接下来，将图像数据img输入到模型的前向传播方法self.forward中，得到预测结果prediction。

        然后，对预测结果进行软最大化处理，使用nn.Softmax(dim=1)对预测结果进行处理，使得每个像素点的预测值变成对应类别的概率值。
        接着，通过argmax(dim=1)获取每个像素点最可能的类别索引。

        接下来，通过循环遍历批次中的每个样本，将真实标签和预测结果的类别索引传递给验证过程中的评估指标self.metrics_val进行更新。

        接着，使用定义的损失函数self.loss计算预测结果prediction和真实标签mask之间的损失值loss_val。

        最后，返回一个包含验证损失值的字典，形如{"loss_val": loss_val}，供训练过程进行统计和日志记录。
        :param batch:
        :param batch_idx:
        :return:
        """
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def validation_epoch_end(self, outputs):
        """
        用于在一个验证周期结束时进行一些计算和日志记录。

        函数接受一个参数outputs，表示在一个验证周期中所有验证步骤的输出。

        在函数中，根据配置中的 log_name 的不同，计算不同数据集的指标。根
        据数据集名称，分别计算mIoU（平均交并比）、F1和OA（整体精度）指标。这些指标是通过调用验证过程中的评估指标对象self.metrics_val的相应方法来计算的。

        接下来，根据不同数据集的名称，将计算得到的指标值存储在相应的变量mIoU、F1和OA中。

        然后，使用np.nanmean函数计算交并比、F1和OA的平均值，并将其存储在eval_value字典中。

        通过print语句打印出计算得到的指标值。

        接下来，构建一个字典iou_value，将每个类别的交并比指标存储在其中。

        通过调用self.metrics_val.reset()重置验证指标对象，以便在下一个验证周期中重新计算指标。

        接下来，计算所有验证步骤的损失值的平均值，并将其存储在loss变量中。

        构建一个日志字典log_dict，包含验证损失、mIoU、F1和OA的值。

        最后，通过self.log_dict方法将日志字典log_dict中的值记录到训练过程的日志中，并使用prog_bar=True显示在进度条中。
        :param outputs:
        :return:
        """
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        log_dict = {"val_loss": loss, 'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        """
        这段代码是用于配置优化器和学习率调度器的函数configure_optimizers。

        函数中首先获取配置中的优化器optimizer和学习率调度器lr_scheduler。

        然后，将优化器和学习率调度器作为列表的形式返回。

        在这个函数中，返回的是包含单个优化器的列表[optimizer]和包含单个学习率调度器的列表[lr_scheduler]。

        这样，在训练过程中，就会使用指定的优化器和学习率调度器来进行模型参数的优化和学习率的调整。
        :return:
        """
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    # 这段代码定义了一个 ModelCheckpoint 回调函数，该回调函数用于在训练过程中保存模型的权重。回调函数的参数如下：
    #
    # save_top_k：保存最好的 k 个模型权重文件。
    # monitor：监视器名称，用于确定模型性能的度量指标。
    # save_last：是否保存最后一个训练时期的模型权重。
    # mode：确定监视器最佳值的计算方式，可以是 min、max 或 auto。
    # dirpath：指定模型权重文件保存的路径。
    # filename：指定模型权重文件的名称。
    # 具体来说，该回调函数在每个训练时期结束时检查监视器的度量指标，并保存最好的 k 个模型权重文件和最后一个训练时期的模型权重文件。
    # 通过使用回调函数，我们可以自动保存模型，而无需手动编写代码来保存模型。
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    early_stop_callback = EarlyStopping(
        monitor='val_F1',  # 监控的指标，可以是训练过程中任意一个指标，如 'val_loss'、'val_accuracy' 等
        min_delta=0.0,       # 指标改善的最小差值，当指标的改善小于该值时，被认为没有进一步改善，触发早停
        patience=10,         # 持续多少个训练周期（epochs）没有改善时触发早停
        verbose=True,        # 是否打印早停信息
        mode='max'
    )


    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback,early_stop_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger)

    trainer.fit(model=model)


if __name__ == "__main__":
    main()
    