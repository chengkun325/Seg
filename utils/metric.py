import numpy as np


class Evaluator(object):
    """
    评估分类结果
    """
    def __init__(self, num_class):
        """
        初始化，设置分类的数量和初始化混淆矩阵。
        :param num_class:
        """
        self.num_class = num_class
        # 创建一个包含2个元素的元组，并通过 np.zeros() 初始化为0。
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        """
        P/N是预测结果的正负， T/F是对预测结果正确与否的判断。也就是说：
        TP是判断为正例(P), 判断是正确的(T) (也就是实际就是正例)；
        TN是判断为负例(N)， 判断是正确的(T) (也就是实际就是负例);
        FP是判断为正例(P)， 判断是错误的(F) (也就是实际是负例);
        FN是判断为负例(N)， 判断是错误的(F) (也就是实际是正例);

        TP（真正例）计算了混淆矩阵的对角线元素，即每个类别被正确分类的样本数。
        FP（假正例）计算了每个类别的列总和减去对角线元素，表示被错误分类为该类别的样本数。
        FN（假负例）计算了每个类别的行总和减去对角线元素，表示被错误分类为其他类别的样本数。
        TN（真负例）计算了混淆矩阵对角线元素的总和减去对角线元素，表示被正确分类为其他类别的样本数。
        最后，该方法返回计算得到的tp、fp、tn和fn值，以元组的形式返回。
        :return:
        """
        # 当给定一个二维数组作为输入时，会提取对角线元素并返回一个一维数组。
        # 当给定一个一维数组作为输入时，会使用该数组的元素创建一个对角矩阵。
        tp = np.diag(self.confusion_matrix)
        # axis=0 表示沿着行方向求和，即求每一列的和。
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        # axis=1 表示沿着列方向求和，即求每一行的和。
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        """
        准确率：它是指被正确预测为正例的样本数（TP）与所有被预测为正例的样本数（TP + FP）之间的比值。
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        """
        召回率：它是指被正确预测为正例的样本数（TP）与所有真实正例样本数（TP + FN）之间的比值。
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        """
        F1 score：它是精确度（Precision）和召回率（Recall）的调和平均。
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        """
        整体准确度（Overall Accuracy），即所有被正确分类的样本数除以总样本数。
        """
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        """
        交并比，也称为Jaccard相似度。它衡量了预测结果与真实结果之间的重叠程度。
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        """
        Dice系数：它衡量了预测结果与真实结果之间的相似度
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        """
        每个类别的像素准确度（Pixel Accuracy）。
        以数组的形式返回，每个数组元素表示对应类别的像素准确度。
        """
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        频权交并比（Frequency Weighted Intersection over Union）。
        它结合了每个类别的频率（Frequency）和对应的交并比（Intersection over Union）。

        首先，使用np.sum(self.confusion_matrix, axis=1)对混淆矩阵的行进行求和，得到每个类别的像素总数。
        然后，将每个类别的像素总数除以混淆矩阵中所有元素的总和（即总像素数），加上一个很小的数self.eps，以避免除以零的情况。
        这样得到每个类别的频率（Frequency）。
        接下来，调用self.Intersection_over_Union()方法计算交并比（Intersection over Union）。
        然后，选择频率大于0的类别，并将对应的频率乘以交并比，得到加权的交并比。
        最后，将加权的交并比求和，得到频权交并比（Frequency Weighted Intersection over Union）。
        最后，该方法返回计算得到的频权交并比值。
        """
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """
        首先，创建一个布尔掩码（mask），通过判断真实标签图像的像素值是否在合理的范围内（大于等于0且小于类别数）来确定有效的像素。
        接下来，将布尔掩码应用于真实标签图像和预测标签图像，获取有效像素的标签值。
        然后，通过将真实标签值乘以类别数，并加上对应的预测标签值，将标签值映射为混淆矩阵的索引。
        使用np.bincount函数统计每个索引值的出现次数，得到一个包含各类别对应索引的计数数组。
        将计数数组重塑为一个二维矩阵，形状为（类别数，类别数），得到混淆矩阵。
        最后，将生成的混淆矩阵返回。
        矩阵的每个元素表示分类器将某个类别的样本预测为另一个类别的数量。
        """
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """
        将一批（batch）的真实标签图像和预测标签图像添加到混淆矩阵中。
        """
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [2, 0, 1],
                   [1, 1, 1]])

    eval = Evaluator(num_class=3)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.Precision())
    print(eval.Recall())
    print(eval.Intersection_over_Union())
    print(eval.OA())
    print(eval.F1())
    print(eval.Frequency_Weighted_Intersection_over_Union())
