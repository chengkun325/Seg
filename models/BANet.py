""" BANet
Paper: ``
    - https://arxiv.org/abs/2106.12413
ResT code and weights: https://github.com/wofmanaf/ResT
"""
import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter
from torchvision import models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    多层感知机
    该MLP包含两个全连接层（fc1和fc2），一个激活函数（act_layer），一个Dropout层（drop）用于在训练过程中随机删除一些神经元，
    以防止过拟合。

    forward()函数执行MLP的前向传递，通过fc1和fc2的全连接操作，将输入x传递到隐藏层和输出层
    并使用激活函数（act_layer）进行非线性变换。
    最后，Dropout层用于随机删除一些神经元。最终输出大小为out_features的张量。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    多头注意力机制
    该模块接受一个形状为 (batch_size, sequence_length, hidden_size)的输入张量，并将其拆分成多个头，
    每个头都使用自己的查询、键和值进行注意力计算，最后将所有头的输出进行合并和处理。
    具体来说，该模块包含以下组件：
    num_heads: 注意力头的数量。
    qkv_bias: 是否在查询、键和值中使用偏置。
    qk_scale: 缩放查询和键的比例因子。默认值为头尺寸的倒数的平方根。
    attn_drop: 注意力权重的 dropout 概率。
    proj_drop: 输出的 dropout 概率。
    sr_ratio: 缩小尺寸比率，如果大于 1，则使用一个卷积层将输入的高和宽都缩小到，再进行注意力计算。默认值为 1，即不进行缩小。
    apply_transform: 是否对注意力计算结果应用 1x1 卷积和层归一化，以增强特征表示能力。
    该模块的 forward 方法接受三个参数：
    输入张量 x 的形状为 (batch_size, sequence_length, hidden_size)，以及输入张量的高度 H 和宽度 W。
    在 forward 方法中，首先将 x 做线性变换，得到查询 q。
    然后根据 sr_ratio 是否大于 1 来确定是使用原始的 x 进行注意力计算，还是对 x 进行尺寸缩小后再进行注意力计算。
    接着将键值对进行线性变换，得到键 k 和值 v。然后计算注意力权重并进行 dropout。
    最后将注意力权重乘以值 v 并进行转置、合并和处理得到最终输出。注意，在 apply_transform 为 True 时，
    还会对注意力计算结果进行一次卷积和层归一化。
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    这是一个Transformer中的Block，包含了一个 Attention 和一个 Feed-Forward 网络（Mlp）。
    其中Attention中使用了多头注意力机制，支持空间降采样（sr_ratio>1）和头之间的转换（apply_transform=True）。
    Feed-Forward网络中包含一个线性层和一个激活函数（默认为GELU）以及一个dropout层。
    同时，该Block还支持Drop Path技术用于随机深度剪枝。norm_layer用于进行归一化处理，常用的是LayerNorm。
    在Forward时需要传入输入x的高度H和宽度W。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    """
    位置编码模块 Position Attention
    PA模块的作用是在卷积层中为每个位置的通道信息引入位置注意力机制。通过独立处理每个通道，模块可以学习到每个位置的语义信息，
    并利用激活函数和元素逐元素相乘的方式对输入张量进行位置编码，以更好地提取图像特征。
    """

    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class GL(nn.Module):
    """
    GL 它实现了一个具有通道组卷积的残差连接。
    该模块的构造函数接受一个整数参数 dim，表示输入和输出张量的通道数。
    模块包含一个Conv2d层，其输入和输出通道数均为dim，卷积核大小为3x3，填充为1，卷积分组数为dim，即对输入的每个通道分别进行卷积。
    在前向传递中，模块将输入张量x传递给Conv2d层进行卷积，并将卷积输出加回到输入张量中，从而实现了残差连接的效果。
    最终输出残差连接后的结果。
    """

    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)


class PatchEmbed(nn.Module):
    """
    PatchEmbed模块是将输入的图像按照指定的 patch_size 分割成若干个块，并将每个块进行卷积和标准化，最后展平成一个向量。
    如果with_pos参数为True，则在每个块的向量上加上其位置编码。函数的输出包含了每个块的向量以及图像被划分成的块的行数和列数。
    """

    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        # 从第 2 维展平，将 x 从一个形状为[B, C, H, W]的四维张量展平为一个形状为[B, C, HW]的三维张量。
        # 然后将张量的第一维和第二维进行转置，变为[B, HW, C]。注：维数从0开始。
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class BasicStem(nn.Module):
    """
    Stem Block

    目的是缩小高宽尺寸、扩大通道尺寸。为了有效捕获低层次信息，引入了 3 个 3 × 3 卷积层，步长为 [2, 1, 2]。
    前两个卷积层后面是 BN 和 ReLU。
    从而将空间分辨率降为原来的 4 倍，信道维数由原来的 3 维扩展到 64 维。
    """

    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x

# ResT 论文中这个地方还有一个Stem类，好像没用到。

class ResT(nn.Module):
    """
    ResT 它实现了一个由四个阶段组成的 ViT 模型。
    其输入张量的通道数为3，输出分类数为 6。模型的特征维度分别为[64, 128, 256, 512]，
    每个阶段的 Transformer 块的数量为 [2, 2, 2, 2]。在每个阶段中，使用了带有不同头数的多头注意力机制，
    以及不同的 MLP 层来处理特征。同时，每个阶段的 Transformer 块都应用了位置嵌入和位置编码来加入空间信息。
    在每个阶段中，都使用了一个 PatchEmbed 操作来将图像分块，并将每个块嵌入到特征空间中。
    此外，还应用了一些正则化技术，如 Dropout 和 LayerNorm，以避免过拟合。最后，该模型输出了第三个和第四个阶段的特征张量。
    """

    def __init__(self, in_chans=3, num_classes=6, embed_dims=None,
                 num_heads=None, mlp_ratios=None, qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 depths=None, sr_ratios=None,
                 norm_layer=nn.LayerNorm, apply_transform=False):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if depths is None:
            depths = [2, 2, 2, 2]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        self.num_classes = num_classes
        self.depths = depths
        self.apply_transform = apply_transform

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])

        self.norm = norm_layer(embed_dims[3])

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        # x1 = x

        # stage 2
        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        # x2 = x

        # stage 3
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x3 = x

        # stage 4
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x4 = x

        return x3, x4

#改了大的
def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_small.pth', **kwargs):
    """
    这段代码定义了一个名为rest_lite的函数，如果参数pretrained为True，就会从路径weight_path加载预训练的模型权重。
    函数返回一个ResT模型的实例，该模型具有一些参数：embed_dims，num_heads，mlp_ratios，qkv_bias，depths，sr_ratios和apply_transform等。

    ResT模型是一个用于图像分类的深度学习模型，它的结构类似于Transformer，但是它使用了一种叫做Spatial Reduction的技术来减小输入图像的分辨率。
    它的参数含义如下：

    embed_dims：每个Transformer块中嵌入向量的维度。
    num_heads：每个Transformer块中自注意力机制头的数量。
    mlp_ratios：每个Transformer块中多层感知机中隐藏层和嵌入层之间维度比率。
    qkv_bias：控制Transformer块中是否应该添加偏置项。
    depths：ResT模型中每个阶段中Transformer块的数量。
    sr_ratios：每个阶段中图像分辨率减小的比率。
    apply_transform：控制ResT模型是否应该对输入图像进行转换（缩放，翻转，旋转等）来进行数据增强。
    这个函数的返回值是一个ResT模型的实例。如果pretrained参数为True，则会从weight_path加载预训练的模型权重。
    这些权重会被加载到模型实例的状态字典中，并返回带有这些预训练权重的模型实例。
    """
    model = ResT(embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


class ConvBNReLU(nn.Module):
    """
    该模块包括一个 2D 卷积层、一个批标准化层和一个 ReLU 激活函数。
    其初始化函数的参数包括输入通道数、输出通道数、卷积核大小、步长和填充大小。
    该模块的前向传递函数对输入进行卷积、批标准化和 ReLU 激活，并返回输出。
    此外，该类还实现了一个初始化函数，用于初始化卷积层权重和偏置，以便更好地训练模型。
    """

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


def l2_norm(x):
    """
    这段代码实现了 L2 归一化的功能。给定一个张量 x，它的形状为 [batch_size, channels, length]，
    其中 batch_size 表示批大小，channels 表示通道数，length 表示张量的长度。
    在进行归一化时，对于 x 中的每个样本，都将它的 L2 范数计算出来，然后将该样本中每个元素除以该范数，即实现了对该样本的 L2 归一化。
    最后返回归一化后的张量。具体实现是使用了 PyTorch 中的 einsum 函数，对张量进行了矩阵乘法和广播操作。
    """
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class LinearAttention(Module):
    """
    这段代码实现了一个 Linear Attention 模块，该模块接收一个输入特征图 x，通过三个卷积层生成对应的 queries、keys 和 values，
    使用 l2_norm 对 queries 和 keys 进行归一化，计算得到注意力矩阵，将注意力矩阵乘以 values 得到加权和，
    最终输出加权和和输入特征图的和，通过调节 gamma 可以控制加权和对输入特征图的影响程度。
    其中，tailor_sum 和 matrix_sum 是计算注意力矩阵的中间变量，eps 是防止分母为 0 的小值。
    """

    def __init__(self, in_places, scale=8, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return x + (self.gamma * weight_value).contiguous()


class Output(nn.Module):
    """
    这是一个输出层的模块，用于将模型提取的特征进行输出，通常用于分类或分割等任务中。

    模块包含一个卷积层和一个像素洗牌层（PixelShuffle），中间还包含一个中间层的卷积层，用于降低通道数。

    构造函数中的参数说明如下：

    in_chan：输入特征图的通道数。
    mid_chan：中间层卷积层的通道数。
    n_classes：分类或分割的类别数。
    up_factor：上采样因子，用于进行像素洗牌，将特征图的分辨率放大。默认为32。
    该模块的forward函数中，先经过中间层的卷积层，然后再经过一个输出层的卷积层，最后通过像素洗牌层进行上采样。返回的是上采样后的特征图。

    该模块还有一个get_params函数，用于获取需要进行权重衰减（weight decay）的参数和不需要进行权重衰减的参数，
    分别存储在wd_params和nowd_params两个列表中。其中需要进行权重衰减的参数包括卷积层和线性层的权重，需要进行正则化。
    不需要进行权重衰减的参数包括卷积层和线性层的偏置和批归一化层的参数，不需要进行正则化。
    """

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32):
        super(Output, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes * up_factor * up_factor
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.PixelShuffle(up_factor)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class UpSample(nn.Module):
    """
    该代码定义了一个上采样模块类UpSample，用于将输入的 feature map 进行上采样。
    该模块包含两个子模块，分别为 proj 和 up。
    其中，proj 为卷积层，用于将输入 feature map 的通道数从 n_chan 扩展为 n_chan * factor * factor。
    up 为像素混洗层，将 feature map 上采样 factor 倍，即将 feature map 中每个像素周围插入 factor - 1 个零值像素，
    然后重排为原来的 1/factor，从而扩大 feature map 的尺寸。

    在 forward 函数中，输入 feature map x 经过 proj 层后得到 feat，然后将 feat 经过 up 层进行上采样，
    最终返回上采样后的 feature map。

    init_weight 函数用于初始化 UpSample 中的参数。其中，proj 层的权重采用了 Xavier 初始化方法。
    """

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class Attention_Embedding(nn.Module):
    """
        这是一个用于实现注意力嵌入的PyTorch模块，输入包括一个高层特征和一个低层特征。
        首先，通过调用 LinearAttention 模块，使用注意力机制计算高层特征的注意力矩阵。
        接下来，通过 ConvBNReLU 模块进行卷积和批量归一化，将注意力矩阵转换为与低层特征相同的通道数。
        最后，使用 UpSample 模块将通道数进行上采样，并将其与低层特征相乘，形成注意力嵌入。最终输出结果为低层特征与注意力嵌入相加的结果。
    """

    def __init__(self, in_channels, out_channels):
        super(Attention_Embedding, self).__init__()
        self.attention = LinearAttention(in_channels)
        self.conv_attn = ConvBNReLU(in_channels, out_channels)
        self.up = UpSample(out_channels)

    def forward(self, high_feat, low_feat):
        A = self.attention(high_feat)
        A = self.conv_attn(A)
        A = self.up(A)

        output = low_feat * A
        output += low_feat

        return output


class FeatureAggregationModule(nn.Module):
    """
    特征聚合模块
    __init__(self, in_chan, out_chan): 初始化方法，接收输入通道数in_chan和输出通道数out_chan作为参数。
    在该方法中，创建了两个子模块 convblk 和 conv_atten，分别用于卷积、归一化和激活操作，以及线性注意力操作。
    还调用了init_weight()方法进行权重的初始化。

    forward(self, fsp, fcp): 前向传播方法，接收两个输入张量 fsp 和 fcp，表示特征张量。
    在该方法中，首先将两个特征张量按通道维度进行拼接，然后通过 convblk 进行卷积操作得到特征张量 feat。
    接下来，通过 conv_atten 对 feat 进行线性注意力操作，得到注意力张量 atten。
    最后，将 feat 和 atten 进行逐元素相乘得到加权特征张量 feat_atten，并与原始特征张量 feat 相加得到最终的输出特征张量 feat_out。

    init_weight(self): 权重初始化方法，通过遍历模块的子模块，对其中的卷积层进行权重初始化，
    使用了 Kaiming 正态分布初始化方法，并对偏置进行常数初始化。

    get_params(self): 获取模型参数方法，返回两个列表 wd_params 和 nowd_params。wd_params 包含需要进行权重
    """

    def __init__(self, in_channel, out_channel):
        super(FeatureAggregationModule, self).__init__()
        self.convblk = ConvBNReLU(in_channel, out_channel, ks=1, stride=1, padding=0)
        self.conv_atten = LinearAttention(out_channel)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class TexturePath(nn.Module):
    def __init__(self):
        super(TexturePath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class DependencyPath(nn.Module):
    """
    DependencyPath是一个继承自nn.Module的类，表示依赖路径。其初始化函数中定义了三个子模块：

    ResT: 调用rest_lite模型，对输入x进行处理得到两个特征图e3和e4。
    AE: Attention_Embedding模块，将e4和e3作为输入，生成对应的嵌入特征图。
    conv_avg: 一个1x1卷积，将嵌入特征图通道数从256降到128。

    在forward函数中，将e4和e3作为输入，通过AE模块生成嵌入特征图e，然后通过conv_avg对嵌入特征图进行处理，
    并最终通过双线性插值上采样2倍输出结果。
    get_params函数返回所有需要权重衰减和不需要权重衰减的参数列表，以便在训练过程中进行参数更新。
    """

    def __init__(self, weight_path='pretrain_weights/rest_small.pth'):
        super(DependencyPath, self).__init__()
        self.ResT = rest_lite(weight_path=weight_path)
        self.AE = Attention_Embedding(512, 256)
        self.conv_avg = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2.)

    def forward(self, x):
        e3, e4 = self.ResT(x)

        e = self.conv_avg(self.AE(e4, e3))

        return self.up(e)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class DependencyPathRes(nn.Module):
    """
    该类实现了一个基于ResNet18的深度神经网络，用于学习图像数据的依赖路径。具体而言，该网络包括以下层：

    第一层：包括ResNet18的conv1卷积层、bn1批量归一化层、relu激活层和maxpool池化层。
    第二层到第五层：包括ResNet18的4个残差块layer1到layer4。
    AE层：实例化Attention_Embedding类，用于将高级特征和低级特征相乘并进行上采样操作。
    conv_avg层：使用ConvBNReLU类实现的卷积层，将256个通道的输出转换为128个通道。
    up层：上采样层，将特征图的大小放大2倍。
    forward方法中，首先对输入数据进行第一层到第五层的卷积和池化操作，得到4个级别的特征图。
    然后将高级特征e4和低级特征e3送入AE层中相乘并上采样，得到128个通道的特征图，最后通过上采样将特征图的大小放大2倍。
    get_params方法返回需要进行权重衰减（weight decay）和不需要进行权重衰减的参数列表。
    """

    def __init__(self):
        super(DependencyPathRes, self).__init__()
        resnet = models.resnet18(True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.AE = Attention_Embedding(512, 256)
        self.conv_avg = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2.)

    def forward(self, x):
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e = self.conv_avg(self.AE(e4, e3))

        return self.up(e)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BANet(nn.Module):
    """
    DependencyPath: 输入图像，经过ResT-lite网络进行特征提取，输出特征图像。
    TexturePath: 输入图像，经过卷积神经网络进行特征提取，输出特征图像。
    FeatureAggregationModule: 输入特征图像，通过注意力机制将两个特征图融合在一起。
    Output: 输入特征图像，通过卷积神经网络进行特征提取和上采样，最终输出6类分类概率值。
    其中，DependencyPath和TexturePath模块分别用于提取图像的不同特征信息，
    FeatureAggregationModule模块用于将两种特征信息融合在一起，Output模块用于分类输出。

    这个模型的初始化权重由init_weight()函数进行，训练时可以使用get_params()函数获取所有可训练参数的权重和偏差。
    """

    def __init__(self, num_classes=6, weight_path='pretrain_weights/rest_small.pth'):
        # 调用父类（也就是nn.Module）的初始化方法。
        super(BANet, self).__init__()
        self.name = 'BANet'
        self.cp = DependencyPath(weight_path=weight_path)
        self.sp = TexturePath()
        self.fam = FeatureAggregationModule(256, 256)
        self.conv_out = Output(256, 256, num_classes, up_factor=8)
        self.init_weight()

    def forward(self, x):
        feat = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.fam(feat_sp, feat)
        feat_out = self.conv_out(feat_fuse)
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        # 初始化四个参数列表。
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        # 遍历所有带名称的子模块。
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            # 如果子模块是 Attention_Embedding 或 Output 类型
            if isinstance(child, (Attention_Embedding, Output)):
                # 将参数添加到 lr_mul_wd_params 列表
                lr_mul_wd_params += child_wd_params
                # 将参数添加到 lr_mul_nowd_params 列表
                lr_mul_nowd_params += child_nowd_params
            else:
                # 将参数添加到 wd_params 列表
                wd_params += child_wd_params
                # 将参数添加到 nowd_params 列表
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
