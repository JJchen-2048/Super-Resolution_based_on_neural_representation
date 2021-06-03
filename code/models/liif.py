import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

# 暗示使用频繁，将其保存在cpu的寄存器中
"""
train_edsr_baseline里面的参数
args里面就是输入给LIIF的参数
  args:
    encoder_spec:
      name: edsr-baseline
      args:
        no_upsampling: true
    imnet_spec:
      name: mlp
      args:
        out_dim: 3
        hidden_list: [256, 256, 256, 256]
"""
# 暗示使用频繁，将其保存在cpu的寄存器中
@register('liif')
class LIIF(nn.Module):
    # encoder_spec是LIIF前面那个编码器的对应参数
    # imnet_spec是LIIF内部的网络参数
    # imnet_spec=None相当于就是没有用LIIF
    # 两个参数的举例如下
    """
    encoder_spec:
      name: edsr-baseline
      args:
        no_upsampling: true
    imnet_spec:
      name: mlp
      args:
        out_dim: 3
        hidden_list: [256, 256, 256, 256]
    """
    # 里面几个参数对应文章里面几个功能
    # feat_unfold 特征展开，就是一块变成附近3*3块的结合
    # local_ensemble 中心预测，一个点的预测值结合附近的上下左右四块的预测的加权
    # cell_decoding 是否考虑输入预测的像素长宽作为预测值
    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        # 建立encorder的模型
        self.encoder = models.make(encoder_spec)

        # 如果用了LIIF，下面是对LIIF的网络结构设置
        if imnet_spec is not None:
            # encoder是进行过初步处理的特征地图
            # encoder的输出是LIIF的输入，所以要求大小一样
            imnet_in_dim = self.encoder.out_dim
            # 如果有特征展开，相当于输入了原本3*3的特征地图
            if self.feat_unfold:
                imnet_in_dim *= 9
            # 输入的变量还有 二维坐标
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                # 用了cell_decode的话，输入的变量还有像素的高宽
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        # 输入encoder，输入图像变成地图
        self.feat = self.encoder(inp)
        # 这里输出的是16*64*48*48
        # print(self.feat.shape)
        return self.feat

    # 问询rgb值
    def query_rgb(self, coord, cell=None):
        # 特征图
        feat = self.feat

        if self.imnet is None:
            # 这个是将，图像的像素对应到坐标上去，把像素整到想要的坐标，得到最终输出
            # unsqueeze在第二维增加一个维度
            # 这里对输出进行了一个翻转，为的是最后输出是(N,W,H,C)
            # 不过为什么下面加了0？是因为只要第一行的内容吗，这玩意儿是舍弃了一个维度吗
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            # unfold从批量的输入张量中提取滑动局部块
            # 这里对应步骤unfold，重新做特征图，unfold输出是一个三维的，第二维内容为每个核中的内容，3*3*3，第三维内容对应滑的次数
            # 现在每个点是与附近九个点的叠加
            # padding是填充，为了地图前后不变
            # unfold操作在input的spatial dimension种滑动一个大小为kernel size的window
            # 并将一个window内的值展平，变为一个三维向量的一列。三维向量的大小为(N,C*卷积核大小，L)
            # 注意：输入的向量为1*64*H*W
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # 对应liif的那个中心功能，就是每个点是附近中心的加权和
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        # 每一小格对应的距离
        # 2/长度，对应的是每一小格子对应的
        # 由于范围是[-1,1],所以再除以2
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # .shape[-2:]输出(h,w)
        # make_coord 后面输出的是[长，宽，维度]
        # permute调整把长宽调到最后两个
        # expand返回tensor的一个新视图，单个维度扩大为更大的尺寸,这里相当于是给每个图，建立一个坐标
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        preds = []
        areas = []

        ###################################################
        # 主要是看以下代码
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                # print('1',coord_.shape)
                # 找到区域横坐标范围
                coord_[:, :, 0] += vx * rx + eps_shift
                # 找到区域纵坐标范围
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # print('2',q_feat.shape)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # print('5',q_coord.shape)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                # print('3', inp.shape)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                # 除了长宽以外的两个参数
                bs, q = coord.shape[:2]
                # 然后还原成一列
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
