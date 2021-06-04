import argparse
import os
from PIL import Image

import torch
from torchvision import transforms
import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', default='div2k')
    parser.add_argument('--model')
    #parser.add_argument('--resolution')
    #parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save_path', default='./save_picture')
    args = parser.parse_args()
    # 以下是输入数据集的路径
    args.input_path = './load/div2k/DIV2K_valid_LR_bicubic/X4'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    for i in range(100):
        # 输入途径
        input_path = args.input_path + '/0' + str(801+i) + 'x4.png'
        # 输出途径
        output_path = args.save_path + '/0' + str(801+i) +'_'+ args.model.split('/')[-2][len('_train_edsr-baseline-'):] +'.png'
        img = transforms.ToTensor()(Image.open(input_path).convert('RGB'))
        w, h = Image.open(input_path).size
        h = 4*h
        w = 4*w
        #h, w = list(map(int, args.resolution.split(',')))
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        transforms.ToPILImage()(pred).save(output_path)
        print('photo: {}'.format(801+i))
