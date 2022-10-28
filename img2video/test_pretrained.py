"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--dataset', default='Datasets', type=str,
                    help='Test Dataset')  # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
# parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

ckpt_path = f'./pretrained_models/model_deblurring.pth'

print(">> ckpt_path:", ckpt_path)

model_restoration = MPRNet().cpu()
utils.load_checkpoint(model_restoration, ckpt_path, False)
print("===>Testing using weights: ", ckpt_path)


# model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = dataset
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                         pin_memory=True)
# print(">> test_loader:", len(test_loader), test_loader)

result_dir = os.path.join(args.result_dir, dataset)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        # torch.cuda.ipc_collect()
        # torch.cuda.empty_cache()

        # input_ = data_test[0].cuda()
        input_ = data_test[0]
        filenames = data_test[1]

        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        # if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
        if h % 8 != 0 or w % 8 != 0:
            factor = 8
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        # print("input_:", input_.shape)
        # print(f"\n>> input_: mean: {input_.mean()}, max: {input_.max()}, min: {input_.min()}")

        restored = model_restoration(input_)
        # print(">> len(restored):", len(restored))
        # restored = restored[0]
        restored = torch.clamp(restored[0], 0, 1)

        # Unpad images to original dimensions
        # if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
        if h % 8 != 0 or w % 8 != 0:
            restored = restored[:, :, :h, :w]

        # print(f">> restored: mean: {restored.mean()}, max: {restored.max()}, min: {restored.min()}")

        restored = restored.permute(0, 2, 3, 1).detach().numpy()

        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            # print(f">> restored_img: mean: {restored_img.mean()}, max: {restored_img.max()}, min: {restored_img.min()}")
            utils.save_img((os.path.join(result_dir, f'{filenames[batch]}_pre.png')), restored_img)
