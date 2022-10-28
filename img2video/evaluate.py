import os

from config import Config

opt = Config('./evaluate.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
from torch.utils.data import DataLoader

import random

import utils
from data_RGB import get_validation_data
from MPRNet import MPRNet

import os
import numpy as np

if __name__ == '__main__':

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION
    val_dir = opt.TRAINING.VAL_DIR

    index1 = 1
    index2 = 2
    index3 = 3
    index4 = 4
    index5 = 5
    index6 = 6
    index7 = 7

    pic_index = [0, 1, 2, 3, 4, 5, 6]

    model_dir1 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F17_1")
    model_dir2 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F26_2")
    model_dir3 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F35_3")
    model_dir4 = os.path.join(opt.TRAINING.SAVE_DIR, session, "centerEsti")
    model_dir5 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F35_5")
    model_dir6 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F26_6")
    model_dir7 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F17_7")
    model_dir_detailed = os.path.join(opt.TRAINING.SAVE_DIR, session, "detailed")

    ######### Model ###########
    model_detailed = MPRNet()
    ckpt_detailed = torch.load(os.path.join(model_dir_detailed, f'detailed_model_best.pth'))
    # ckpt_detailed = torch.load("./pretrained_models/model_deblurring.pth")
    model_detailed.load_state_dict(ckpt_detailed['state_dict'])
    model_detailed.cuda()

    model_restoration1 = MPRNet()
    model_restoration2 = MPRNet()
    model_restoration3 = MPRNet()
    model_restoration4 = MPRNet()
    model_restoration5 = MPRNet()
    model_restoration6 = MPRNet()
    model_restoration7 = MPRNet()
    ckpt1 = torch.load(os.path.join(model_dir1, "F17_model_1_best.pth"))
    model_restoration1.load_state_dict(ckpt1['state_dict'])
    ckpt2 = torch.load(os.path.join(model_dir2, "F26_model_2_best.pth"))
    model_restoration2.load_state_dict(ckpt1['state_dict'])
    ckpt3 = torch.load(os.path.join(model_dir3, "F35_model_3_best.pth"))
    model_restoration3.load_state_dict(ckpt1['state_dict'])
    ckpt4 = torch.load(os.path.join(model_dir4, "centerEsti_model_best.pth"))
    model_restoration4.load_state_dict(ckpt1['state_dict'])
    ckpt5 = torch.load(os.path.join(model_dir5, "F35_model_5_best.pth"))
    model_restoration5.load_state_dict(ckpt1['state_dict'])
    ckpt6 = torch.load(os.path.join(model_dir6, "F26_model_6_best.pth"))
    model_restoration6.load_state_dict(ckpt1['state_dict'])
    ckpt7 = torch.load(os.path.join(model_dir7, "F17_model_7_best.pth"))
    model_restoration7.load_state_dict(ckpt1['state_dict'])
    model_restoration1.cuda()
    model_restoration2.cuda()
    model_restoration3.cuda()
    model_restoration4.cuda()
    model_restoration5.cuda()
    model_restoration6.cuda()
    model_restoration7.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    ######### DataLoaders ###########
    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS}, 7, pic_index)
    val_loader = DataLoader(dataset=val_dataset, batch_size=70, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    print('===> Loading datasets')

    psnr_sharp1 = []
    psnr_sharp2 = []
    psnr_sharp3 = []
    psnr_sharp4 = []
    psnr_sharp5 = []
    psnr_sharp6 = []
    psnr_sharp7 = []

    ssim_sharp1 = []
    ssim_sharp2 = []
    ssim_sharp3 = []
    ssim_sharp4 = []
    ssim_sharp5 = []
    ssim_sharp6 = []
    ssim_sharp7 = []

    for ii, data_val in enumerate(val_loader, 0):
        print(f"\repoch: {ii + 1} of {len(val_loader)}", end="")
        target1 = data_val[0][0].cuda()
        target2 = data_val[0][1].cuda()
        target3 = data_val[0][2].cuda()
        target4 = data_val[0][3].cuda()
        target5 = data_val[0][4].cuda()
        target6 = data_val[0][5].cuda()
        target7 = data_val[0][6].cuda()
        input_ = data_val[1].cuda()
        # print(">> input_:", input_.shape)

        with torch.no_grad():
            restored1 = model_restoration1(input_)[0]
            restored2 = model_restoration2(input_)[0]
            restored3 = model_restoration2(input_)[0]
            restored4 = model_restoration2(input_)[0]
            restored5 = model_restoration2(input_)[0]
            restored6 = model_restoration6(input_)[0]
            restored7 = model_restoration7(input_)[0]
            restored1 = torch.clamp(restored1, 0, 1)
            restored2 = torch.clamp(restored2, 0, 1)
            restored3 = torch.clamp(restored3, 0, 1)
            restored4 = torch.clamp(restored4, 0, 1)
            restored5 = torch.clamp(restored5, 0, 1)
            restored6 = torch.clamp(restored6, 0, 1)
            restored7 = torch.clamp(restored7, 0, 1)

            restored1 = model_detailed(restored1)[0]
            restored1 = torch.clamp(restored1, 0, 1)
            restored2 = model_detailed(restored2)[0]
            restored2 = torch.clamp(restored2, 0, 1)
            restored3 = model_detailed(restored3)[0]
            restored3 = torch.clamp(restored3, 0, 1)
            restored4 = model_detailed(restored4)[0]
            restored4 = torch.clamp(restored4, 0, 1)
            restored5 = model_detailed(restored5)[0]
            restored5 = torch.clamp(restored5, 0, 1)
            restored6 = model_detailed(restored6)[0]
            restored6 = torch.clamp(restored6, 0, 1)
            restored7 = model_detailed(restored7)[0]
            restored7 = torch.clamp(restored7, 0, 1)

        for res_det1, res_det2, res_det3, res_det4, res_det5, res_det6, res_det7, \
            tar1, tar2, tar3, tar4, tar5, tar6, tar7 \
            in zip(restored1, restored2, restored3, restored4, restored5, restored6, restored7,
                   target1, target2, target3, target4, target5, target6, target7,
            ):

            psnr_sharp1.append(utils.torchPSNR(res_det1, tar1).item())
            psnr_sharp2.append(utils.torchPSNR(res_det2, tar2).item())
            psnr_sharp3.append(utils.torchPSNR(res_det3, tar3).item())
            psnr_sharp4.append(utils.torchPSNR(res_det4, tar4).item())
            psnr_sharp5.append(utils.torchPSNR(res_det5, tar5).item())
            psnr_sharp6.append(utils.torchPSNR(res_det6, tar6).item())
            psnr_sharp7.append(utils.torchPSNR(res_det7, tar7).item())

            ssim_sharp1.append(utils.calculate_ssim(res_det1 * 255, tar1 * 255))
            ssim_sharp2.append(utils.calculate_ssim(res_det2 * 255, tar2 * 255))
            ssim_sharp3.append(utils.calculate_ssim(res_det3 * 255, tar3 * 255))
            ssim_sharp4.append(utils.calculate_ssim(res_det4 * 255, tar4 * 255))
            ssim_sharp5.append(utils.calculate_ssim(res_det5 * 255, tar5 * 255))
            ssim_sharp6.append(utils.calculate_ssim(res_det6 * 255, tar6 * 255))
            ssim_sharp7.append(utils.calculate_ssim(res_det7 * 255, tar7 * 255))

    psnr_sharp1 = np.mean(psnr_sharp1)
    psnr_sharp2 = np.mean(psnr_sharp2)
    psnr_sharp3 = np.mean(psnr_sharp3)
    psnr_sharp4 = np.mean(psnr_sharp4)
    psnr_sharp5 = np.mean(psnr_sharp5)
    psnr_sharp6 = np.mean(psnr_sharp6)
    psnr_sharp7 = np.mean(psnr_sharp7)

    ssim_sharp1 = np.mean(ssim_sharp1)
    ssim_sharp2 = np.mean(ssim_sharp2)
    ssim_sharp3 = np.mean(ssim_sharp3)
    ssim_sharp4 = np.mean(ssim_sharp4)
    ssim_sharp5 = np.mean(ssim_sharp5)
    ssim_sharp6 = np.mean(ssim_sharp6)
    ssim_sharp7 = np.mean(ssim_sharp7)


    print(">>\n"
          "PSNR out: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n"
          "SSIM out: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n" % (
              psnr_sharp1, psnr_sharp2, psnr_sharp3, psnr_sharp4, psnr_sharp5, psnr_sharp6, psnr_sharp7,
              ssim_sharp1, ssim_sharp2, ssim_sharp3, ssim_sharp4, ssim_sharp5, ssim_sharp6, ssim_sharp7,
          ))
