import os

from config import Config

opt = Config('./training_detailed.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from MPRNet import MPRNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm

if __name__ == '__main__':

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

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
    utils.mkdir(model_dir_detailed)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    ######### Model ###########
    model_detailed = MPRNet()
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

    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(model_detailed.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_detailed = utils.get_last_path(model_dir_detailed, f'detailed_model_latest.pth')
        utils.load_checkpoint(model_detailed, path_chk_detailed)
        utils.load_optim(optimizer, path_chk_detailed)

        start_epoch = utils.load_start_epoch(path_chk_detailed) + 1

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]

        print('------------------------------------------------------------------------------')
        print(
            f"==> Resuming Training with learning rate - new_lr: {new_lr}")
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_detailed = nn.DataParallel(model_detailed, device_ids=device_ids)

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    # criterion = losses.F17_N9Loss_F26_N9Loss()

    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS}, 7, pic_index)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=8,
                              drop_last=False, pin_memory=True)
    # print("!! Read val_dataset")
    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS}, 7, pic_index)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = (0, 0, 0, 0, 0, 0, 0)
    best_epoch = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model_restoration1.train()

        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            for param in model_detailed.parameters():
                param.grad = None

            input_ = data[1].cuda()
            target1 = data[0][0].cuda()
            target2 = data[0][1].cuda()
            target3 = data[0][2].cuda()
            target4 = data[0][3].cuda()
            target5 = data[0][4].cuda()
            target6 = data[0][5].cuda()
            target7 = data[0][6].cuda()

            restored1 = model_restoration1(input_)[0].detach()
            restored2 = model_restoration2(input_)[0].detach()
            restored3 = model_restoration2(input_)[0].detach()
            restored4 = model_restoration2(input_)[0].detach()
            restored5 = model_restoration2(input_)[0].detach()
            restored6 = model_restoration6(input_)[0].detach()
            restored7 = model_restoration7(input_)[0].detach()
            restored_detailed1 = model_detailed(restored1)[0]
            restored_detailed2 = model_detailed(restored2)[0]
            restored_detailed3 = model_detailed(restored3)[0]
            restored_detailed4 = model_detailed(restored4)[0]
            restored_detailed5 = model_detailed(restored5)[0]
            restored_detailed6 = model_detailed(restored6)[0]
            restored_detailed7 = model_detailed(restored7)[0]

            # Compute loss at each stage
            loss_char = criterion_char(restored_detailed1, target1) * 1.5
            loss_edge = criterion_edge(restored_detailed1, target1) * 1.5

            loss_char += criterion_char(restored_detailed2, target2)
            loss_edge += criterion_edge(restored_detailed2, target2)

            loss_char += criterion_char(restored_detailed3, target3)
            loss_edge += criterion_edge(restored_detailed3, target3)

            loss_char += criterion_char(restored_detailed4, target4)
            loss_edge += criterion_edge(restored_detailed4, target4)

            loss_char += criterion_char(restored_detailed5, target5)
            loss_edge += criterion_edge(restored_detailed5, target5)

            loss_char += criterion_char(restored_detailed6, target6)
            loss_edge += criterion_edge(restored_detailed6, target6)

            loss_char += criterion_char(restored_detailed7, target7) * 1.5
            loss_edge += criterion_edge(restored_detailed7, target7) * 1.5

            loss = (loss_char + (0.05 * loss_edge)) / 7

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_detailed.eval()

            psnr_val_rgb1 = []
            psnr_val_rgb2 = []
            psnr_val_rgb3 = []
            psnr_val_rgb4 = []
            psnr_val_rgb5 = []
            psnr_val_rgb6 = []
            psnr_val_rgb7 = []

            psnr_val_rgb_before1 = []
            psnr_val_rgb_before2 = []
            psnr_val_rgb_before3 = []
            psnr_val_rgb_before4 = []
            psnr_val_rgb_before5 = []
            psnr_val_rgb_before6 = []
            psnr_val_rgb_before7 = []
            for ii, data_val in enumerate((val_loader), 0):
                target1 = data_val[0][0].cuda()
                target2 = data_val[0][1].cuda()
                target3 = data_val[0][2].cuda()
                target4 = data_val[0][3].cuda()
                target5 = data_val[0][4].cuda()
                target6 = data_val[0][5].cuda()
                target7 = data_val[0][6].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored1 = model_restoration1(input_)[0]
                    restored_detailed1 = model_detailed(restored1)[0]
                    restored2 = model_restoration2(input_)[0]
                    restored_detailed2 = model_detailed(restored2)[0]
                    restored3 = model_restoration2(input_)[0]
                    restored_detailed3 = model_detailed(restored3)[0]
                    restored4 = model_restoration2(input_)[0]
                    restored_detailed4 = model_detailed(restored4)[0]
                    restored5 = model_restoration2(input_)[0]
                    restored_detailed5 = model_detailed(restored5)[0]
                    restored6 = model_restoration6(input_)[0]
                    restored_detailed6 = model_detailed(restored6)[0]
                    restored7 = model_restoration7(input_)[0]
                    restored_detailed7 = model_detailed(restored7)[0]

                for res_det1, res_det2, res_det3, res_det4, res_det5, res_det6, res_det7, tar1, tar2, tar3, tar4, tar5, tar6, tar7, res1, res2, res3, res4, res5, res6, res7 \
                        in zip(restored_detailed1, restored_detailed2, restored_detailed3, restored_detailed4,
                               restored_detailed5, restored_detailed6, restored_detailed7,
                               target1, target2, target3, target4, target5, target6, target7,
                               restored1, restored2, restored3, restored4, restored5, restored6, restored7,):
                    psnr_val_rgb1.append(utils.torchPSNR(res_det1, tar1))
                    psnr_val_rgb2.append(utils.torchPSNR(res_det2, tar2))
                    psnr_val_rgb3.append(utils.torchPSNR(res_det3, tar3))
                    psnr_val_rgb4.append(utils.torchPSNR(res_det4, tar4))
                    psnr_val_rgb5.append(utils.torchPSNR(res_det5, tar5))
                    psnr_val_rgb6.append(utils.torchPSNR(res_det6, tar6))
                    psnr_val_rgb7.append(utils.torchPSNR(res_det7, tar7))

                    psnr_val_rgb_before1.append(utils.torchPSNR(res1, tar1))
                    psnr_val_rgb_before2.append(utils.torchPSNR(res2, tar2))
                    psnr_val_rgb_before3.append(utils.torchPSNR(res3, tar3))
                    psnr_val_rgb_before4.append(utils.torchPSNR(res4, tar4))
                    psnr_val_rgb_before5.append(utils.torchPSNR(res5, tar5))
                    psnr_val_rgb_before6.append(utils.torchPSNR(res6, tar6))
                    psnr_val_rgb_before7.append(utils.torchPSNR(res7, tar7))

            psnr_val_rgb1 = torch.stack(psnr_val_rgb1).mean().item()
            psnr_val_rgb2 = torch.stack(psnr_val_rgb2).mean().item()
            psnr_val_rgb3 = torch.stack(psnr_val_rgb3).mean().item()
            psnr_val_rgb4 = torch.stack(psnr_val_rgb4).mean().item()
            psnr_val_rgb5 = torch.stack(psnr_val_rgb5).mean().item()
            psnr_val_rgb6 = torch.stack(psnr_val_rgb6).mean().item()
            psnr_val_rgb7 = torch.stack(psnr_val_rgb7).mean().item()

            psnr_val_rgb_before1 = torch.stack(psnr_val_rgb_before1).mean().item()
            psnr_val_rgb_before2 = torch.stack(psnr_val_rgb_before2).mean().item()
            psnr_val_rgb_before3 = torch.stack(psnr_val_rgb_before3).mean().item()
            psnr_val_rgb_before4 = torch.stack(psnr_val_rgb_before4).mean().item()
            psnr_val_rgb_before5 = torch.stack(psnr_val_rgb_before5).mean().item()
            psnr_val_rgb_before6 = torch.stack(psnr_val_rgb_before6).mean().item()
            psnr_val_rgb_before7 = torch.stack(psnr_val_rgb_before7).mean().item()

            if (psnr_val_rgb1 - best_psnr[0]) * 1.5 + (psnr_val_rgb2 - best_psnr[1]) + \
                    (psnr_val_rgb3 - best_psnr[2]) + (psnr_val_rgb4 - best_psnr[3]) + \
                    (psnr_val_rgb5 - best_psnr[4]) + (psnr_val_rgb6 - best_psnr[5]) + \
                    (psnr_val_rgb7 - best_psnr[6]) * 1.5 > 0:
                best_psnr = (psnr_val_rgb1, psnr_val_rgb2, psnr_val_rgb3, psnr_val_rgb4, psnr_val_rgb5, psnr_val_rgb6, psnr_val_rgb7)
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_detailed.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir_detailed, f'detailed_model_best.pth'))

            print(">> epoch %d --- best_epoch %d\n"
                  "PSNR:     %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n"
                  "Best_PSNR %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n"
                  "Pre_PSNR  %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (
                      epoch, best_epoch,
                      psnr_val_rgb1, psnr_val_rgb2, psnr_val_rgb3, psnr_val_rgb4, psnr_val_rgb5, psnr_val_rgb6, psnr_val_rgb7,
                      best_psnr[0], best_psnr[1], best_psnr[2], best_psnr[3], best_psnr[4], best_psnr[5],best_psnr[6],
                      psnr_val_rgb_before1, psnr_val_rgb_before2, psnr_val_rgb_before3, psnr_val_rgb_before4, psnr_val_rgb_before5, psnr_val_rgb_before6, psnr_val_rgb_before7
            ))

            torch.save({'epoch': epoch,
                        'state_dict': model_detailed.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir_detailed, f"detailed_model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate1 {:.6f}".format(
                epoch,
                time.time() - epoch_start_time,
                epoch_loss,
                scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_detailed.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir_detailed, f'detailed_model_latest.pth'))
