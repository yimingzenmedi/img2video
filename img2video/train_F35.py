import os

from config import Config

opt = Config('./training_F35.yml')

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
from FCDiscriminator import FCDiscriminator
import torch.nn.functional as F
from torch.autograd import Variable

if __name__ == '__main__':

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION
    index3 = 3
    index5 = 5

    model_dir_desc_35 = os.path.join(opt.TRAINING.SAVE_DIR, session, "desc_35")
    model_dir3 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F35_3")
    model_dir5 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F35_5")
    utils.mkdir(model_dir_desc_35)
    utils.mkdir(model_dir3)
    utils.mkdir(model_dir5)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    ######### Model ###########
    model_restoration3 = MPRNet()
    model_restoration5 = MPRNet()
    model_restoration3.cuda()
    model_restoration5.cuda()

    new_lr = opt.OPTIM.LR_INITIAL
    optimizer1 = optim.Adam(model_restoration3.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer2 = optim.Adam(model_restoration5.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    discriminator = FCDiscriminator()
    discriminator.cuda()
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=new_lr)

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                             eta_min=opt.OPTIM.LR_MIN)
    scheduler1 = GradualWarmupScheduler(optimizer1, multiplier=1, total_epoch=warmup_epochs,
                                        after_scheduler=scheduler_cosine1)
    scheduler1.step()

    scheduler_cosine2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                             eta_min=opt.OPTIM.LR_MIN)
    scheduler2 = GradualWarmupScheduler(optimizer2, multiplier=1, total_epoch=warmup_epochs,
                                        after_scheduler=scheduler_cosine2)
    scheduler2.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_disc_35 = utils.get_last_path(model_dir_desc_35, f"disc_35_latest.pth")
        utils.load_checkpoint(discriminator, path_chk_disc_35)
        utils.load_optim(discriminator_optimizer, path_chk_disc_35)

        path_chk_rest1 = utils.get_last_path(model_dir3, f'F35_model_{index3}_latest.pth')
        utils.load_checkpoint(model_restoration3, path_chk_rest1)
        utils.load_optim(optimizer1, path_chk_rest1)

        path_chk_rest2 = utils.get_last_path(model_dir5, f'F35_model_{index5}_latest.pth')
        utils.load_checkpoint(model_restoration5, path_chk_rest2)
        utils.load_optim(optimizer2, path_chk_rest2)

        start_epoch = utils.load_start_epoch(path_chk_rest1) + 1

        for i in range(1, start_epoch):
            scheduler1.step()
            scheduler2.step()
        new_lr1 = scheduler1.get_lr()[0]
        new_lr2 = scheduler2.get_lr()[0]

        print('------------------------------------------------------------------------------')
        print(f"==> Resuming Training with learning rate - new_lr2: {new_lr1}, new_lr2: {new_lr2}")
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration3 = nn.DataParallel(model_restoration3, device_ids=device_ids)
        model_restoration5 = nn.DataParallel(model_restoration5, device_ids=device_ids)

    ######### Loss ###########
    # criterion_char = losses.CharbonnierLoss()
    # criterion_edge = losses.EdgeLoss()
    criterion = losses.F35_N8Loss()

    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS}, 7, [index3 - 1, index5 - 1])
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=8,
                              drop_last=False, pin_memory=True)
    # print("!! Read val_dataset")
    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS}, 7, [index3 - 1, index5 - 1])
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = (0, 0)
    best_epoch3 = 0
    best_epoch5 = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model_restoration3.train()
        model_restoration5.train()

        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            for param in model_restoration3.parameters():
                param.grad = None
            for param in model_restoration5.parameters():
                param.grad = None

            input_ = data[1].cuda()
            target3 = data[0][0].cuda()
            target5 = data[0][1].cuda()
            # print(f">> input_: mean: {input_.mean()}, max: {input_.max()}, min: {input_.min()}")
            # print(f">> target1: mean: {target1.mean()}, max: {target1.max()}, min: {target1.min()}")
            # print(f">> target2: mean: {target2.mean()}, max: {target2.max()}, min: {target2.min()}")

            restored3 = model_restoration3(input_)[0]
            restored5 = model_restoration5(input_)[0]

            # print(f">> restored1: mean: {restored1.mean()}, max: {restored1.max()}, min: {restored1.min()}")
            # print(f">> restored2: mean: {restored2.mean()}, max: {restored2.max()}, min: {restored2.min()}")

            #######################################################
            # discriminator:
            CE = torch.nn.BCEWithLogitsLoss()

            dis_output3 = discriminator(target3, restored3.detach())
            dis_output3 = F.interpolate(dis_output3, size=(target3.shape[2], target3.shape[3]),
                                        mode="bilinear", align_corners=True)
            out_labels3 = Variable(torch.FloatTensor(np.zeros(dis_output3.shape))).cuda()
            tgt_labels3 = Variable(torch.FloatTensor(np.ones(dis_output3.shape))).cuda()
            dis_output3_loss = CE(dis_output3, out_labels3) * np.prod(dis_output3.shape)
            # print("> dis_output3_loss:", dis_output3_loss)

            dis_output5 = discriminator(target5, restored5.detach())
            dis_output5 = F.interpolate(dis_output5, size=(target5.shape[2], target5.shape[3]),
                                        mode="bilinear", align_corners=True)
            out_labels5 = Variable(torch.FloatTensor(np.zeros(dis_output5.shape))).cuda()
            tgt_labels5 = Variable(torch.FloatTensor(np.ones(dis_output5.shape))).cuda()
            dis_output_loss5 = CE(dis_output5, out_labels5) * np.prod(dis_output5.shape)
            # print("> dis_output2_loss:", dis_output_loss5)

            dis_output_loss = dis_output3_loss + dis_output_loss5
            dis_output_loss = dis_output_loss * 0.2
            # print("> dis_output_loss:", dis_output_loss.data)

            # Compute loss at each stage
            loss = criterion(restored3, restored5, target3, target5) + dis_output_loss
            # print(f"\nrestored1: {restored1.mean()}, restored2: {restored2.mean()}, \ntarget1:   {target1.mean()}, target2:   {target2.mean()}, ")

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            # ##########################################################################################
            # train discriminator:
            torch.autograd.set_detect_anomaly(True)

            dis_pred3 = torch.sigmoid(restored3).detach()
            dis_output3 = F.interpolate(discriminator(target3, dis_pred3),
                                        size=(target3.shape[2], target3.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_output3_loss = CE(dis_output3, out_labels3)
            # print(f"> target3: {target3.shape}, labels3: {labels3.shape}, restored[0]: {restored[0].shape}, "
            #       f"dis_output1: {dis_output1.shape}, dis_output1_loss: {dis_output1_loss.shape}")
            dis_target3 = discriminator(target3, target3)
            dis_target3 = F.interpolate(dis_target3, size=(target3.shape[2], target3.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_target3_loss = CE(dis_target3, tgt_labels3)

            dis_pred5 = torch.sigmoid(restored5).detach()
            dis_output5 = F.interpolate(discriminator(target5, dis_pred5),
                                        size=(target5.shape[2], target5.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_output2_loss = CE(dis_output5, out_labels5)

            dis_target5 = discriminator(target5, target5)
            dis_target5 = F.interpolate(dis_target5, size=(target5.shape[2], target5.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_target5_loss = CE(dis_target5, tgt_labels5)

            dis_loss = 0.25 * (dis_output3_loss + dis_target3_loss + dis_output2_loss + dis_target5_loss)
            # print(" dis_loss:", dis_loss)
            dis_loss.backward()
            discriminator_optimizer.step()

            epoch_loss += loss.item()

        #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration3.eval()
            model_restoration5.eval()
            psnr_val_rgb3 = []
            psnr_val_rgb5 = []
            for ii, data_val in enumerate((val_loader), 0):
                target3 = data_val[0][0].cuda()
                target5 = data_val[0][1].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored3 = model_restoration3(input_)
                    restored5 = model_restoration5(input_)
                restored3 = restored3[0]
                restored5 = restored5[0]
                # print(restored1.shape)
                # print(restored2.shape)
                # print(target3.shape)
                # print(target5.shape)
                for res1, res2, tar1, tar2 in zip(restored3, restored5, target3, target5):
                    psnr_val_rgb3.append(utils.torchPSNR(res1, tar1))
                    psnr_val_rgb5.append(utils.torchPSNR(res2, tar2))

            psnr_val_rgb3 = torch.stack(psnr_val_rgb3).mean().item()
            psnr_val_rgb5 = torch.stack(psnr_val_rgb5).mean().item()

            psnr_diff3 = psnr_val_rgb3 - best_psnr[0]
            psnr_diff5 = psnr_val_rgb5 - best_psnr[1]

            if psnr_val_rgb3 - best_psnr[0] > 0:
                best_psnr = (psnr_val_rgb3, best_psnr[1])
                best_epoch3 = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration3.state_dict(),
                            'optimizer': optimizer1.state_dict()
                            }, os.path.join(model_dir3, f'F35_model_{index3}_best.pth'))

            if psnr_val_rgb5 - best_psnr[1] > 0:
                best_psnr = (best_psnr[0], psnr_val_rgb5)
                best_epoch5 = epoch

                torch.save({'epoch': epoch,
                            'state_dict': model_restoration5.state_dict(),
                            'optimizer': optimizer2.state_dict()
                            }, os.path.join(model_dir5, f'F35_model_{index5}_best.pth'))

            print(
                "[epoch %d PSNR: %.4f, %.4f --- best_epoch %d, %d Best_PSNR %.4f (%.4f), %.4f (%.4f)]" % (
                    epoch, psnr_val_rgb3, psnr_val_rgb5, best_epoch3, best_epoch5, best_psnr[0], psnr_diff3, best_psnr[1], psnr_diff5))

            torch.save({'epoch': epoch,
                        'state_dict': model_restoration3.state_dict(),
                        'optimizer': optimizer1.state_dict()
                        }, os.path.join(model_dir3, f"F35_model_{index3}_epoch_{epoch}.pth"))
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration5.state_dict(),
                        'optimizer': optimizer2.state_dict()
                        }, os.path.join(model_dir5, f"F35_model_{index5}_epoch_{epoch}.pth"))

        scheduler1.step()
        scheduler2.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate1 {:.6f}\tLearningRate2 {:.6f}".format(epoch,
                                                                                                         time.time() - epoch_start_time,
                                                                                                         epoch_loss,
                                                                                                         scheduler1.get_lr()[
                                                                                                             0],
                                                                                                         scheduler2.get_lr()[
                                                                                                             0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration3.state_dict(),
                    'optimizer': optimizer1.state_dict()
                    }, os.path.join(model_dir3, f'F35_model_{index3}_latest.pth'))
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration5.state_dict(),
                    'optimizer': optimizer2.state_dict()
                    }, os.path.join(model_dir5, f'F35_model_{index5}_latest.pth'))

        torch.save({'epoch': epoch,
                    'state_dict': discriminator.state_dict(),
                    'optimizer': discriminator_optimizer.state_dict()
                    }, os.path.join(model_dir_desc_35, f'disc_35_latest.pth'))
