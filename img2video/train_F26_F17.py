import os

from config import Config

opt = Config('./training_F26_F17.yml')

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

    index1 = 1
    index2 = 2
    index6 = 6
    index7 = 7

    pic_index = [0, 1, 5, 6]

    model_dir_disc_17_26 = os.path.join(opt.TRAINING.SAVE_DIR, session, "disc_17_26")
    model_dir1 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F17_1")
    model_dir2 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F26_2")
    model_dir6 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F26_6")
    model_dir7 = os.path.join(opt.TRAINING.SAVE_DIR, session, "F17_7")
    utils.mkdir(model_dir_disc_17_26)
    utils.mkdir(model_dir1)
    utils.mkdir(model_dir2)
    utils.mkdir(model_dir6)
    utils.mkdir(model_dir7)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR
    new_lr = opt.OPTIM.LR_INITIAL

    ######### Model ###########
    model_restoration1 = MPRNet()
    model_restoration2 = MPRNet()
    model_restoration6 = MPRNet()
    model_restoration7 = MPRNet()
    model_restoration1.cuda()
    model_restoration2.cuda()
    model_restoration6.cuda()
    model_restoration7.cuda()

    discriminator = FCDiscriminator()
    discriminator.cuda()
    discriminator_params = discriminator.parameters()
    discriminator_optimizer = optim.Adam(discriminator_params, new_lr)

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    optimizer1 = optim.Adam(model_restoration1.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer2 = optim.Adam(model_restoration2.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer6 = optim.Adam(model_restoration6.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer7 = optim.Adam(model_restoration7.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

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

    scheduler_cosine6 = optim.lr_scheduler.CosineAnnealingLR(optimizer6, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                             eta_min=opt.OPTIM.LR_MIN)
    scheduler6 = GradualWarmupScheduler(optimizer6, multiplier=1, total_epoch=warmup_epochs,
                                        after_scheduler=scheduler_cosine6)
    scheduler6.step()

    scheduler_cosine7 = optim.lr_scheduler.CosineAnnealingLR(optimizer7, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                             eta_min=opt.OPTIM.LR_MIN)
    scheduler7 = GradualWarmupScheduler(optimizer7, multiplier=1, total_epoch=warmup_epochs,
                                        after_scheduler=scheduler_cosine7)
    scheduler7.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_disc = utils.get_last_path(model_dir_disc_17_26, f'disc_17_26_latest.pth')
        utils.load_checkpoint(discriminator, path_chk_disc)
        utils.load_optim(discriminator_optimizer, path_chk_disc)

        path_chk_rest1 = utils.get_last_path(model_dir1, f'F17_model_{index1}_latest.pth')
        utils.load_checkpoint(model_restoration1, path_chk_rest1)
        utils.load_optim(optimizer1, path_chk_rest1)

        path_chk_rest2 = utils.get_last_path(model_dir2, f'F26_model_{index2}_latest.pth')
        utils.load_checkpoint(model_restoration2, path_chk_rest2)
        utils.load_optim(optimizer2, path_chk_rest2)

        path_chk_rest6 = utils.get_last_path(model_dir6, f'F26_model_{index6}_latest.pth')
        utils.load_checkpoint(model_restoration6, path_chk_rest6)
        utils.load_optim(optimizer6, path_chk_rest6)

        path_chk_rest7 = utils.get_last_path(model_dir7, f'F17_model_{index7}_latest.pth')
        utils.load_checkpoint(model_restoration7, path_chk_rest7)
        utils.load_optim(optimizer7, path_chk_rest7)

        start_epoch = utils.load_start_epoch(path_chk_rest1) + 1

        for i in range(1, start_epoch):
            scheduler1.step()
            scheduler2.step()
            scheduler6.step()
            scheduler7.step()
        new_lr1 = scheduler1.get_lr()[0]
        new_lr2 = scheduler2.get_lr()[0]
        new_lr6 = scheduler6.get_lr()[0]
        new_lr7 = scheduler7.get_lr()[0]

        print('------------------------------------------------------------------------------')
        print(
            f"==> Resuming Training with learning rate - new_lr2: {new_lr1}, new_lr2: {new_lr2}, new_lr6: {new_lr6}, new_lr7: {new_lr7}")
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration1 = nn.DataParallel(model_restoration1, device_ids=device_ids)
        model_restoration2 = nn.DataParallel(model_restoration2, device_ids=device_ids)
        model_restoration6 = nn.DataParallel(model_restoration6, device_ids=device_ids)
        model_restoration7 = nn.DataParallel(model_restoration7, device_ids=device_ids)

    ######### Loss ###########
    # criterion_char = losses.CharbonnierLoss()
    # criterion_edge = losses.EdgeLoss()
    criterion = losses.F17_N9Loss_F26_N9Loss()

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

    best_psnr = (0, 0, 0, 0)
    best_epoch1 = 0
    best_epoch2 = 0
    best_epoch6 = 0
    best_epoch7 = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model_restoration1.train()
        model_restoration2.train()
        model_restoration6.train()
        model_restoration7.train()

        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            for param in model_restoration1.parameters():
                param.grad = None
            for param in model_restoration2.parameters():
                param.grad = None
            for param in model_restoration6.parameters():
                param.grad = None
            for param in model_restoration7.parameters():
                param.grad = None

            input_ = data[1].cuda()
            target1 = data[0][0].cuda()
            target2 = data[0][1].cuda()
            target6 = data[0][2].cuda()
            target7 = data[0][3].cuda()
            # print(f">> input_: mean: {input_.mean()}, max: {input_.max()}, min: {input_.min()}")
            # print(f">> target1: mean: {target1.mean()}, max: {target1.max()}, min: {target1.min()}")
            # print(f">> target2: mean: {target2.mean()}, max: {target2.max()}, min: {target2.min()}")

            restored1 = model_restoration1(input_)[0]
            restored2 = model_restoration2(input_)[0]
            restored6 = model_restoration6(input_)[0]
            restored7 = model_restoration7(input_)[0]

            # print(f">> restored1: mean: {restored1.mean()}, max: {restored1.max()}, min: {restored1.min()}")
            # print(f">> restored2: mean: {restored2.mean()}, max: {restored2.max()}, min: {restored2.min()}")

            #######################################################
            # discriminator:
            CE = torch.nn.BCEWithLogitsLoss()

            # 1:
            dis_output1 = discriminator(target1, restored1.detach())
            dis_output1 = F.interpolate(dis_output1, size=(target1.shape[2], target1.shape[3]),
                                        mode="bilinear", align_corners=True)
            out_labels1 = Variable(torch.FloatTensor(np.zeros(dis_output1.shape))).cuda()
            tgt_labels1 = Variable(torch.FloatTensor(np.ones(dis_output1.shape))).cuda()
            dis_output1_loss = CE(dis_output1, out_labels1) * np.prod(dis_output1.shape)
            # print("> dis_output1_loss:", dis_output1_loss)

            # 2:
            dis_output2 = discriminator(target2, restored2.detach())
            dis_output2 = F.interpolate(dis_output2, size=(target2.shape[2], target2.shape[3]),
                                        mode="bilinear", align_corners=True)
            out_labels2 = Variable(torch.FloatTensor(np.zeros(dis_output2.shape))).cuda()
            tgt_labels2 = Variable(torch.FloatTensor(np.ones(dis_output2.shape))).cuda()
            dis_output2_loss = CE(dis_output2, out_labels2) * np.prod(dis_output1.shape)
            # print("> dis_output2_loss:", dis_output2_loss)

            # 6:
            dis_output6 = discriminator(target6, restored6.detach())
            dis_output6 = F.interpolate(dis_output6, size=(target6.shape[2], target6.shape[3]),
                                        mode="bilinear", align_corners=True)
            out_labels6 = Variable(torch.FloatTensor(np.zeros(dis_output6.shape))).cuda()
            tgt_labels6 = Variable(torch.FloatTensor(np.ones(dis_output6.shape))).cuda()
            dis_output6_loss = CE(dis_output6, out_labels6) * np.prod(dis_output1.shape)
            # print("> dis_output6_loss:", dis_output6_loss)

            # 7:
            dis_output7 = discriminator(target7, restored7.detach())
            dis_output7 = F.interpolate(dis_output7, size=(target7.shape[2], target7.shape[3]),
                                        mode="bilinear", align_corners=True)
            out_labels7 = Variable(torch.FloatTensor(np.zeros(dis_output7.shape))).cuda()
            tgt_labels7 = Variable(torch.FloatTensor(np.ones(dis_output7.shape))).cuda()
            dis_output7_loss = CE(dis_output7, out_labels7) * np.prod(dis_output1.shape)
            # print("> dis_output7_loss:", dis_output7_loss)

            dis_output_loss = dis_output1_loss + dis_output2_loss + dis_output6_loss + dis_output7_loss
            dis_output_loss = dis_output_loss * 0.25
            # print("> dis_output_loss:", dis_output_loss.data)

            # Compute loss at each stage
            loss = criterion(restored1, restored7, restored2, restored6, target1, target7, target2,
                             target6) + dis_output_loss
            # print(f"\nrestored1: {restored1.mean()}, restored2: {restored2.mean()}, \ntarget1:   {target1.mean()}, target2:   {target2.mean()}, ")

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer6.step()
            optimizer7.step()

            # ##########################################################################################
            # train discriminator:
            torch.autograd.set_detect_anomaly(True)

            # 1:
            dis_pred1 = torch.sigmoid(restored1).detach()
            dis_output1 = F.interpolate(discriminator(target1, dis_pred1),
                                        size=(target1.shape[2], target1.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_output1_loss = CE(dis_output1, out_labels1)
            dis_target1 = discriminator(target1, target1)
            dis_target1 = F.interpolate(dis_target1, size=(target1.shape[2], target1.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_target1_loss = CE(dis_target1, tgt_labels1)

            # 2:
            dis_pred2 = torch.sigmoid(restored2).detach()
            dis_output2 = F.interpolate(discriminator(target2, dis_pred2),
                                        size=(target2.shape[2], target2.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_output2_loss = CE(dis_output2, out_labels2)

            dis_target2 = discriminator(target2, target2)
            dis_target2 = F.interpolate(dis_target2, size=(target2.shape[2], target2.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_target2_loss = CE(dis_target2, tgt_labels2)

            # 6:
            dis_pred6 = torch.sigmoid(restored6).detach()
            dis_output6 = F.interpolate(discriminator(target6, dis_pred6),
                                        size=(target6.shape[2], target6.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_output6_loss = CE(dis_output6, out_labels6)
            dis_target6 = discriminator(target6, target6)
            dis_target6 = F.interpolate(dis_target6, size=(target6.shape[2], target6.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_target6_loss = CE(dis_target6, tgt_labels6)

            # 7:
            dis_pred7 = torch.sigmoid(restored7).detach()
            dis_output7 = F.interpolate(discriminator(target7, dis_pred7),
                                        size=(target7.shape[2], target7.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_output7_loss = CE(dis_output7, out_labels7)
            dis_target7 = discriminator(target7, target7)
            dis_target7 = F.interpolate(dis_target7, size=(target7.shape[2], target7.shape[3]),
                                        mode="bilinear", align_corners=True)
            dis_target7_loss = CE(dis_target7, tgt_labels7)

            dis_loss = 0.1 * (dis_output1_loss + dis_target1_loss + dis_output2_loss + dis_target2_loss +
                              dis_output6_loss + dis_target6_loss + dis_output7_loss + dis_target7_loss)
            dis_loss.backward()
            discriminator_optimizer.step()

            epoch_loss += loss.item()

        #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration1.eval()
            model_restoration2.eval()
            model_restoration6.eval()
            model_restoration7.eval()
            psnr_val_rgb1 = []
            psnr_val_rgb2 = []
            psnr_val_rgb6 = []
            psnr_val_rgb7 = []
            for ii, data_val in enumerate((val_loader), 0):
                target1 = data_val[0][0].cuda()
                target2 = data_val[0][1].cuda()
                target6 = data_val[0][2].cuda()
                target7 = data_val[0][3].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored1 = model_restoration1(input_)
                    restored2 = model_restoration2(input_)
                    restored6 = model_restoration6(input_)
                    restored7 = model_restoration7(input_)
                restored1 = restored1[0]
                restored2 = restored2[0]
                restored6 = restored6[0]
                restored7 = restored7[0]

                for res1, res7, res2, res6, tar1, tar7, tar2, tar6 in zip(restored1, restored7, restored2, restored6,
                                                                          target1, target7, target2, target6):
                    psnr_val_rgb1.append(utils.torchPSNR(res1, tar1))
                    psnr_val_rgb2.append(utils.torchPSNR(res2, tar2))
                    psnr_val_rgb6.append(utils.torchPSNR(res6, tar6))
                    psnr_val_rgb7.append(utils.torchPSNR(res7, tar7))

            psnr_val_rgb1 = torch.stack(psnr_val_rgb1).mean().item()
            psnr_val_rgb2 = torch.stack(psnr_val_rgb2).mean().item()
            psnr_val_rgb6 = torch.stack(psnr_val_rgb6).mean().item()
            psnr_val_rgb7 = torch.stack(psnr_val_rgb7).mean().item()

            psnr_diff1 = psnr_val_rgb1 - best_psnr[0]
            psnr_diff2 = psnr_val_rgb2 - best_psnr[1]
            psnr_diff6 = psnr_val_rgb6 - best_psnr[2]
            psnr_diff7 = psnr_val_rgb7 - best_psnr[3]

            if psnr_diff1 > 0:
                best_psnr = (psnr_val_rgb1, best_psnr[1], best_psnr[2], best_psnr[3])
                best_epoch1 = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration1.state_dict(),
                            'optimizer': optimizer1.state_dict()
                            }, os.path.join(model_dir1, f'F17_model_{index1}_best.pth'))

            if psnr_diff2 > 0:
                best_psnr = (best_psnr[0], psnr_val_rgb2, best_psnr[2], best_psnr[3])
                best_epoch2 = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration2.state_dict(),
                            'optimizer': optimizer2.state_dict()
                            }, os.path.join(model_dir2, f'F26_model_{index2}_best.pth'))

            if psnr_diff6 > 0:
                best_psnr = (best_psnr[0], best_psnr[1], psnr_val_rgb6, best_psnr[3])
                best_epoch6 = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration6.state_dict(),
                            'optimizer': optimizer6.state_dict()
                            }, os.path.join(model_dir6, f'F26_model_{index6}_best.pth'))

            if psnr_diff7 > 0:
                best_psnr = (best_psnr[0], best_psnr[1], best_psnr[2], psnr_val_rgb7)
                best_epoch7 = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration7.state_dict(),
                            'optimizer': optimizer7.state_dict()
                            }, os.path.join(model_dir7, f'F17_model_{index7}_best.pth'))

            print(
                "[epoch %d PSNR: %.4f, %.4f, %.4f, %.4f --- best_epoch %d %d %d %d \n Best_PSNR %.4f (%.4f), %.4f (%.4f), %.4f (%.4f), %.4f (%.4f)]" % (
                    epoch, psnr_val_rgb1, psnr_val_rgb2, psnr_val_rgb6, psnr_val_rgb7,
                    best_epoch1, best_epoch2, best_epoch6, best_epoch7,
                    best_psnr[0], psnr_diff1, best_psnr[1], psnr_diff2, best_psnr[2], psnr_diff6, best_psnr[3],
                    psnr_diff7))

            torch.save({'epoch': epoch,
                        'state_dict': model_restoration1.state_dict(),
                        'optimizer': optimizer1.state_dict()
                        }, os.path.join(model_dir1, f"F17_model_{index1}_epoch_{epoch}.pth"))
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration2.state_dict(),
                        'optimizer': optimizer2.state_dict()
                        }, os.path.join(model_dir2, f"F26_model_{index2}_epoch_{epoch}.pth"))
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration6.state_dict(),
                        'optimizer': optimizer6.state_dict()
                        }, os.path.join(model_dir6, f"F26_model_{index6}_epoch_{epoch}.pth"))
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration7.state_dict(),
                        'optimizer': optimizer7.state_dict()
                        }, os.path.join(model_dir7, f"F17_model_{index7}_epoch_{epoch}.pth"))

        scheduler1.step()
        scheduler2.step()
        scheduler6.step()
        scheduler7.step()

        print("------------------------------------------------------------------")
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate1 {:.6f}\tLearningRate2 {:.6f}\tLearningRate6 {:.6f}\tLearningRate7 {:.6f}".format(
                epoch,
                time.time() - epoch_start_time,
                epoch_loss,
                scheduler1.get_lr()[
                    0],
                scheduler2.get_lr()[
                    0],
                scheduler6.get_lr()[
                    0],
                scheduler7.get_lr()[
                    0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration1.state_dict(),
                    'optimizer': optimizer1.state_dict()
                    }, os.path.join(model_dir1, f'F17_model_{index1}_latest.pth'))
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration2.state_dict(),
                    'optimizer': optimizer2.state_dict()
                    }, os.path.join(model_dir2, f'F26_model_{index2}_latest.pth'))
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration6.state_dict(),
                    'optimizer': optimizer6.state_dict()
                    }, os.path.join(model_dir6, f'F26_model_{index6}_latest.pth'))
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration7.state_dict(),
                    'optimizer': optimizer7.state_dict()
                    }, os.path.join(model_dir7, f'F17_model_{index7}_latest.pth'))

        torch.save({'epoch': epoch,
                    'state_dict': discriminator.state_dict(),
                    'optimizer': discriminator_optimizer.state_dict()
                    }, os.path.join(model_dir_disc_17_26, f'disc_17_26_latest.pth'))
