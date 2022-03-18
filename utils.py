

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure

import os
def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    return ssim_list


def validation(net, net_var, val_data_loader, device, category, save_tag=True):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        net_var.train()
        with torch.no_grad():
            input_img, gt, image_name = val_data
            input_img = input_img.cuda()
            gt = gt.cuda()
            input_imagex2 = torch.nn.functional.interpolate(input_img,scale_factor=0.5)
            batch_size = 1
            Var1 = torch.zeros([batch_size,3,256,256], dtype=torch.float32)
            Num_itr = 10
            x_output = torch.zeros([Num_itr,batch_size,3,256,256], dtype=torch.float32)
            for k in range(Num_itr):
              pred_image1,pred_imagex2 = net_var(input_img,input_imagex2)
              x_output[k,:,:,:,:].copy_(pred_image1.data)
            for j in range(batch_size):
              tx_output = x_output[:,j,:,:,:]
              tx_output = torch.squeeze(tx_output)
              Var1[j,:,:,:] = torch.std(tx_output,dim=0)
            pred_image,pred_imagex2 = net(input_img,input_imagex2,5*Var1)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

def gen_varmaps(net,  val_data_loader, device, category, save_tag=True):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []
    net.train()
    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_img, gt, image_name = val_data
            input_img = input_img.cuda()
            gt = gt.cuda()
            input_imagex2 = torch.nn.functional.interpolate(input_img,scale_factor=0.5)
            batch_size = 8
            Var1 = torch.zeros([batch_size,3,256,256], dtype=torch.float32)
            Num_itr = 10
            x_output = torch.zeros([Num_itr,batch_size,3,256,256], dtype=torch.float32)
            for k in range(Num_itr):
              pred_image,pred_imagex2 = net(input_img,input_imagex2)
              x_output[k,:,:,:,:].copy_(pred_image.data)
            for j in range(batch_size):
              tx_output = x_output[:,j,:,:,:]
              tx_output = torch.squeeze(tx_output)
              Var1[j,:,:,:] = torch.std(tx_output,dim=0)
              print(Var1.max())
            

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            save_image(5*Var1, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(pred_image, image_name, category):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    if os.path.exists('./{}_results'.format(category))==False:
        os.mkdir('./{}_results'.format(category))
    # print("hi")
    for ind in range(batch_num):
        utils.save_image(pred_image_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 40 if category == 'distort' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
