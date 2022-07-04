

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import TrainData
from val_data import ValData
from face_turv2 import Turb,Turb_mcdrp
from vgg16 import Vgg16
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import itertools
import config_tdrn as config
import os
import pdb
plt.switch_backend('agg')
from tqdm import tqdm

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256,256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=2, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.002, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (distort?)', default='distort', type=str)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default='./results/',type=str)
parser.add_argument('-lambda_GP', help='Set the lambda_GP for gploss in loss function', default=0.0015, type=float)
args = parser.parse_args()

distance='300'
learning_rate = args.learning_rate
epoch_start = args.epoch_start
crop_size = args.crop_size
train_batch_size = args.train_batch_size
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
exp_name = args.exp_name
lambgp = args.lambda_GP

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\n'
      'num_dense_layer: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size,  lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'distort':
    num_epochs = 200
    train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/FFHD_data/images512x512/'
    val_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/Tubfaces89/300M/tubimages/'
elif category == 'dehaze':
    num_epochs = 10
    train_data_dir = './data/train/dehaze/'
    val_data_dir = './data/test/dehaze/'
else:
    raise Exception('Wrong image category. Set it to derain or dehaze for RESIDE dateset.')


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Turb()
net_var = Turb_mcdrp()

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --- Multi-GPU --- #
net = net.cuda()
net_var = net_var.cuda()
net = nn.DataParallel(net)
net_var = nn.DataParallel(net_var)


# --- Define the perceptual loss network --- #
vgg = Vgg16()

state_dict_g = torch.load('VGG_FACE.pth')
new_state_dict_g = {}
new_state_dict_g["conv1_1.weight"]= state_dict_g["0.weight"]
new_state_dict_g["conv1_1.bias"]= state_dict_g["0.bias"]
new_state_dict_g["conv1_2.weight"]= state_dict_g["2.weight"]
new_state_dict_g["conv1_2.bias"]= state_dict_g["2.bias"]
new_state_dict_g["conv2_1.weight"]= state_dict_g["5.weight"]
new_state_dict_g["conv2_1.bias"]= state_dict_g["5.bias"]
new_state_dict_g["conv2_2.weight"]= state_dict_g["7.weight"]
new_state_dict_g["conv2_2.bias"]= state_dict_g["7.bias"]
new_state_dict_g["conv3_1.weight"]= state_dict_g["10.weight"]
new_state_dict_g["conv3_1.bias"]= state_dict_g["10.bias"]
new_state_dict_g["conv3_2.weight"]= state_dict_g["12.weight"]
new_state_dict_g["conv3_2.bias"]= state_dict_g["12.bias"]
new_state_dict_g["conv3_3.weight"]= state_dict_g["14.weight"]
new_state_dict_g["conv3_3.bias"]= state_dict_g["14.bias"]
new_state_dict_g["conv4_1.weight"]= state_dict_g["17.weight"]
new_state_dict_g["conv4_1.bias"]= state_dict_g["17.bias"]
new_state_dict_g["conv4_2.weight"]= state_dict_g["19.weight"]
new_state_dict_g["conv4_2.bias"]= state_dict_g["19.bias"]
new_state_dict_g["conv4_3.weight"]= state_dict_g["21.weight"]
new_state_dict_g["conv4_3.bias"]= state_dict_g["21.bias"]
new_state_dict_g["conv5_1.weight"]= state_dict_g["24.weight"]
new_state_dict_g["conv5_1.bias"]= state_dict_g["24.bias"]
new_state_dict_g["conv5_2.weight"]= state_dict_g["26.weight"]
new_state_dict_g["conv5_2.bias"]= state_dict_g["26.bias"]
new_state_dict_g["conv5_3.weight"]= state_dict_g["28.weight"]
new_state_dict_g["conv5_3.bias"]= state_dict_g["28.bias"]
vgg.load_state_dict(new_state_dict_g)

vgg = torch.nn.DataParallel(vgg)
vgg.cuda()
for param in vgg.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg)
loss_network.eval()


# net_var.load_state_dict(torch.load('./turb_drpv1/{}_best'.format(category)))
# --- Load the network weight --- #


if os.path.exists('./{}'.format(exp_name))==False:
    os.mkdir('./{}'.format(exp_name)) 

val_data_dir = '/media/labuser/sdd/ICIP_Turbulence_files/cropped_256/'
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=8)

### area comment ends
# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

config.__init__()
# config.warp_gen()

# --- Load training data and validation/test data --- #
labeled_name = 'train.txt'
val_filename = 'test.txt'
# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=8)

num_labeled = train_batch_size*len(train_data_loader) # number of labeled images
# --- Previous PSNR and SSIM in testing --- #
# old_val_psnr, old_val_ssim = validation(net, net_var, val_data_loader, device, distance)
old_val_psnr,old_val_ssim=10,10
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

def angular_loss(X,Y):
    shape = X.size()
    loss = torch.mean(torch.acos(F.cosine_similarity(X.view(-1,shape[1]*shape[2]*shape[3]),Y.view(-1,shape[1]*shape[2]*shape[3]))))
    return loss
def gradient(y):
    gradient_h=y[:, :, :, :-1] - y[:, :, :, 1:]
    gradient_v=y[:, :, :-1, :] - y[:, :, 1:, :]

    return gradient_h, gradient_v
def gradient_loss(X,Y):
    gradient_h_X, gradient_v_X = gradient(X)
    gradient_h_Y, gradient_v_Y = gradient(Y)
    loss = angular_loss(gradient_v_X,gradient_v_Y) + angular_loss(gradient_h_X,gradient_h_Y)
    return loss


for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)
    if (epoch-epoch_start)% 4 == 0:
        # config.init()
        config.warp_gen()
    batch_id=0
#-------------------------------------------------------------------------------------------------------------
    #Labeled phase
    for  train_data in tqdm(train_data_loader):
        batch_id=batch_id+1
        input_image, gt, imgid = train_data
        input_image = input_image.cuda()
        gt = gt.cuda()
        input_imagex2 = torch.nn.functional.interpolate(input_image,scale_factor=0.5)
        gt_x2 = torch.nn.functional.interpolate(gt,scale_factor=0.5)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        net_var.train()
        # with torch.no_grad():
        batch_size = train_batch_size
        shape = input_image.size()
        Var1 = torch.zeros([shape[0],shape[1],shape[2],shape[3]], dtype=torch.float32)
        Num_itr = 10
        x_output = torch.zeros([Num_itr,shape[0],shape[1],shape[2],shape[3]], dtype=torch.float32)
        for k in range(Num_itr):
            pred_image,pred_imagex2 = net_var(input_image,input_imagex2)
            x_output[k,:,:,:,:].copy_(pred_image.data)
        for j in range(batch_size):
            tx_output = x_output[:,j,:,:,:]
            tx_output = torch.squeeze(tx_output)
            Var1[j,:,:,:] = torch.std(tx_output,dim=0)

        pred_image,pred_imagex2 = net(input_image,input_imagex2,5*Var1)




        smooth_loss = F.smooth_l1_loss(pred_image, gt) + 0.33*F.smooth_l1_loss(pred_imagex2, gt_x2)
        ang_loss = angular_loss(pred_image, gt) + 0.33*angular_loss(pred_imagex2, gt_x2)
        grad_loss = gradient_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt) + 0.33*loss_network(pred_imagex2, gt_x2) 

        loss = smooth_loss + lambda_loss*perceptual_loss + ang_loss + 0.25*grad_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        if (batch_id+1)%(int(0.5*len(train_data_loader)+2))==0:
            val_psnr, val_ssim = validation(net, net_var, val_data_loader, device, category)
            train_psnr = sum(psnr_list) / len(psnr_list)
            one_epoch_time = time.time() - start_time
            print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
            if val_psnr >= old_val_psnr:
                torch.save(net.state_dict(), './{}/{}_best'.format(exp_name,category))
                old_val_psnr = val_psnr


        # if not (batch_id % 100):
        #     print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), './{}/{}'.format(exp_name,category))
    torch.save(net.state_dict(), './{}/{}_epoch{}'.format(exp_name,category,epoch))
    torch.save(net_var.state_dict(), './{}/{}_var_epoch{}'.format(exp_name,category,epoch))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, net_var, val_data_loader, device, distance)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, distance)

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(net.state_dict(), './{}/{}_best'.format(exp_name,category))
        old_val_psnr = val_psnr