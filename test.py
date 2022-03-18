"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: test.py
about: main entrance for validating/testing the GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data import ValData
from face_turv2 import Turb, Turb_mcdrp
from utils import validation,gen_varmaps
import os

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=2, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-category', help='Set image category (distort?)', default='distort', type=str)
args = parser.parse_args()

lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
exp_name = args.exp_name

print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\nlambda_loss: {}\ncategory: {}'
      .format(val_batch_size,lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'distort':
    val_data_dir = './data/test/CelebA_full/'
elif category == 'dehaze':
    val_data_dir = './data/test/dehaze/'
else:
    raise Exception('Wrong image category. Set it to derain or dehaze for RESIDE dateset.')


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #
val_filename = 'test_full.txt'
val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
net = Turb()
net_var = Turb_mcdrp()


# --- Multi-GPU --- #
net = net.cuda()
net_var = net_var.cuda()
net = nn.DataParallel(net)
net_var = nn.DataParallel(net_var)


# --- Load the network weight --- #
net.load_state_dict(torch.load('./{}/{}_best'.format(exp_name,category)))
net_var.load_state_dict(torch.load('./turb_drpv1/{}_best'.format(category)))


# --- Use the evaluation model in testing --- #
net.eval()
if os.path.exists('./{}_results/'.format(category))==False:
	os.mkdir('./{}_results/'.format(category)) 
	os.mkdir('./{}_results/{}/'.format(category,exp_name))
	# os.mkdir('./{}_results/{}/rain/'.format(category,exp_name))
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation(net,net_var, val_data_loader, device, category, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
