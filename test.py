

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
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
import itertools
import config_tdrn as config
import os
import pdb
plt.switch_backend('agg')
from tqdm import tqdm

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-crop_size', help='Set the crop_size', default=[256,256], nargs='+', type=int)
parser.add_argument('-val_batch_size', help='Set the training batch size', default=1, type=int)
parser.add_argument('-checkpoint', help='Pretrained model location', default='./checkpoint/', type=str)
parser.add_argument('-val_dir', help='test_data_dir', default='./test_images/', type=str)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default='./results/',type=str)
args = parser.parse_args()

checkpoint=args.checkpoint
crop_size = args.crop_size
val_batch_size = args.val_batch_size
exp_name = args.exp_name
val_dir =args.val_dir
checkpoint=args.checkpoint


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Turb()
net_var = Turb_mcdrp()

# --- Build optimizer --- #

# --- Multi-GPU --- #
net = net.cuda()
net_var = net_var.cuda()
net = nn.DataParallel(net)
net_var = nn.DataParallel(net_var)

# --- Load the network weight --- #


if os.path.exists('./{}'.format(exp_name))==False:
    os.mkdir('./{}'.format(exp_name)) 

## area to comment starts
try:
    net.load_state_dict(torch.load(os.path.join(checkpoint,'model_best')))
    net_var.load_state_dict(torch.load(os.path.join(checkpoint,'model_best_var')))

    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

val_data_loader = DataLoader(ValData(val_dir), batch_size=val_batch_size, shuffle=False, num_workers=8)
old_val_psnr, old_val_ssim = validation(net, net_var, val_data_loader, device, '128')

