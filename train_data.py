

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import pdb
import numpy as np
import re
import os
import config_tdrn as config
import random
import torch
from scipy import signal

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        train_list=sorted(os.listdir(train_data_dir))
        train_list=train_list[:10000]
        input_names=train_list#[train_data_dir+_ for _ in train_list]
      
        self.input_names = input_names
        self.gt_names = input_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/',input_name)[-1][:-4]

        input_img = Image.open(os.path.join(self.train_data_dir, input_name))

        try:
            gt_img = Image.open(os.path.join(self.train_data_dir,gt_name))
        except:
            gt_img = Image.open(os.path.join(self.train_data_dir + gt_name)).convert('RGB')

        width, height = input_img.size
        # print(width,height)
        if width != 256 or height != 256 :
            input_img = input_img.resize((256+32,256+42), Image.ANTIALIAS)
            gt_img = gt_img.resize((256+32,256+42), Image.ANTIALIAS)
        # print(width,height)

        # if width < crop_width and height < crop_height :
        #     input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
        #     gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        # elif width < crop_width :
        #     input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
        #     gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        # elif height < crop_height :
        #     input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
        #     gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size
        # --- x,y coordinate of left-top corner --- #
        # x, y = 16,21
        # input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        # gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        # print(input_img.size)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_img = transform_input(input_img)
        gt = transform_gt(gt_img)

        ch,w, h = input_img.size()

        input_img= input_img.numpy()
        
        
        index_blr = random.randint(0,config.Num_k-10)
        input_img[0,:,:]= signal.convolve(input_img[0,:,:],config.kernels[index_blr,:,:],mode='same')
        input_img[1,:,:]= signal.convolve(input_img[1,:,:],config.kernels[index_blr,:,:],mode='same')
        input_img[2,:,:]= signal.convolve(input_img[2,:,:],config.kernels[index_blr,:,:],mode='same')

        index_d = random.randint(0,config.Num_D-1)
        xx,yy = np.meshgrid(np.arange(0,h),np.arange(0,w))
        # pdb.set_trace()
        # print(config.Warp_mat[index_d,0,:,:].shape,xx.shape,yy.shape)
        X_new = xx + config.Warp_mat[index_d,0,:,:]
        Y_new = yy + config.Warp_mat[index_d,1,:,:]
        # print(X_new.max(),Y_new.max())

        input_img[0,:,:] = config.warp(input_img[0,:,:],X_new,Y_new)
        input_img[1,:,:] = config.warp(input_img[1,:,:],X_new,Y_new)
        input_img[2,:,:] = config.warp(input_img[2,:,:],X_new,Y_new)
        # input_img = torch.from_numpy(input_img)
        input_img = torch.from_numpy(input_img)
        x, y = 21,16
        input_img = input_img[:,x:x+ crop_width, y:y + crop_height]
        gt = gt[:,x:x+ crop_width, y:y + crop_height]

        # --- Check the channel is 3 or not --- #
        if list(input_img.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return input_img, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

