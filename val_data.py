
# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import os

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        train_list=os.listdir(val_data_dir)
        input_names=train_list#[val_data_dir+_ for _ in train_list]
      

        self.input_names = input_names
        self.gt_names = input_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]#[:-4]+'gt.png'
        input_img = Image.open(os.path.join(self.val_data_dir, input_name) )
        gt_img = Image.open(os.path.join(self.val_data_dir, input_name) )
        # print(input_name)


        wd_new = int(16*np.ceil(input_img.size[0]/16.0))
        ht_new = int(16*np.ceil(input_img.size[1]/16.0))
        if ht_new>256:
            ht_new = 256
        if wd_new>256:
            wd_new = 256
        

        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
