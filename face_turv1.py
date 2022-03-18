import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable



def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 2
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)
class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv3(self.relu(self.bn3(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_bilinear(out, scale_factor=2)



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out







class scale_residue_est(nn.Module):
    def __init__(self):
        super(scale_residue_est, self).__init__()

        self.conv1 = BottleneckBlock(32, 32)
        self.trans_block1 = TransitionBlock3(64, 32)
        self.conv2 = BottleneckBlock(32, 32)
        self.trans_block2 = TransitionBlock3(64, 32)
        self.conv3 = BottleneckBlock(32, 32)
        self.trans_block3 = TransitionBlock3(64, 32)
        self.conv_refin = nn.Conv2d(32, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        x4 = self.relu((self.conv_refin(x3)))
        residual = self.tanh(self.refine3(x4))

        return residual



class BottleneckBlockcf(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockcf, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class scale_kernel_conf(nn.Module):
    def __init__(self):
        super(scale_kernel_conf, self).__init__()

        self.conv1 = nn.Conv2d(6,16,3,1,1)#BottleneckBlock(35, 16)
        self.trans_block1 = TransitionBlock1(16, 16)
        self.conv2 = BottleneckBlockcf(16, 32)
        self.trans_block2 = TransitionBlock1(32, 16)
        self.conv3 = BottleneckBlockcf(16, 32)
        self.trans_block3 = TransitionBlock1(32, 16)
        self.conv4 = BottleneckBlockcf(16, 32)
        self.trans_block4 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 3, 1, 1, 0)
        self.sig = torch.nn.Sigmoid()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x,target):
        x1=self.conv1(torch.cat([x,target],1))
        x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        x4=self.conv3(x3)
        x4 = self.trans_block4(x4)
        #print(x4.size())
        residual = self.sig(self.conv_refin(self.sig(F.avg_pool2d(x4,16))))
        #print(residual)
        residual = F.upsample_nearest(residual, scale_factor=128)
        #print(residual.size())
        return residual



def gradient(y):
    gradient_h=y[:, :, :, :-1] - y[:, :, :, 1:]
    gradient_v=y[:, :, :-1, :] - y[:, :, 1:, :]

    return gradient_h, gradient_v

def TV(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_v=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_v

class scale_conf(nn.Module):
    def __init__(self):
        super(scale_conf, self).__init__()

        self.conv1_t = nn.Conv2d(3,16,3,1,1)#BottleneckBlock(35, 16)
        #self.trans_block1 = TransitionBlock3(51, 8)
        self.conv1_x = nn.Conv2d(3,16,3,1,1)
        self.conv2 = BottleneckBlock(16, 16)
        self.trans_block2 = TransitionBlock3(32, 16)
        self.conv3 = BottleneckBlock(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.sig = torch.nn.Sigmoid()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x,target):
        x1_t=self.conv1_t(target)
        x1_x=self.conv1_x(x)
        #x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        residual = self.sig(self.refine3(x3))

        return residual

class scale_conf_grad(nn.Module):
    def __init__(self):
        super(scale_conf_grad, self).__init__()

        self.conv1_t = nn.Conv2d(3,16,3,1,1)#BottleneckBlock(35, 16)
        #self.trans_block1 = TransitionBlock3(51, 8)
        self.conv1_x = nn.Conv2d(3,16,3,1,1)
        self.conv2 = BottleneckBlock(32, 16)
        self.trans_block2 = TransitionBlock3(48, 16)
        self.conv3 = BottleneckBlock(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.sig = torch.nn.Sigmoid()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x,target):
        x1_t=self.conv1_t(target)
        x1_x=self.conv1_x(x)
        #x1 = self.trans_block1(x1)
        x2=self.conv2(torch.cat([x1_t, x1_x], 1))
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        residual = self.sig(self.refine3(x3))

        return residual



class Deform_v1(nn.Module):
    def __init__(self):
        super(Deform_v1, self).__init__()
        dropout_p = 0.25
        ############# Block1-scale 1.0  ##############
        self.dense_block1=BottleneckBlock1(3,29)

        ############# Block2-scale 0.50  ##############
        self.trans_block2=TransitionBlock1(32,32)
        self.dense_block2=BottleneckBlock1(35,32,dropout_p)
        self.trans_block2_o=TransitionBlock3(67,32)

        ############# Block3-scale 0.250  ##############
        self.trans_block3=TransitionBlock1(32,64)
        self.dense_block3=BottleneckBlock1(64,64,dropout_p)
        self.trans_block3_o=TransitionBlock3(128,128)

        ############# Block4-scale 0.1250  ##############
        #self.trans_block4=TransitionBlock1(64,128)
        self.dense_block4=BottleneckBlock1(128,128,dropout_p)
        self.trans_block4_o=TransitionBlock3(256,128)

        ############# Block5-scale 0.3125  ##############
        #self.trans_block5=TransitionBlock1(128,128)
        self.dense_block5=BottleneckBlock1(128,128,dropout_p)
        self.trans_block5_o=TransitionBlock3(256,128)

        ############# Block6-scale 0.3125  ##############
        self.dense_block6=BottleneckBlock1(128,128,dropout_p)
        self.trans_block6_o=TransitionBlock3(256,128)

        ############# Block7-scale 0.125  ############## 7--4 skip connection
        #self.trans_block7=TransitionBlock(128,128)
        self.dense_block7=BottleneckBlock1(256,64,dropout_p)
        self.trans_block7_o=TransitionBlock3(320,64)

        ############# Block8-scale 0.25  ############## 8--3 skip connection
        #self.trans_block8=TransitionBlock(64,64)
        self.dense_block8=BottleneckBlock1(192,64,dropout_p)
        self.trans_block8_o=TransitionBlock3(256,64)

        ############# Block9-scale 0.5  ############## 9--2 skip connection
        self.trans_block9=TransitionBlock(64,32)
        self.dense_block9=BottleneckBlock1(64,32,dropout_p)
        self.trans_block9_o=TransitionBlock3(96,32)

        ############# Block10-scale 1.0  ############## 10--1 skip connection
        self.trans_block10=TransitionBlock(32,32)
        self.dense_block10=BottleneckBlock1(67,32)
        self.trans_block10_o=TransitionBlock3(99,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.tanh=nn.Tanh()

        self.conv_refin_in=nn.Conv2d(3,16,3,1,1)
        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.conf_im = scale_conf_grad()
        self.conf_gradh = scale_conf_grad()
        self.conf_gradv = scale_conf_grad()
        self.res_est = scale_residue_est()

    def forward(self, x,x_64, target):

        #Size - 1.0
        x1=(self.dense_block1(x))

        #Size - 0.5        
        x2_i=self.trans_block2(x1)
        x2_i=self.dense_block2(torch.cat([x2_i, x_64], 1))
        x2=self.trans_block2_o(x2_i)

        #Size - 0.25
        x3_i=self.trans_block3(x2)
        x3_i=self.dense_block3(x3_i)
        x3=self.trans_block3_o(x3_i)

        #Size - 0.125
        #x4_i=self.trans_block4(x3)
        x4_i=self.dense_block4(x3)
        x4=self.trans_block4_o(x4_i)

        #x5_i=self.trans_block5(x4)
        x5_i=self.dense_block5(x4)
        x5=self.trans_block5_o(x5_i)

        x6_i=self.dense_block6(x5)
        x6=self.trans_block6_o(x6_i)

        #x7_i=self.trans_block7(x6)
        # print(x4.size())
        # print(x7_i.size())
        x7_i=self.dense_block7(torch.cat([x6, x4], 1))
        x7=self.trans_block7_o(x7_i)

        #x8_i=self.trans_block8(x7)
        x8_i=self.dense_block8(torch.cat([x7, x3], 1))
        x8=self.trans_block8_o(x8_i)

        x9_i=self.trans_block9(x8)
        x9_i=self.dense_block9(torch.cat([x9_i, x2], 1))
        x9=self.trans_block9_o(x9_i)

        xhat_64_res = self.res_est(x9)

        xhat_64 = x_64 - xhat_64_res

        x10_i=self.trans_block10(x9)
        x10_i=self.dense_block10(torch.cat([x10_i, x1,x], 1))
        x10=self.trans_block10_o(x10_i)
        x11 = self.relu(self.conv_refin_in(x)) - self.relu(self.conv_refin(x10))
        x11 = self.relu((self.conv_refin(x11)))
        residual=self.tanh(self.refine3(x11))
        clean = x - residual
        clean=self.relu(self.refineclean1(clean))
        clean=self.tanh(self.refineclean2(clean))
        Est_h,Est_v = gradient(clean)
        Tar_h, Tar_v = gradient(target)
        conf_gardv = self.conf_gradv(Est_v,Tar_v)
        Est_v_eff = conf_gardv*Est_v + (1-conf_gardv)*Tar_v
        conf_gardh = self.conf_gradh(Est_h,Tar_h)
        Est_h_eff = conf_gardh*Est_h + (1-conf_gardh)*Tar_h
        conf_mp = self.conf_im(clean,target)

        return residual, clean,xhat_64, x6,Est_v_eff,Est_h_eff,Tar_v,Tar_h,conf_mp,conf_gardv,conf_gardh


class Deblur_v1(nn.Module):
    def __init__(self):
        super(Deblur_v1, self).__init__()
        dropout_p = 0.5
        ############# Block1-scale 1.0  ##############
        self.dense_block1=BottleneckBlock1(3,29)

        ############# Block2-scale 0.50  ##############
        self.trans_block2=TransitionBlock1(32,32)
        self.dense_block2=BottleneckBlock1(35,32,dropout_p)
        self.trans_block2_o=TransitionBlock3(67,32)

        ############# Block3-scale 0.250  ##############
        self.trans_block3=TransitionBlock1(32,64)
        self.dense_block3=BottleneckBlock1(64,64,dropout_p)
        self.trans_block3_o=TransitionBlock3(128,128)

        ############# Block4-scale 0.1250  ##############
        #self.trans_block4=TransitionBlock1(64,128)
        self.dense_block4=BottleneckBlock1(128,128,dropout_p)
        self.trans_block4_o=TransitionBlock3(256,128)

        ############# Block5-scale 0.3125  ##############
        #self.trans_block5=TransitionBlock1(128,128)
        self.dense_block5=BottleneckBlock1(128,128,dropout_p)
        self.trans_block5_o=TransitionBlock3(256,128)

        ############# Block6-scale 0.3125  ##############
        self.dense_block6=BottleneckBlock1(128,128,dropout_p)
        self.trans_block6_o=TransitionBlock3(256,128)

        ############# Block7-scale 0.125  ############## 7--4 skip connection
        #self.trans_block7=TransitionBlock(128,128)
        self.dense_block7=BottleneckBlock1(256,64,dropout_p)
        self.trans_block7_o=TransitionBlock3(320,64)

        ############# Block8-scale 0.25  ############## 8--3 skip connection
        #self.trans_block8=TransitionBlock(64,64)
        self.dense_block8=BottleneckBlock1(192,64,dropout_p)
        self.trans_block8_o=TransitionBlock3(256,64)

        ############# Block9-scale 0.5  ############## 9--2 skip connection
        self.trans_block9=TransitionBlock(64,32)
        self.dense_block9=BottleneckBlock1(64,32,dropout_p)
        self.trans_block9_o=TransitionBlock3(96,32)

        ############# Block10-scale 1.0  ############## 10--1 skip connection
        self.trans_block10=TransitionBlock(32,32)
        self.dense_block10=BottleneckBlock1(67,32)
        self.trans_block10_o=TransitionBlock3(99,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.tanh=nn.Tanh()

        self.conv_refin_in=nn.Conv2d(3,16,3,1,1)
        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.conf_ker = scale_kernel_conf()
        self.res_est = scale_residue_est()

    def forward(self, x,x_64):

        #Size - 1.0
        x1=(self.dense_block1(x))

        #Size - 0.5        
        x2_i=self.trans_block2(x1)
        x2_i=self.dense_block2(torch.cat([x2_i, x_64], 1))
        x2=self.trans_block2_o(x2_i)

        #Size - 0.25
        x3_i=self.trans_block3(x2)
        x3_i=self.dense_block3(x3_i)
        x3=self.trans_block3_o(x3_i)

        #Size - 0.125
        #x4_i=self.trans_block4(x3)
        x4_i=self.dense_block4(x3)
        x4=self.trans_block4_o(x4_i)

        #x5_i=self.trans_block5(x4)
        x5_i=self.dense_block5(x4)
        x5=self.trans_block5_o(x5_i)

        x6_i=self.dense_block6(x5)
        x6=self.trans_block6_o(x6_i)

        #x7_i=self.trans_block7(x6)
        # print(x4.size())
        # print(x7_i.size())
        x7_i=self.dense_block7(torch.cat([x6, x4], 1))
        x7=self.trans_block7_o(x7_i)

        #x8_i=self.trans_block8(x7)
        x8_i=self.dense_block8(torch.cat([x7, x3], 1))
        x8=self.trans_block8_o(x8_i)

        x9_i=self.trans_block9(x8)
        x9_i=self.dense_block9(torch.cat([x9_i, x2], 1))
        x9=self.trans_block9_o(x9_i)

        xhat_64_res = self.res_est(x9)

        xhat_64 = x_64 - xhat_64_res

        x10_i=self.trans_block10(x9)
        x10_i=self.dense_block10(torch.cat([x10_i, x1,x], 1))
        x10=self.trans_block10_o(x10_i)
        x11 = self.relu(self.conv_refin_in(x)) - self.relu(self.conv_refin(x10))
        x11 = self.relu((self.conv_refin(x11)))
        residual=self.tanh(self.refine3(x11))
        clean = x - residual
        clean=self.relu(self.refineclean1(clean))
        clean=self.tanh(self.refineclean2(clean))



        return residual, clean,xhat_64


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()