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

import pdb
import math



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


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist block of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality
        self.convin = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv1 = nn.Conv2d(inplanes, D*C*scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C*scale)
        self.SE = SEBlock(planes,C)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, stride = stride, padding=1, groups=C, bias=False))
          bns.append(nn.BatchNorm2d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D*C*scale, planes  , kernel_size=1, stride=1, padding=0, bias=False)        
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = self.convin(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)
        #out = self.SE(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #pdb.set_trace()
        out += residual
        out = self.relu(out)

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


class BottleneckBlockdls(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockdls, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.sharewconv1 = ShareSepConv(3)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.sharewconv2 = ShareSepConv(3)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.sharewconv2(self.relu(self.bn4(out))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlockdl(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockdl, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, dilation=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.sharewconv1 = ShareSepConv(3)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.sharewconv = ShareSepConv(3)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.sharewconv1(self.relu(self.bn3(out))))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.sharewconv(self.relu(self.bn4(out))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlockrs1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockrs1, self).__init__()
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
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlockrs(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockrs, self).__init__()
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
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
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
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)





class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
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
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out







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


class TransitionBlockbil(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockbil, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = F.upsample_bilinear(out, scale_factor=2)
        return out




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

def Second_grad(y):
    gradient_xy=4*y[:, :, 1:-1, 1:-1] - y[:, :, 1:-1, :-2]- y[:, :, 1:-1, 2:]- y[:, :, :-2, 1:-1]- y[:, :, 2:, 1:-1]

    return gradient_xy

def TV(y):
    gradient_h=y[:, :, :, :-1] - y[:, :, :, 1:]
    gradient_v=y[:, :, :-1, :] - y[:, :, 1:, :]

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
        x1_t=self.relu(self.conv1_t(target))
        x1_x=self.relu(self.conv1_x(x))
        #x1 = self.trans_block1(x1)
        x2=self.relu(self.conv2(torch.cat([x1_t, x1_x], 1)))
        x2 = self.trans_block2(x2)
        x3=self.relu(self.conv3(x2))
        x3 = self.trans_block3(x3)
        residual = self.sig(self.refine3(x3))

        return residual



class Turb_mcdrp(nn.Module):
    def __init__(self):
        super(Turb_mcdrp, self).__init__()
        self.baseWidth = 16#4#16
        self.cardinality = 16#8#16
        self.scale = 6#4#5
        self.stride = 1
        dropout_p = 0.25
        self.conv_input=nn.Conv2d(3,32,3,1,1)
        self.dense_block1=Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block1=BottleneckBlockrs(9,55)
        self.trans_block1=TransitionBlock1(64,64,dropout_p)

        ############# Block2-down 32-32  ##############
        self.dense_block2=Bottle2neckX(67,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block2=BottleneckBlockrs1(67,64)
        self.trans_block2=TransitionBlock3(131,64,dropout_p)

        ############# Block3-down  16-16 ##############
        self.dense_block3=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3=BottleneckBlockdl(64,64)
        self.trans_block3=TransitionBlock3(128,64,dropout_p)
        
        self.dense_block3_1=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3_1=BottleneckBlockdl(64,64)
        self.trans_block3_1=TransitionBlock3(128,64,dropout_p)

        self.dense_block3_2=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3_2=BottleneckBlockdl(64,64)
        self.trans_block3_2=TransitionBlock3(128,64,dropout_p)

        self.dense_block3_3=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3_2=BottleneckBlockdl(64,64)
        self.trans_block3_3=TransitionBlock3(128,64,dropout_p)

        ############# Block4-up  8-8  ##############
        self.dense_block4=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block4=BottleneckBlockdl(64,64)
        self.trans_block4=TransitionBlock3(128,64,dropout_p)

        ############# Block5-up  16-16 ##############
        self.dense_block5=Bottle2neckX(128,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block5=BottleneckBlockrs1(128,64)
        self.trans_block5=TransitionBlockbil(195,64,dropout_p)

        self.dense_block6=Bottle2neckX(67,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block6=BottleneckBlockrs(73,64)
        self.trans_block6=TransitionBlock3(131,16,dropout_p)


        self.conv_refin=nn.Conv2d(24,16,3,1,1)
        self.conv_refin_in=nn.Conv2d(3,16,3,1,1)
        self.conv_refin_in64=nn.Conv2d(3,16,3,1,1)
        self.conv_refin64=nn.Conv2d(192,16,3,1,1)
        self.tanh=nn.Tanh()


        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)

        self.upsample =  F.upsample_bilinear

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)
        
        self.conv11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_31 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm


    def forward(self, x,x_64):
        ## 128x128
        xi = self.relu(self.conv_input(x))
        x1=self.dense_block1(xi)
        x1=self.trans_block1(x1)

        x2=(self.dense_block2(torch.cat([x1,x_64],1)))
        # print(x2.size(),x1.size(),x_64.size())
        x2=self.trans_block2(x2)


        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        x3_1 = (self.dense_block3_1(x3))
        x3_1 = self.trans_block3_1(x3_1)
        x3_2 = (self.dense_block3_2(x3_1))
        x3_2 = self.trans_block3_2(x3_2)
        x3_3 = (self.dense_block3_2(x3_2))
        x3_3 = self.trans_block3_2(x3_3)


        x4=(self.dense_block4(x3_3))
        x4=self.trans_block4(x4)
        x5_in=torch.cat([x4, x1], 1)
        x5_i=(self.dense_block5(x5_in))
        
        xhat64 = self.relu(self.conv_refin_in64(x_64)) - self.relu(self.conv_refin64(x5_i))
        xhat64 = self.tanh(self.refine3(xhat64))
        x5=self.trans_block5(torch.cat([x5_i,xhat64],1))
        x6=(self.dense_block6(torch.cat([x5,x],1)))
        x6=(self.trans_block6(x6))
        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv21(x2))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv31(x3))), size=shape_out)
        x3_11 = self.upsample(self.relu((self.conv3_11(x3_1))), size=shape_out)
        x3_21 = self.upsample(self.relu((self.conv3_21(x3_2))), size=shape_out)
        x3_31 = self.upsample(self.relu((self.conv3_21(x3_3))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv41(x4))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv51(x5))), size=shape_out)
        x6=torch.cat([x6,x51,x41,x3_31,x3_21,x3_11,x31,x21,x11],1)
        x7=self.relu(self.conv_refin_in(x)) - self.relu(self.conv_refin(x6))
        residual=self.tanh(self.refine3(x7))
        clean = x - residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.tanh(self.refineclean2(clean))
        
        clean64 = x_64 - xhat64
        clean64 = self.relu(self.refineclean1(clean64))
        clean64 = self.tanh(self.refineclean2(clean64))

        return clean,clean64


class Turb(nn.Module):
    def __init__(self):
        super(Turb, self).__init__()
        self.baseWidth = 16#4#16
        self.cardinality = 16#8#16
        self.scale = 6#4#5
        self.stride = 1
        dropout_p = 0.0
        self.conv_input=nn.Conv2d(6,32,3,1,1)
        self.dense_block1=Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block1=BottleneckBlockrs(9,55)
        self.trans_block1=TransitionBlock1(64,64,dropout_p)

        ############# Block2-down 32-32  ##############
        self.dense_block2=Bottle2neckX(67,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block2=BottleneckBlockrs1(67,64)
        self.trans_block2=TransitionBlock3(131,64,dropout_p)

        ############# Block3-down  16-16 ##############
        self.dense_block3=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3=BottleneckBlockdl(64,64)
        self.trans_block3=TransitionBlock3(128,64,dropout_p)
        
        self.dense_block3_1=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3_1=BottleneckBlockdl(64,64)
        self.trans_block3_1=TransitionBlock3(128,64,dropout_p)

        self.dense_block3_2=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3_2=BottleneckBlockdl(64,64)
        self.trans_block3_2=TransitionBlock3(128,64,dropout_p)

        self.dense_block3_3=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block3_2=BottleneckBlockdl(64,64)
        self.trans_block3_3=TransitionBlock3(128,64,dropout_p)

        ############# Block4-up  8-8  ##############
        self.dense_block4=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block4=BottleneckBlockdl(64,64)
        self.trans_block4=TransitionBlock3(128,64,dropout_p)

        ############# Block5-up  16-16 ##############
        self.dense_block5=Bottle2neckX(128,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block5=BottleneckBlockrs1(128,64)
        self.trans_block5=TransitionBlockbil(195,64,dropout_p)

        self.dense_block6=Bottle2neckX(70,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        # self.dense_block6=BottleneckBlockrs(73,64)
        self.trans_block6=TransitionBlock3(134,16,dropout_p)


        self.conv_refin=nn.Conv2d(24,16,3,1,1)
        self.conv_refin_in=nn.Conv2d(3,16,3,1,1)
        self.conv_refin_in64=nn.Conv2d(3,16,3,1,1)
        self.conv_refin64=nn.Conv2d(192,16,3,1,1)
        self.tanh=nn.Tanh()
        self.sig=nn.Sigmoid()


        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)

        self.upsample =  F.upsample_bilinear

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)
        
        self.conv11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_31 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm


    def forward(self, x,x_64,var):
        ## 128x128
        xi = self.relu(self.conv_input(torch.cat([var,x],1)))
        x1=self.dense_block1(xi)
        x1=self.trans_block1(x1)

        x2=(self.dense_block2(torch.cat([x1,x_64],1)))
        # print(x2.size(),x1.size(),x_64.size())
        x2=self.trans_block2(x2)


        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        x3_1 = (self.dense_block3_1(x3))
        x3_1 = self.trans_block3_1(x3_1)
        x3_2 = (self.dense_block3_2(x3_1))
        x3_2 = self.trans_block3_2(x3_2)
        x3_3 = (self.dense_block3_2(x3_2))
        x3_3 = self.trans_block3_2(x3_3)


        x4=(self.dense_block4(x3_3))
        x4=self.trans_block4(x4)
        x5_in=torch.cat([x4, x1], 1)
        x5_i=(self.dense_block5(x5_in))
        
        xhat64 = self.relu(self.conv_refin_in64(x_64)) - self.relu(self.conv_refin64(x5_i))
        xhat64 = self.tanh(self.refine3(xhat64))
        x5=self.trans_block5(torch.cat([x5_i,xhat64],1))
        x6=(self.dense_block6(torch.cat([x5,x,var],1)))
        x6=(self.trans_block6(x6))
        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv21(x2))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv31(x3))), size=shape_out)
        x3_11 = self.upsample(self.relu((self.conv3_11(x3_1))), size=shape_out)
        x3_21 = self.upsample(self.relu((self.conv3_21(x3_2))), size=shape_out)
        x3_31 = self.upsample(self.relu((self.conv3_21(x3_3))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv41(x4))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv51(x5))), size=shape_out)
        x6=torch.cat([x6,x51,x41,x3_31,x3_21,x3_11,x31,x21,x11],1)
        x7=self.relu(self.conv_refin_in(x)) - self.relu(self.conv_refin(x6))
        residual=self.tanh(self.refine3(x7))
        clean = x - residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.sig(self.refineclean2(clean))
        
        clean64 = x_64 - xhat64
        clean64 = self.relu(self.refineclean1(clean64))
        clean64 = self.sig(self.refineclean2(clean64))

        return clean,clean64