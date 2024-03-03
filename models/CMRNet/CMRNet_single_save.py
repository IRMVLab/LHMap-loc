"""
Original implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren

Modified version (CMRNet) by Daniele Cattaneo

"""

import dataclasses
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/'  # a writable directory

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.CMRNet.correlation_package.correlation import Correlation


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class CMRNet(nn.Module):
    def __init__(self, image_size, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(CMRNet, self).__init__()
        input_lidar = 1
        self.use_feat_from = use_feat_from
        if use_reflectance:
            input_lidar = 2

        # self.w_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        # self.w_q = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.conv0a = conv(3, 8, kernel_size=3, stride=1)
        self.conv0aa = conv(8, 8, kernel_size=3, stride=1)
        self.conv0b = conv(8, 8, kernel_size=3, stride=1)
        self.conv1a = conv(8, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)


        self.lconv0a = conv(input_lidar, 8, kernel_size=3, stride=1)
        self.lconv0aa = conv(8, 8, kernel_size=3, stride=1)
        self.lconv0b = conv(8, 8, kernel_size=3, stride=1)
        self.lconv1a = conv(8, 16, kernel_size=3, stride=2)
        self.lconv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.lconv1b = conv(16, 16, kernel_size=3, stride=1)
        self.lconv2a = conv(16, 32, kernel_size=3, stride=2)
        self.lconv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.lconv2b = conv(32, 32, kernel_size=3, stride=1)
        self.lconv3a = conv(32, 64, kernel_size=3, stride=2)
        self.lconv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.lconv3b = conv(64, 64, kernel_size=3, stride=1)
        self.lconv4a = conv(64, 96, kernel_size=3, stride=2)
        self.lconv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.lconv4b = conv(96, 96, kernel_size=3, stride=1)
        self.lconv5a = conv(96, 128, kernel_size=3, stride=2)
        self.lconv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.lconv5b = conv(128, 128, kernel_size=3, stride=1)
        self.lconv6a = conv(128, 196, kernel_size=3, stride=2)
        self.lconv6aa = conv(196, 196, kernel_size=3, stride=1)
        self.lconv6b = conv(196, 196, kernel_size=3, stride=1)

        self.llconv1a = conv(input_lidar, 16, kernel_size=3, stride=2)
        self.llconv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.llconv1b = conv(16, 16, kernel_size=3, stride=1)
        self.llconv2a = conv(16, 32, kernel_size=3, stride=2)
        self.llconv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.llconv2b = conv(32, 32, kernel_size=3, stride=1)
        self.llconv3a = conv(32, 64, kernel_size=3, stride=2)
        self.llconv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.llconv3b = conv(64, 64, kernel_size=3, stride=1)
        self.llconv4a = conv(64, 96, kernel_size=3, stride=2)
        self.llconv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.llconv4b = conv(96, 96, kernel_size=3, stride=1)
        self.llconv5a = conv(96, 128, kernel_size=3, stride=2)
        self.llconv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.llconv5b = conv(128, 128, kernel_size=3, stride=1)
        self.llconv6a = conv(128, 196, kernel_size=3, stride=2)
        self.llconv6aa = conv(196, 196, kernel_size=3, stride=1)
        self.llconv6b = conv(196, 196, kernel_size=3, stride=1)

        self.dconv0a = conv(input_lidar, 8, kernel_size=3, stride=1)
        self.dconv0aa = conv(8, 8, kernel_size=3, stride=1)
        self.dconv0b = conv(8, 8, kernel_size=3, stride=1)
        self.dconv1a = conv(8, 16, kernel_size=3, stride=2)
        self.dconv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.dconv1b = conv(16, 16, kernel_size=3, stride=1)
        self.dconv2a = conv(16, 32, kernel_size=3, stride=2)
        self.dconv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.dconv2b = conv(32, 32, kernel_size=3, stride=1)
        self.dconv3a = conv(32, 64, kernel_size=3, stride=2)
        self.dconv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.dconv3b = conv(64, 64, kernel_size=3, stride=1)

        self.deconvdh3 = deconv(64, 32, kernel_size=4, stride=2, padding=1)
        self.convdh2 = conv(64, 16, kernel_size=3, stride=1)
        self.deconvdh2 = deconv(16, 16, kernel_size=4, stride=2, padding=1)
        self.convdh1 = conv(32, 8, kernel_size=3, stride=1)
        self.deconvdh1 = deconv(8, 8, kernel_size=4, stride=2, padding=1)
        self.convdh0 = conv(16, 64, kernel_size=3, stride=1)

        # corr
        self.conv4_0 = conv(81, 128, kernel_size=3, stride=1)
        self.conv4c = conv(209, 256, kernel_size=3, stride=1)
        self.deconvx41 = deconv(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconvx42 = deconv(128, 64, kernel_size=4, stride=2, padding=1)
        self.predict_flow2 = predict_flow(256)
        self.deconv21 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.deconv22 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.conv2_0 = conv(81, 64, kernel_size=3, stride=1)
        self.conv2c = conv(64+81, 64, kernel_size=3, stride=1)
        self.deconvx21 = deconv(64+64, 64, kernel_size=4, stride=2, padding=1)
        self.deconvx22 = deconv(64, 32, kernel_size=4, stride=2, padding=1)
        self.predict_flow0 = predict_flow(64)
        self.deconv01 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.deconv02 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.conv0_0 = conv(49, 32, kernel_size=3, stride=1)
        self.conv0c = conv(49+32, 32, kernel_size=3, stride=1)
        self.convx01 =  conv(64, 64, kernel_size=3, stride=1)

        self.fc01 = nn.Linear(64, 512)
        self.fc01_trasl = nn.Linear(512, 256)
        self.fc01_rot = nn.Linear(512, 256)
        self.fc02_trasl = nn.Linear(256, 3)
        self.fc02_rot = nn.Linear(256, 4)



        self.deconvh1 = deconv(196, 128, kernel_size=4, stride=2, padding=1)
        self.deconvh2 = deconv(128, 96, kernel_size=4, stride=2, padding=1)
        self.convh1 = conv(256, 128, kernel_size=3, stride=1)
        self.convh2 = conv(192, 256, kernel_size=3, stride=1)
        self.convh3 = conv(256, 512, kernel_size=3, stride=1)

        self.convlast1 = deconv(661, 512, kernel_size=4, stride=2, padding=1)

        self.softmax = nn.Softmax(dim = -1)


        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.corr3 = Correlation(pad_size=md-1, kernel_size=1, max_displacement=md-1, stride1=1, stride2=1, corr_multiply=1)

        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32]) # 累和

        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 1:
            self.predict_flow6 = predict_flow(od + dd[4])
            self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + 128 + 4
            self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
            self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)


        self.fc1 = nn.Linear(512, 512)

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # print(f'B,C,H,W:{B},{C},{H},{W}')
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1)#.float()
        grid = grid.type(flo.dtype)

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / max(W-1, 1)-1.0
        vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / max(H-1, 1)-1.0

        vgrid = vgrid.permute(0, 2, 3, 1) # 交换维度
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()).type(flo.dtype)).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output*mask

    def forward(self, rgb, lidar, depth, flagt, u, v, pcl, info): #


        B = lidar.shape[0]

        c10 = self.conv0b(self.conv0aa(self.conv0a(rgb))) 
        c11 = self.conv1b(self.conv1aa(self.conv1a(c10)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11))) 
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12))) 
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13))) 
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14))) 
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15))) 
        #
        u = u.long()
        v = v.long()
        c20 = self.lconv0b(self.lconv0aa(self.lconv0a(lidar)))
        c21 = self.lconv1b(self.lconv1aa(self.lconv1a(c20)))
        c22 = self.lconv2b(self.lconv2aa(self.lconv2a(c21)))
        c23 = self.lconv3b(self.lconv3aa(self.lconv3a(c22)))
        c24 = self.lconv4b(self.lconv4aa(self.lconv4a(c23)))
        # c25 = self.lconv5b(self.lconv5aa(self.lconv5a(c24)))
        # c26 = self.lconv6b(self.lconv6a(self.lconv6aa(c25)))

        c30 = self.dconv0b(self.dconv0aa(self.dconv0a(depth)))
        c31 = self.dconv1b(self.dconv1aa(self.dconv1a(c30)))
        c32 = self.dconv2b(self.dconv2aa(self.dconv2a(c31)))
        c33 = self.dconv3b(self.dconv3aa(self.dconv3a(c32)))

        ch3= self.deconvdh3(c33)
        ch2 = torch.cat((ch3, c32), dim=1)
        ch2 = self.deconvdh2(self.convdh2(ch2)) 
        ch1 = torch.cat((ch2, c31), dim=1) 
        ch1 = self.deconvdh1(self.convdh1(ch1)) 
        ch0 = torch.cat((ch1, c30), dim=1)
        ch0 = self.convdh0(ch0)

        # h_lidar = ch0*flagt
        W, H = ch0.shape[2], ch0.shape[3]
        h_lidar = torch.reshape(ch0,(B,64,W*H))
        h_lidar = self.softmax(h_lidar)
        h_lidar = torch.reshape(h_lidar,(B,64,W,H))

        corr4 = self.corr(c14, c24) 
        corr4 = self.leakyRELU(corr4)
        x = self.conv4c(torch.cat((self.conv4_0(corr4), corr4), 1)) 
        x4 = self.deconvx42(self.deconvx41(x)) 

        flow2 = self.predict_flow2(x)
        up_flow2 = self.deconv22(self.deconv21(flow2)) 
        warp2 = self.warp(c22, up_flow2)
        corr2 = self.leakyRELU(self.corr(c12, warp2))   
        x = torch.cat((self.conv2_0(corr2), corr2), 1) 
        x = self.conv2c(x)                              
        x2 = self.deconvx22(self.deconvx21(torch.cat((x4, x), dim = 1))) 

        flow0 = self.predict_flow0(x) 
        flow0 = flow0 + up_flow2
        up_flow0 = self.deconv02(self.deconv01(flow0))
        warp0 = self.warp(c20, up_flow0) #
        corr0 = self.leakyRELU(self.corr3(c10, warp0))
        x = torch.cat((self.conv0_0(corr0), corr0), 1)  
        x = self.conv0c(x)                              
        x = self.convx01(torch.cat((x2, x),1))          

        x = torch.sum(x * h_lidar, (2,3), keepdim=False)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc01(x))

        transl0 = self.leakyRELU(self.fc01_trasl(x))
        rot0 = self.leakyRELU(self.fc01_rot(x))
        transl0 = self.fc02_trasl(transl0)
        rot0 = self.fc02_rot(rot0)
        rot0 = F.normalize(rot0, dim=1)


        h_lidar = torch.sum(h_lidar,(1),keepdim=True) 
        h_lidar = h_lidar*flagt
        h_lidar = torch.reshape(h_lidar,(B,W*H))
        values, index = h_lidar.topk(5000, dim=-1, largest=True, sorted=False) 
        u = torch.reshape(u,(B, W*H))
        v = torch.reshape(v,(B, W*H))
        f = torch.zeros((B,1,W,H), device='cuda')
        pcl_save = torch.zeros((B,3,5000), device='cuda')

        pcl = torch.reshape(pcl,(B,3,W*H))
        for i in range(B):
            f[i,0,u[i,index[i,:]],v[i,index[i,:]]] = 1
            pcl_save[i,:,:] = pcl[i,:,index[i,:]]

            run =str(info[i][0]) 
     
            file = os.path.join('./KITTI_ODOMETRY/sequences', run,'pcl', f'{info[i][1]:06d}.npy')
            if not os.path.exists(os.path.join('./KITTI_ODOMETRY/sequences', run)):
                os.mkdir(os.path.join('./KITTI_ODOMETRY/sequences',  run))
            if not os.path.exists(os.path.join('./KITTI_ODOMETRY/sequences',run,'pcl')):
                os.mkdir(os.path.join('./KITTI_ODOMETRY/sequences',  run, 'pcl'))
            pcl_s = np.array(pcl_save[i,:,:].cpu().detach())
            np.save(file, pcl_s.T)
          

        return transl0, rot0, transl0, rot0#, transl0, rot0
