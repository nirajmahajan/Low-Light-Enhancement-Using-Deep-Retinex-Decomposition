#
# author: Sachin Mehta
# Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file contains the CNN models
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def rgb_to_grayscale(img, num_output_channels: int = 1):
    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    if num_output_channels == 3:
        return l_img.expand(img.shape)
    return l_img

class DecomNet(nn.Module):
    def __init__(self, num_layers = 5):
        super(DecomNet, self).__init__()
        layerlist = [
                    nn.Conv2d(4, 64, 9, stride=1, padding=4, bias=False),
                    nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                    nn.ReLU(True)
            ]
        for l in range(num_layers):
            layerlist.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False))
            layerlist.append(nn.BatchNorm2d(64, momentum=0.95, eps=1e-03))
            layerlist.append(nn.ReLU(True))
        layerlist.append(nn.Conv2d(64, 4, 3, stride=1, padding=1, bias=False))
        layerlist.append(nn.Sigmoid())
        self.model = nn.Sequential(*layerlist)

    def forward(self, input):
        input_max = input.max(1)[0].unsqueeze(1)
        input = torch.cat((input_max,input), 1)
        outp = self.model(input)
        R = outp[:,0:3,:,:]
        I = outp[:,3:4,:,:]
        return R, I

class RelightNet(nn.Module):
    def __init__(self):
        super(RelightNet, self).__init__()
        self.conv0 = nn.Conv2d(4, 64, 3, stride = 1, padding = 1)
        self.conv1 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride = 2, padding = 0),
                nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride = 2, padding = 0),
                nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                nn.ReLU(True)
            )
        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride = 2, padding = 0),
                nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                nn.ReLU(True)
            )

        self.deconv1 = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                nn.ReLU(True)
            )
        self.deconv2 = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                nn.ReLU(True)
            )
        self.deconv3 = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                nn.ReLU(True)
            )
        self.prefusion1 = nn.Upsample(scale_factor = 2)
        self.prefusion2 = nn.Upsample(scale_factor = 4)
        self.fusion = nn.Sequential(
                nn.Conv2d(64*3, 64, 1, stride = 1, padding = 0),
                nn.BatchNorm2d(64, momentum=0.95, eps=1e-03),
                nn.Conv2d(64, 1, 3, stride = 1, padding = 1)
            )

    def forward(self, iR, iI):
        input_im = torch.cat((iR,iI),1)
        conv0_outp = self.conv0(input_im)
        conv1_outp = self.conv1(F.pad(conv0_outp, (0,1,0,1))) # 112, 112
        conv2_outp = self.conv2(F.pad(conv1_outp, (0,1,0,1))) # 56, 56
        conv3_outp = self.conv3(F.pad(conv2_outp, (0,1,0,1))) # 28, 28

        deconv1_outp = self.deconv1(conv3_outp) + conv2_outp
        deconv2_outp = self.deconv2(deconv1_outp) + conv1_outp
        deconv3_outp = self.deconv3(deconv2_outp) + conv0_outp

        deconv1_resize = self.prefusion2(deconv1_outp)
        deconv2_resize = self.prefusion1(deconv2_outp)

        gathered_features = torch.cat((deconv1_resize, deconv2_resize, deconv3_outp), 1)
        outp = self.fusion(gathered_features)
        return outp

class LowLightEnhancer(nn.Module):
    def __init__(self, optim_choice = 'Adam', lr = 1e-3, device = torch.device("cpu")):
        super(LowLightEnhancer, self).__init__()
        self.optim_type = optim_choice 
        self.lr = lr
        self.device = device
        self.relight = RelightNet()
        self.decom = DecomNet()
        self.optimizer = optim.Adam(list(self.relight.parameters()) + list(self.decom.parameters()), lr=self.lr, betas=(.5, 0.999))

    def train(self, S_low, S_high):
        self.optimizer.zero_grad()
        S_low, S_high = S_low.to(self.device), S_high.to(self.device)
        self.decom.train()
        R_high, I_high = self.decom(S_high)
        R_low, I_low = self.decom(S_low)

        Ihat = self.relight(R_low, I_low)

        L_recon_low = (R_low*I_low - S_low).abs().mean()
        L_recon_high = (R_high*I_high - S_high).abs().mean()
        L_recon_dual_high = (R_high*I_low - S_low).abs().mean()
        L_recon_dual_low = (R_low*I_high - S_high).abs().mean()
        L_same_R = (R_low - R_high).abs().mean()

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(Ihat, R_low)

        self.L_Decom = L_recon_low + 5*L_recon_high
        self.L_Decom += 0.001 * (L_recon_dual_low + 5*L_recon_dual_high)
        self.L_Decom += 0.1 * (self.Ismooth_loss_low + self.Ismooth_loss_high)
        self.L_Decom += 0.1 * L_same_R
        
        self.relight.train()
        L_relight = (R_low.detach()*Ihat - S_high).abs().mean()
        self.L_Relight = L_relight + 3*(self.Ismooth_loss_delta)

        loss = self.L_Decom + self.L_Relight

        loss.backward()
        self.optimizer.step()

        return self.L_Decom.item(), self.L_Relight.item()
         

    def denoise(self, I, R):
        with torch.no_grad():
            t = 3
            UBs = [1, 0.08, 0.03]
            alphas = [1, 10, 100]
            sigmas = [10, 20, 40]
            Rt = R.copy()
            for ti in range(t):
                Mt = np.clip(I, 0, UBs[ti])/UBs[ti]
                Mt_hat = Mt ** alphas[ti]
                bm3d_R = R.copy()
                for i in range(R.shape[0]):
                    bm3d_R[i,:,:,:] = self.BM3D(R[i,:,:,:], sigmas[ti])
                Rt = (Rt*Mt_hat) + (bm3d_R[i,:,:,:]*(1-Mt_hat))
        return Rt

    def BM3D(self, R, sigma):
        output = cv2.fastNlMeansDenoising(np.uint8(R*255), None, sigma, 7, 21)
        return output/255.

    def smooth(self, I, R):
        gray = rgb_to_grayscale(R)
        term1 = self.gradient(I, "x")
        term2 = (-10*self.avg_gradient(gray, "x")).exp()
        term3 = self.gradient(I, "y")
        term4 = (-10*self.avg_gradient(gray, "y")).exp()
        return ((term1*term2)+(term3*term4)).mean()

    def gradient(self, input, direction):
        self.smooth_kernel_x = torch.tensor([[0, 0], [-1, 1.]]).view(1, 1, 2, 2).to(self.device)
        self.smooth_kernel_y = torch.permute(self.smooth_kernel_x, (0, 1, 3, 2))
        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return (F.conv2d(input, kernel, stride=1, padding=1)).abs()
    # def gradient(self, input, direction):
    #     sobely = [[1, 2., 1], [0, 0, 0], [-1, -2, -1]]
    #     sobelx = [[1, 0., -1], [2, 0, -2], [1, 0, -1]]
    #     self.smooth_kernel_x = torch.tensor(sobelx).view(1, 1, 3, 3).to(self.device)
    #     self.smooth_kernel_y = torch.tensor(sobely).view(1, 1, 3, 3).to(self.device)
    #     if direction == "x":
    #         kernel = self.smooth_kernel_x
    #     elif direction == "y":
    #         kernel = self.smooth_kernel_y
    #     return (F.conv2d(input, kernel, stride=1, padding=1)).abs()

    def avg_gradient(self, input, direction):
        outp1 = self.gradient(input, direction)
        return F.avg_pool2d(outp1, 3, stride = 1, padding = 1)

    def evaluate(self, S_low):
        with torch.no_grad():
            S_low = S_low.to(self.device)
            self.decom.eval()
            R_low, I_low = self.decom(S_low)
            self.relight.eval()
            Ihat = self.relight(R_low, I_low).permute(0,2,3,1).cpu().numpy()
            # Rhat = self.denoise(I_low.permute(0,2,3,1).cpu().numpy(), R_low.permute(0,2,3,1).cpu().numpy())
            Rhat = R_low.permute(0,2,3,1).cpu().numpy()
            ans = Ihat*Rhat
            return ans, Ihat, Rhat