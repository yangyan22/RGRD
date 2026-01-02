import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def gradient(img):
    height = img.size(2)
    width = img.size(3)

    gradient_h2 = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w2 = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h2 = F.pad(gradient_h2, [0, 0, 1, 1], 'replicate')
    gradient_w2 = F.pad(gradient_w2, [1, 1, 0, 0], 'replicate')

    gradient_h = (img[:,:,1:,:]-img[:,:,:height-1,:]).abs()
    gradient_w = (img[:, :, :, 1:] - img[:, :, :, :width-1]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 0, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 0, 0, 0], 'replicate')    

    return 1/2*(gradient_h2+gradient_h), 1/2*(gradient_w2+gradient_w)


def rec_loss(img, L, R):
    return torch.nn.MSELoss()(img, L*R)

def L_loss(img, L):
    g_kernel_size = 5
    pad = 2
    sigma = 10
    kx = cv2.getGaussianKernel(g_kernel_size,sigma)
    ky = cv2.getGaussianKernel(g_kernel_size,sigma)
    gaussian_kernel = np.multiply(kx,np.transpose(ky))
    gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).cuda()

    gray_tensor = 0.299*img[0,0,:,:] + 0.587*img[0,1,:,:] + 0.114*img[0,2,:,:]
    # gray_tensor = img[0,0,:,:] + img[0,1,:,:] + img[0,2,:,:]
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_illu_h, gradient_illu_w = gradient(L)

    weight_h = 1/(F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=pad)+1e-6)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=pad)+1e-6)
     
    loss_h = gradient_illu_h*gradient_illu_h * weight_h
    loss_w = gradient_illu_w*gradient_illu_w * weight_w

    max_rgb, _ = torch.max(img, 1) 
    max_rgb = max_rgb.unsqueeze(1)

    loss1 = torch.nn.MSELoss()(L, max_rgb) 
    loss2 = loss_h.mean() + loss_w.mean() 
    return loss1 + loss2


def R_loss(img, L, R):
    loss = torch.nn.MSELoss(reduction='none')(R, img/L) * L
    return loss.mean()


def perceptual_loss(input_features, target_features):

    loss = 0.0
    for (inp_feat, tar_feat) in zip(input_features, target_features):
        loss += torch.mean(torch.abs(inp_feat - tar_feat))
    return loss


def kl_loss(input_features, target_features, dim=-1, normalize=True):
    input_flat = input_features.reshape(input_features.shape[0], -1)
    target_flat = target_features.reshape(target_features.shape[0], -1)

    if normalize:
        input_flat = (input_flat - input_flat.mean(dim=dim, keepdim=True)) / (input_flat.std(dim=dim, keepdim=True) + 1e-5)
        target_flat = (target_flat - target_flat.mean(dim=dim, keepdim=True)) / (target_flat.std(dim=dim, keepdim=True) + 1e-5)
    input_log_probs = F.log_softmax(input_flat, dim=dim)
    target_probs = F.softmax(target_flat, dim=dim)
    loss = torch.nn.KLDivLoss(reduction='batchmean')

    return loss(input_log_probs, target_probs)