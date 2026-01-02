import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import numpy as np
import glob
import cv2
import lpips
from PIL import Image
from scipy import stats


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    sem = stats.sem(a)  
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, h


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

def metrics(im_dir, label_dir, flag=None):
    all_psnr = []
    all_ssim = []
    all_lpips = []
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()


    for item in sorted(glob.glob(im_dir)):

        n += 1

        if flag == 'LOLv1':
            im1 = Image.open(item).convert('RGB') 
            name = item.split('/')[-1]
            im2 = Image.open(label_dir + name).convert('RGB')

        if flag == 'SICEp2':
            im1 = Image.open(item).convert('RGB') 
            name = item.split('/')[-1]
            name = name.split('_')[0] + '.JPG'
            im2 = Image.open(label_dir + name).convert('RGB')

        if flag == 'LOLv2r':
            im1 = Image.open(item).convert('RGB') 
            name = item.split('/')[-1]
            name = name.replace('low', '')
            im2 = Image.open(label_dir + name).convert('RGB')  

        if flag == 'LOLv2s':
            im1 = Image.open(item).convert('RGB') 
            name = item.split('/')[-1]
            im2 = Image.open(label_dir + name).convert('RGB')                 

        (h, w) = im2.size
        im1 = im1.resize((h, w)) 

        im1 = np.array(im1)
        im2 = np.array(im2)

        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)

        ex_p0 = lpips.im2tensor(cv2.resize(lpips.load_image(item), (h, w)))
        ex_ref = lpips.im2tensor(lpips.load_image(label_dir + name))
        ex_p0 = ex_p0.cuda()
        ex_ref = ex_ref.cuda()
        score_lpips = loss_fn.forward(ex_ref, ex_p0).item()


        all_psnr.append(score_psnr)
        all_ssim.append(score_ssim)
        all_lpips.append(score_lpips)
        # print(f"{item}: PSNR={score_psnr:.4f}, SSIM={score_ssim:.4f}, LPIPS={score_lpips:.4f}")
        
    
    # ---------- 95% CI ----------
    print(f"Total images: {n}")
    avg_psnr, ci_psnr = mean_confidence_interval(all_psnr, confidence=0.95)
    avg_ssim, ci_ssim = mean_confidence_interval(all_ssim, confidence=0.95) 
    avg_lpips, ci_lpips = mean_confidence_interval(all_lpips, confidence=0.95)


    print(f"PSNR: {avg_psnr:.2f} ± {ci_psnr:.2f}")
    print(f"SSIM: {avg_ssim:.3f} ± {ci_ssim:.2f}")
    print(f"LPIPS: {avg_lpips:.3f} ± {ci_lpips:.2f}")


if __name__ == '__main__':
    im_dir = 'results/LOL/I/*'
    label_dir = 'test-dataset/LOL/reference/'
    metrics(im_dir, label_dir, flag='LOLv1')

    im_dir = 'results/SICEp2/I/*'
    label_dir = 'test-dataset/SICEp2/label/'
    metrics(im_dir, label_dir, flag='SICEp2')

    im_dir = 'results/LOLv2r/I/*'
    label_dir = 'test-dataset/LOLv2r/Normal/'
    metrics(im_dir, label_dir, flag='LOLv2r')

    im_dir = 'results/LOLv2s/I/*'
    label_dir = 'test-dataset/LOLv2s/Normal/'
    metrics(im_dir, label_dir, flag='LOLv2s')




