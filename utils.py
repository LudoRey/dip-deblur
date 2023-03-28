import numpy as np
import skimage as sk
import torch.nn.functional as F
import torch
from torchvision import transforms

def read_img(path, as_tensor=False):
    '''Returns a H x W x C ndarray by default, or 1 x C x H x W if as_tensor is True'''
    im = sk.img_as_float32(sk.io.imread(path))
    if as_tensor:
        im = transforms.ToTensor()(im).unsqueeze_(0)
    return im

def gaussian_kernel(sigma=1, truncate=3.0, as_tensor=False):
    '''Returns a H x W ndarray by default, or 1 x 1 x H x W if as_tensor is True.
    H = W = 2r+1 with r = round(truncate*sigma)'''
    # calcul du rayon du noyau
    radius = round(truncate * sigma)
    # creation de la grille de coordonnées x, y
    x, y = np.mgrid[-radius:radius+1,
                    -radius:radius+1]
    # calcul de la valeur de la fonction gaussien en chaque point
    ker = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    # normaliser
    ker = ker / np.sum(ker)
    if as_tensor:
        # retourne un tenseur 1x1xHxW
        ker = torch.tensor(ker, dtype=torch.float32).reshape(1,1,ker.shape[0],ker.shape[1])
    return ker

def blur(im, ker, padding_mode='reflect'):
    '''im and ker are tensors'''
    pad = (ker.shape[-1] // 2,)*2 + (ker.shape[-2] // 2,)*2
    im_pad = F.pad(im, pad, mode=padding_mode)
    im_blur = F.conv2d(im_pad, ker, padding='valid')
    return im_blur

def poisson_noise(im, peak):
    '''im is a tensor, peak is a scalar'''
    noisy_im = torch.poisson(im * peak) / peak
    return noisy_im

def display(im):
    return im.squeeze().detach().numpy()