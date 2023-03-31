import numpy as np
import skimage as sk
import torch.nn.functional as F
import torch
from torch import nn
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
        ker = transforms.ToTensor()(ker.astype('float32')).unsqueeze_(0)
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

class CsiszarDiv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # Output is variable, target is GT
        return torch.sum(target*torch.log(target/output) - target + output)

def to_numpy(im):
    return im.squeeze().detach().cpu().numpy()

def cdf(im, bits=16):
    hist, _ = np.histogram(im.flatten(), bins=2**bits, range=(0,1))
    cdf = np.cumsum(hist) / np.prod(im.shape)
    return cdf

def match_hist(im, ref, bits=16):
    # compute cdf
    cdf_ref = cdf(ref, bits)
    cdf_im = cdf(im, bits)
    # keep only leftmost quantiles for interpolation
    is_left = np.hstack([True, cdf_ref[:-1] - cdf_ref[1:] < 0])
    left_cdf_ref = cdf_ref[is_left]
    left_values = (np.nonzero(is_left)[0] / (2**bits-1)).astype(im.dtype)
    # 2**bits array representing the matching function
    matches = np.interp(cdf_im, left_cdf_ref, left_values)
    im_matched = matches[(im*(2**bits-1)).astype(f'uint{str(bits)}')]
    return im_matched