import os, argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn as nn

import utils
from EDSR.edsr import EDSR
from modules import DSN
from adaptive_gridsampler.gridsampler import Downsampler
from skimage.color import rgb2ycbcr


parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
parser.add_argument('--model_dir', type=str, default='./models', help='path to the pre-trained model')
parser.add_argument('--img_dir', type=str, help='path to the HR images to be downscaled')
parser.add_argument('--scale', type=int, help='downscale factor')
parser.add_argument('--output_dir', type=str, help='path to store results')
args = parser.parse_args()


SCALE = args.scale
KSIZE = 3 * SCALE + 1
OFFSET_UNIT = SCALE


kernel_generation_net = DSN(k_size=KSIZE, scale=SCALE).cuda()
upscale_net = EDSR(32, 256, scale=SCALE).cuda()


kernel_generation_net = nn.DataParallel(kernel_generation_net, [0])
upscale_net = nn.DataParallel(upscale_net, [0])


kernel_generation_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'kgn.pth')))
upscale_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'usn.pth')))
torch.set_grad_enabled(False)


def validation(img, name, save_imgs=False, save_dir=None):
    kernel_generation_net.eval()
    upscale_net.eval()

    kernels, offsets_h, offsets_v = kernel_generation_net(img)
   
    reconstructed_img = upscale_net(img)

    img = img * 255
    img = img.data.cpu().numpy().transpose(0, 2, 3, 1)
    img = np.uint8(img)

    reconstructed_img = torch.clamp(reconstructed_img, 0, 1) * 255
    reconstructed_img = reconstructed_img.data.cpu().numpy().transpose(0, 2, 3, 1)
    reconstructed_img = np.uint8(reconstructed_img)

    orig_img = img[0, ...].squeeze()
    recon_img = reconstructed_img[0, ...].squeeze()
    

    if save_imgs and save_dir:
        img = Image.fromarray(orig_img)
        img.save(os.path.join(save_dir, name + '_orig2.png'))


        img = Image.fromarray(recon_img)
        img.save(os.path.join(save_dir, name + '_recon.png'))

if __name__ == '__main__':
    img_list = glob(os.path.join(args.img_dir, '**', '*.png'), recursive=True)
    assert len(img_list) > 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_file in tqdm(img_list):
        name = os.path.basename(img_file)
        name = os.path.splitext(name)[0]
        img = utils.load_img(img_file)
        validation(img, name, save_imgs=True, save_dir=args.output_dir)
        