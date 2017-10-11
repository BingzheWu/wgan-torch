import os
import torch
from torch.autograd import Variable
from models import netG
from options import opt
import torchvision.utils as vutils
"""
using trained generator to generate fake images
"""

def gen_images(gen_image_num, save_path):
    net_g = netG(opt.imageSize, int(opt.nz), int(opt.nc), int(opt.ndf), int(opt.ngpu))
    net_g.load_state_dict(torch.load(opt.net_g))

    for i in range(gen_image_num):
        noise = torch.FloatTensor(1, int(opt.nz), 1, 1).normal_(0,1)
        if opt.cuda:
            net_g.cuda()
            noise = noise.cuda()
        
        noise_v = Variable(noise, volatile = True)
        fake = net_g(noise_v)
        fake.data = fake.data.mul(0.5).add(0.5)
        vutils.save_image(fake.data, os.path.join(save_path, 'fake_sample{0}.png'.format(i)))


if __name__ == '__main__':
    gen_image_num = 10000
    save_path = 'experiments/gen_scene_images/'
    gen_images(gen_image_num, save_path)


