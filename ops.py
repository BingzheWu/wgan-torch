import torch
import torch.nn as nn


def conv_block(in_c, out_c, k_size, strides, padding, name, 
	alpha = 0.2, bias = False, batch_norm = True):
	out = nn.Sequential()
	out.add_module(name+'_conv', nn.Conv2d(in_c, out_c, k_size, strides, padding, bias = bias))
	if batch_norm:
		out.add_module(name+'_norm', nn.BatchNorm2d(out_c))
	out.add_module(name+'_activation', nn.LeakyReLU(alpha, inplace = True))
	return out
def upsample(in_c, out_c, k_size, strides, padding, name, alpha = 0.2, 
	bias = False, batch_norm = False):
	out = nn.Sequential()
	out.add_module(name+'.conv', nn.ConvTranspose2d(in_c, out_c, k_size, strides, padding, bias = bias))
	if batch_norm:
		out.add_module(name+'.norm', nn.BatchNorm2d(out_c))
	out.add_module(name+'.activation', nn.LeakyReLU(alpha, inplace = True))
	return out
	