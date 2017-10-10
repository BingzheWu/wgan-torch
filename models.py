import torch
import torch.nn as nn
import torch.nn.parallel
from ops import conv_block, upsample
class netD(nn.Module):
	def __init__(self, i_size, nz, nc, ndf, ngpu, n_extra_layers = 0):
		super(netD, self).__init__()
		self.ngpu = ngpu
		assert i_size % 16 == 0
		out = nn.Sequential()
		out.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
			nn.Conv2d(nc, ndf, 4, 2, 1, bias = False))
		out.add_module('initial.relu.{0}'.format(ndf),
			nn.LeakyReLU(0.2, inplace = True))
		csize, cndf = i_size / 2, ndf

		while csize > 4:
			in_feat = cndf
			out_feat = cndf * 2
			name = ''
			out.add_module('pyramid.{0}-{1}'.format(in_feat, out_feat),
				conv_block(in_feat, out_feat, 4, 2, 1, 'block'))
			cndf = cndf * 2
			csize = csize / 2
		out.add_module('final.{0}-{1}.conv'.format(cndf, 1), nn.Conv2d(cndf, 1, 4, 1, 0, bias = False))
		self.out = out
	def forward(self, x):
		if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.out, x, range(self.ngpu))
		else:
			output = self.out(x)
		out = output.mean(0)
		return out.view(1)

class netG(nn.Module):
	def __init__(self, i_size, nz, nc, ngf, ngpu, n_extra_layers = 0):
		super(netG, self).__init__()
		self.ngpu = ngpu
		assert i_size % 16 == 0
		cngf, tsize = ngf//2, 4
		while tsize != i_size:
			cngf = cngf * 2
			tsize = tsize*2 
		out = nn.Sequential()
		out.add_module('initial.{0}-{1}'.format(nz, cngf), upsample(nz, cngf, 4, 1, 0, 'upsample'))
		csize, cndf = 4, cngf
		while csize < i_size//2:
			out.add_module('pyramid.{0}-{1}'.format(cngf, cngf//2), upsample(cngf, cngf//2, 4,2,1, 'upsample'))
			cngf = cngf // 2
			csize = csize * 2
		out.add_module('final.{0}.{1}.convt'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1))
		out.add_module('final.{0}.tanh'.format(nc), nn.Tanh())
		self.out = out
	def forward(self, x):
		if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.out, x, range(self.ngpu))
		else:
			output = self.out(x)
		#print(output.size())
		return output
	