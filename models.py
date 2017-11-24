import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.utils as vutils
import torch.optim as optim
from ops import conv_block, upsample, onehot
from datasets import scene_dataset
import time
from torch.autograd import Variable
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
		return output
class acgan_g(netG):
	def __init__(self, num_classes, i_size, nz, nc, ngf, ngpu, n_extra_layers = 0):
		super(acgan_g, self).__init__(i_size, nz+num_classes, nc, ngf, ngpu, n_extra_layers)
		self.i_dim = nz + num_classes
		self.fc = nn.Sequential(nn.Linear(self.i_dim, 128*1*1),
		nn.BatchNorm1d(128),
		nn.ReLU(),
		)
	def forward(self, input, label):
		x = torch.cat([input, label], 1)		
		if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			out = nn.parallel.data_parallel(self.out, x, raneg(self.ngpu))
		else:
			out = self.out(x)
		return out
class acgan_d(nn.Module):
	def __init__(self, num_classes, i_size, nz, nc, ndf, ngpu):
		super(acgan_d, self).__init__()
		self.ngpu = ngpu
		self.num_classes = num_classes
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
		out.add_module('GlobalAverage', nn.AvgPool2d(4,1))
		self.out = out
		
		self.dc = nn.Sequential(
			nn.Linear(out_feat, 1),
			nn.Sigmoid(),
		)
		self.cl = nn.Sequential(
			nn.Linear(out_feat, self.num_classes),
		)
	def forward(self, x):
		x  = self.out(x)
		x = x.view(-1,512)
		d = self.dc(x)
		c = self.cl(x)
		return d, c		
		
class acgan(object):
	def __init__(self, opt):
		self.opt = opt
		self.sample_num = 64
		self.num_classes = opt.num_classes
		self.ngpu = int(opt.ngpu)
		self.nz = int(opt.nz)
		self.ngf = int(opt.ngf)
		self.ndf = int(opt.ndf)
		self.nc = int(opt.nc)
		self.G = acgan_g(opt.num_classes, opt.imageSize, self.nz, self.nc, self.ngf, self.ngpu)
		self.D = acgan_d(opt.num_classes, opt.imageSize, self.nz, self.nc, self.ndf, self.ngpu)
		self.g_optimizer = optim.Adam(self.G.parameters(), lr = self.opt.lrG)
		self.d_optimizer = optim.Adam(self.D.parameters(), lr = self.opt.lrD)
		if self.opt.cuda:
			self.G.cuda()
			self.D.cuda()
			self.BCE_loss = nn.BCELoss().cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda()

	def train(self):
		x = torch.FloatTensor(self.opt.batchSize, 3, self.opt.imageSize, self.opt.imageSize)
		data_iter = scene_dataset(self.opt.dataroot, self.opt.list_file, is_train = True)
		#fixed noise and 
		self.sample_z = torch.zeros((self.sample_num, self.nz,1, 1))
		if self.opt.cuda:
			self.y_real = Variable(torch.ones(self.opt.batchSize, 1).cuda())
			self.y_fake = Variable(torch.zeros(self.opt.batchSize, 1).cuda())
		
		for i in range(self.sample_num):
			self.sample_z[i] = torch.rand(1, self.nz)
		self.sample_z.view(self.sample_num, self.nz, 1, 1)
		temp = torch.zeros((self.num_classes, 1))
		temp = torch.randperm(self.opt.num_classes)
		temp = temp[0:self.sample_num]
		#self.sample_y = torch.zeros((self.sample_num, self.num_classes))
		self.sample_y = onehot(temp, self.opt.num_classes)
		self.sample_y.view(-1, self.opt.num_classes, 1, 1)
		if self.opt.cuda:
			self.sample_z = Variable(self.sample_z.cuda(), volatile = True)
			self.sample_y = Variable(self.sample_y.cuda(), volatile = True)
		x = torch.FloatTensor(self.opt.batchSize, 3, self.opt.imageSize, self.opt.imageSize)
		noise = torch.FloatTensor(self.opt.batchSize, self.nz, 1, 1)
		fixed_noise = torch.FloatTensor(self.opt.batchSize, self.nz, 1, 1).normal_(0,1)
		one = torch.FloatTensor([1]).cuda()
		mone = one*-1
		for epoch in range(self.opt.niter):
			self.G.train()
			epoch_start_time = time.time()
			i =  0
			for iter_, data in enumerate(data_iter):
				for p in self.D.parameters():
					p.data.clamp_(self.opt.clamp_lower, self.opt.clamp_upper)
				#data = data_iter.next()
				real_data, label = data
				self.D.zero_grad()
				batch_size = real_data.size(0)
				noise.resize_(self.opt.batchSize, self.opt.nz, 1, 1)
				if self.opt.cuda:
					x = x.cuda()
					real_data = real_data.cuda()
					label = onehot(label, self.opt.num_classes)
					label = label.view(-1, self.opt.num_classes, 1, 1)
					label = Variable(label.cuda())
					noise_v = Variable(noise.cuda())
				x.resize_as_(real_data).copy_(real_data)
				x_v = Variable(x)
				d_real, c_real = self.D(x_v)
				#d_real_loss = d_real
				#d_real.backward(one, retain_variables = True)
				d_real_loss = self.BCE_loss(d_real, self.y_real)
				c_real_loss = self.CE_loss(c_real, torch.max(label,1)[1].view(-1))
				#c_real_loss.backward(one, retain_variables = True)
				fake = self.G(noise_v, label)
				d_fake, c_fake = self.D(fake)
				#d_fake_loss = d_fake
				#d_fake.backward(mone, retain_variables = True)
				d_fake_loss = self.BCE_loss(d_fake, self.y_fake)
				c_fake_loss = self.CE_loss(c_fake, torch.max(label, 1)[1].view(-1))
				#c_fake_loss.backward(one, retain_variables = True)
				d_loss = d_real_loss + c_real_loss + d_fake_loss + c_fake_loss
				d_loss.backward()
				self.d_optimizer.step()
				self.G.zero_grad()
				fake = self.G(noise_v, label)
				d_fake, c_fake = self.D(fake)
				#g_loss = d_fake
				#g_loss.backward(retain_variables = True)
				g_loss = self.BCE_loss(d_fake, self.y_real)
				c_fake_loss = self.CE_loss(c_fake, torch.max(label,1)[1].view(-1))
				#c_fake_loss.backward(retain_variables = True)
				g_loss += c_fake_loss
				g_loss.backward()
				self.g_optimizer.step()
				if ((iter_+1)%100) == 0:
					print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f"%
					((epoch +1), (iter_+1), len(data_iter), d_loss.cpu().data[0], g_loss.cpu().data[0]
					))
				self.visualize_results(epoch)
	def visualize_results(self, epoch, fix = True):
		self.G.eval()
		if not os.path.exists(os.path.join(self.opt.experiment, 'vis_results')):
			os.makedirs(os.path.join(self.opt.experiment, 'vis_results'))
		if fix:
			samples = self.G(self.sample_z, self.sample_y)
		vutils.save_image(samples.data, os.path.join(self.opt.experiment,'fake_samples_{0}.png'.format(epoch)))


				

