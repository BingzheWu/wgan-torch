import torch
import torch.nn as nn
import torch.optim as optim
from options import opt
from models import netD, netG, acgan
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.parallel
from datasets import scene_dataset
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm')!=-1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
def train():
	## set parameter
	ngpu = int(opt.ngpu)
	nz = int(opt.nz)
	ngf = int(opt.ngf)
	ndf = int(opt.ndf)
	nc = int(opt.nc)

	x = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
	noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
	fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0,1)
	one = torch.FloatTensor([1])
	mone = one*-1
	## dataset
	dataloader = scene_dataset(opt.dataroot, opt.list_file, is_train = True)
	## net build
	net_d = netD(opt.imageSize, nz, nc, ndf, ngpu)
	net_d.apply(weights_init)
	net_g = netG(opt.imageSize, nz, nc, ngf, ngpu)
	net_g.apply(weights_init)
	if opt.net_d != '':
		net_d.load_state_dict(torch.load(opt.net_d))
	if opt.net_g != '':
		net_g.load_state_dict(torch.load(opt.net_g))

	## train setup
	if opt.cuda:
		net_d.cuda()
		net_g.cuda()
		x = x.cuda()
		one, mone = one.cuda(), mone.cuda()
		noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
	## optimizer

	optimizer_d = optim.RMSprop(net_d.parameters(), lr = opt.lrD)
	optimizer_g = optim.RMSprop(net_g.parameters(), lr = opt.lrG)
	gen_iterations = 0
	for epoch in xrange(opt.niter):
		data_iter = iter(dataloader)
		i = 0
		while i < len(dataloader):
			for p in net_d.parameters():
				p.requir_grad = True
			if gen_iterations < 25 or gen_iterations % 500 == 0:
				Diters = 100
			else:
				Diters = opt.Diters
			j = 0
			while j < Diters and i < len(dataloader):
				j += 1
				for p in net_d.parameters():
					p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
				data = data_iter.next()
				i += 1
				real_data, _ = data
				net_d.zero_grad()
				batch_size = real_data.size(0)

				if opt.cuda:
					real_data = real_data.cuda()
				x.resize_as_(real_data).copy_(real_data)
				x_v = Variable(x)
				errD_real = net_d(x_v)
				errD_real.backward(one)
				noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
				noise_v = Variable(noise, volatile = True)
				fake_data  = Variable(net_g(noise_v).data)
				errD_fake = net_d(fake_data)
				errD_fake.backward(mone)
				errD = errD_real - errD_fake

				optimizer_d.step()

			for p in net_d.parameters():
				p.requir_grad = False
			net_g.zero_grad()
			noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
			noise_v = Variable(noise)
			fake = net_g(noise_v)
			errG = net_d(fake)
			errG.backward(one)
			optimizer_g.step()
			gen_iterations += 1
			print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'%
				(epoch, opt.niter, i, len(dataloader), gen_iterations,
				errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
			if gen_iterations % 500 == 0:
				real_data = real_data.mul(0.5).add(0.5)
				vutils.save_image(real_data, '{0}/real_samples_{1}.png'.format(opt.experiment, gen_iterations))
				fake = net_g(Variable(fixed_noise, volatile = True))
				fake.data = fake.data.mul(0.5).add(0.5)
				vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
		torch.save(net_g.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
		torch.save(net_d.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

def main():
	model_acgan = acgan(opt)
	model_acgan.train()

if __name__ == '__main__':
	main()