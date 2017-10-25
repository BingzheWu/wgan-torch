import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from options import opt
import json
import PIL.Image as Image
import os
class dataloader(data.Dataset):
	def __init__(self, root_dir, list_file, train, transform):
		self.root_dir = root_dir
		self.is_train = train
		self.transform = transform
		with open(list_file, 'r') as f:
			self.labels = json.load(f)
		self.num_samples = len(self.labels)
	def __getitem__(self, idx):
		label = self.labels[idx]
		image = Image.open(os.path.join(self.root_dir, label['image_id'])).convert('RGB')
		label = label['label_id']
		image = self.transform(image)
		target = int(label)
		return image, target
	def __len__(self):
		return self.num_samples
def scene_dataset(root_dir, list_file, is_train):
	transform = transforms.Compose([transforms.Scale(opt.imageSize), 
		transforms.CenterCrop(opt.imageSize),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	out = dataloader(root_dir, list_file, is_train, transform)
	out = torch.utils.data.DataLoader(out, batch_size = opt.batchSize, shuffle = True, 
		num_workers = int(opt.workers), drop_last = True)
	return out
