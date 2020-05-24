import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


class GeometricBlock(nn.Module):
	def __init__(self, dim, pool=True):
		super().__init__()
		self.pool = pool
		self.ln = nn.Linear(dim, 1024, bias=True)
		self.lr = nn.LeakyReLU(0.2)

	def forward(self, u):
		'''
			Inverse Steregraphic Projection.
		'''

		u = self.lr(u)

		# Global Average Pooling is not implemented in pytorch. Instead use adaptive avg. pooling and reduce spatial dim's to 1.
		if self.pool:
			u = F.adaptive_avg_pool2d(u, (1, 1))
	
		# Flatten
		u = u.view(u.size()[0], -1)
		
		# Dense Layer
		u = self.ln(u)

		# Inverse Projection
		u_hat = 2*u / (torch.pow(torch.norm(u, dim=1, keepdim=True), 2) + 1)
		v = (torch.pow(torch.norm(u, dim=1, keepdim=True), 2) - 1 ) / (torch.pow(torch.norm(u, dim=1, keepdim=True), 2) + 1)
		out = torch.cat((u_hat, v), dim=1)
		return out 

class View(nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape

	def forward(self, input):
		'''
		Reshapes the input according to the shape saved in the view data structure.
		'''
		batch_size = input.size(0)
		shape = (batch_size, *self.shape)
		out = input.view(shape)
		return out


class Flatten(nn.Module):
	def __init__(self,):
		super().__init__()

	def forward(self, x):
		x = x.view(x.size()[0], -1)
		return x

def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		nn.init.xavier_normal_(m.weight)
	
	elif isinstance(m, nn.Linear):
		nn.init.normal_(m.weight)

	elif isinstance(m, nn.BatchNorm2d):
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)

def total_moment_distance(output, target, moments, device):
	#print(target.size(), output.size())
	
	distance = torch.zeros(output.size()).to(device)
	for i in range(1, moments+1):
		distance += get_distance_to_north_pole(output, i, device) # / normalization(r=i)

	return torch.mean(target*distance)

def get_distance_to_north_pole(input, moment, device):
	north_pole = torch.zeros((1, input.size()[-1])).to(device)
	north_pole[:, -1] = 1.0

	dot_prod = torch.matmul(input ,torch.transpose(north_pole, dim0=0, dim1=1).to(device) )

	return torch.acos(dot_prod) ** moment

def normalization(r, mode="normalization", decay_ratio=3):
		if mode == 'normalization':
			return (1/decay_ratio)*np.pi**r
		elif mode == 'half':
			return (np.pi)**r
