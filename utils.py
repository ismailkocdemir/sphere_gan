import json
from collections import OrderedDict
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GeometricBlock(nn.Module):
    def __init__(self, dim, pool=True):
        super().__init__()
        self.pool = pool
        self.ln = nn.Linear(dim, 1024, bias=False)

    def forward(self, u):
        '''
            Inverse Steregraphic Projection.
        '''

        # Here there should be another LReLU according to the paper. But we already applied that in the last Conv. layer before Geometric Block?
        
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
        nn.init.xavier_normal_(m.weight.data)
    
    elif isinstance(m, nn.Linear):
    	nn.init.normal_(m.weight.data)

    elif isinstance(m, nn.BatchNorm2d):
	    nn.init.constant_(m.weight, 1)
	    nn.init.constant_(m.bias, 0)

def total_moment_distance(output, target, moments, device):
	#print(target.size(), output.size())
	
	distance = torch.zeros(output.size()).to(device)
	for i in range(1, moments+1):
		distance += get_distance_to_north_pole(output, i, device)

	return torch.mean(target*distance)

def get_distance_to_north_pole(input, moment, device):
	north_pole = torch.zeros((1, input.size()[-1])).to(device)
	north_pole[:, -1] = 1.0

	return torch.acos(torch.matmul(input ,torch.transpose(north_pole, dim0=0, dim1=1).to(device) )) ** moment

def get_cifar10_dataloader(dataroot, batch_size, workers, image_size=32):
	train_set = dset.CIFAR10(root=dataroot, download=True, train=True,
						   transform=transforms.Compose([
							   transforms.Resize(image_size),
							   transforms.ToTensor(),
							   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						   ]))
	train_loader = torch.utils.data.DataLoader(
						train_set,
						batch_size=batch_size, shuffle=True,
						num_workers=workers, pin_memory=False, drop_last=True)

	return train_loader
 

def get_stl10_dataloader(dataroot, batch_size, workers, image_size=48):
	train_set = dset.STL10(root=dataroot, download=True,
					transform=transforms.Compose([
					   transforms.Resize(image_size),
					   transforms.ToTensor(),
					   transforms.Normalize((0.44671097, 0.4398105 , 0.4066468), (0.2603405 , 0.25657743, 0.27126738)),
					   ]))

	train_loader = torch.utils.data.DataLoader(
						train_set,
						batch_size=batch_size, shuffle=True,
						num_workers=workers, pin_memory=False, drop_last=True)

	return train_loader



class ConfigReader():
	""" Custom Config Parser for our CENG796 project """

	def __init__(self, config_file):
		self.config_dict = self.parse_config(config_file)

	def parse_config(self, config_file):
		"""  Parse configuration file and return the dictionary """
		with open(config_file, 'rt') as handle:
			config = json.load(handle, object_hook=OrderedDict)

		assert "GENERAL" in config, "Place the train parameters under GENERAL section in the configuration file"
		assert "DISCRIMINATOR" in config, "Place the discriminator parameters under DISCRIMINATOR section in the configuration file"
		assert "GENERATOR" in config, "Place the generator parameters under GENERATOR section in the configuration file"

		return config

	def get_train_params(self,):
		"""
		_dict = dict(self.config_dict["GENERAL"])
		for item in _dict:
			if item != "dataset":
				if item == "lr" or item == "beta1":
					_dict[item] = float(_dict[item])
				else:
					_dict[item] = int(_dict[item])
		"""
		return dict(self.config_dict["GENERAL"])

	def get_generator_params(self,):
		return {**self.config_dict["GENERATOR"],\
					"latent_size": self.config_dict["GENERAL"]["latent_size"],
					"image_channels" : self.config_dict["GENERAL"]["image_channels"]}
		'''
		return {"num_channels": int(self.config_dict["GENERATOR"]["num_channels"]),
					"kernel_size" : int(self.config_dict["GENERATOR"]["kernel_size"]),
					"latent_size": int(self.config_dict["GENERAL"]["latent_size"]),
					"image_channels" : int(self.config_dict["GENERAL"]["image_channels"])}
		'''

	def get_discriminator_params(self,):
		return {**self.config_dict["DISCRIMINATOR"], \
					"image_channels" : self.config_dict["GENERAL"]["image_channels"]  }

	def get_dataloader_params(self,):
		return dict(self.config_dict["DATALOADER"])
		"""
		_dict = dict(self.config_dict["DATALOADER"])
		
		for item in _dict:
			if item != "dataroot":
				_dict[item] = int(_dict[item])
		return _dict
		"""

