import json
from collections import OrderedDict
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

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
						num_workers=workers, pin_memory=False)

	return train_loader
 

def get_stl10_dataloader(dataroot, batch_size, workers, image_size=48):
	train_set = dset.STL10(root=dataroot, download=True,
					transform=transforms.Compose([
					   transforms.Resize(image_size),
					   transforms.ToTensor(),
					   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					   ]))

	train_loader = torch.utils.data.DataLoader(
						train_set,
						batch_size=batch_size, shuffle=True,
						num_workers=workers, pin_memory=False)

	return train_loader






