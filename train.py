import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from model import Discriminator, Generator
from utils import *

def train(config_file):

	# Parse the configuration file
	configparser = ConfigReader(config_file)
	train_params = configparser.get_train_params()
	num_gpu = train_params["num_gpu"]
	manual_seed = train_params["manual_seed"]
	
	latent_size = train_params["latent_size"]
	lr = train_params["lr"]
	beta1 = train_params["beta1"]
	num_epochs = train_params["num_epochs"]
	dataset = train_params["dataset"].lower()
	
	# Set the seed manually for reproducability
	print("Random Seed: ", manual_seed)
	random.seed(manual_seed)
	torch.manual_seed(manual_seed)

	# Look for available GPU
	device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")	

	# Create the generator
	gen_params = configparser.get_generator_params()
	netG = Generator(**gen_params).to(device)
	if (device.type == 'cuda') and (num_gpu > 1):
		netG = nn.DataParallel(netG, list(range(num_gpu)))

	# Create the Discriminator
	disc_params = configparser.get_discriminator_params()
	netD = Discriminator(**disc_params).to(device)
	if (device.type == 'cuda') and (num_gpu > 1):
		netD = nn.DataParallel(netD, list(range(num_gpu)))

	# Create the Dataloader
	loader_params = configparser.get_dataloader_params()
	if dataset == "cifar10":
		dataloader = get_cifar10_dataloader(**loader_params)
	elif dataset == "stl10":
		dataloader = get_stl10_dataloader(**loader_params)
	else:
		print(dataset)
		raise ValueError("Invalid dataset: {}".format(dataset))

	# Plot some training images
	real_batch = next(iter(dataloader))
	plt.figure(figsize=(8,8))
	plt.axis("off")
	plt.title("Training Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

	print(netD)
	print(netG)


	# Initialize BCELoss function
	criterion = nn.BCELoss()

	# Create batch of latent vectors that we will use to visualize
	#  the progression of the generator
	fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)

	# Establish convention for real and fake labels during training
	real_label = 1
	fake_label = 0

	# Setup Adam optimizers for both G and D
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

	# TRAINING
	# Lists to keep track of progress
	img_list = []
	G_losses = []
	D_losses = []
	iters = 0

	# For each epoch
	for epoch in range(num_epochs):
		# For each batch in the dataloader
		for i, data in enumerate(dataloader, 0):

			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################
			## Train with all-real batch
			netD.zero_grad()
			# Format batch
			real_cpu = data[0].to(device)
			b_size = real_cpu.size(0)
			label = torch.full((b_size,), real_label, device=device)
			# Forward pass real batch through D
			output = netD(real_cpu).view(-1)
			# Calculate loss on all-real batch
			errD_real = criterion(output, label)
			# Calculate gradients for D in backward pass
			errD_real.backward()
			D_x = output.mean().item()

			## Train with all-fake batch
			# Generate batch of latent vectors
			noise = torch.randn(b_size, latent_size, 1, 1, device=device)
			# Generate fake image batch with G
			fake = netG(noise)
			label.fill_(fake_label)
			# Classify all fake batch with D
			output = netD(fake.detach()).view(-1)
			# Calculate D's loss on the all-fake batch
			errD_fake = criterion(output, label)
			# Calculate the gradients for this batch
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			# Add the gradients from the all-real and all-fake batches
			errD = errD_real + errD_fake
			# Update D
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			###########################
			netG.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			# Since we just updated D, perform another forward pass of all-fake batch through D
			output = netD(fake).view(-1)
			# Calculate G's loss based on this output
			errG = criterion(output, label)
			# Calculate gradients for G
			errG.backward()
			D_G_z2 = output.mean().item()
			# Update G
			optimizerG.step()

			# Output training stats
			if i % 50 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, num_epochs, i, len(dataloader),
						 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

			# Save Losses for plotting later
			G_losses.append(errG.item())
			D_losses.append(errD.item())

			# Check how the generator is doing by saving G's output on fixed_noise
			if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
				with torch.no_grad():
					fake = netG(fixed_noise).detach().cpu()
				img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

			iters += 1

	plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="D")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig('loss.svg')
	plt.clf()

	"""
	fig = plt.figure(figsize=(8,8))
	plt.axis("off")
	ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
	ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
	HTML(ani.to_jshtml())
	"""

	# Grab a batch of real images from the dataloader
	real_batch = next(iter(dataloader))

	# Plot the real images
	plt.figure(figsize=(15,15))
	plt.subplot(1,2,1)
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

	# Plot the fake images from the last epoch
	plt.subplot(1,2,2)
	plt.axis("off")
	plt.title("Fake Images")
	plt.imshow(np.transpose(img_list[-1],(1,2,0)))
	plt.savefig("real_vs_fake.svg")

if __name__ == "__main__":
	import sys
	if len(sys.argv) == 2:
		train(sys.argv[1])
