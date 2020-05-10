import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, cfg, image_channels, latent_size, kernel_size):
        super(Generator, self).__init__()
        self.main = self.make_layers(cfg, image_channels, latent_size, kernel_size)
        '''
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_size, num_channels * 8, kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels * 8),
            nn.ReLU(True),
            # state size. (num_channels*8) x 4 x 4
            nn.ConvTranspose2d(num_channels * 8, num_channels * 4, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(True),
            # state size. (num_channels*4) x 8 x 8
            nn.ConvTranspose2d( num_channels * 4, num_channels * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(True),
            # state size. (num_channels*2) x 16 x 16
            nn.ConvTranspose2d( num_channels * 2, num_channels, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            # state size. (num_channels) x 32 x 32
            nn.ConvTranspose2d( num_channels, image_channels, kernel_size, 2, 1, bias=False),
            nn.Tanh()
            # state size. (image_channels) x 64 x 64
        )
        '''

        self.weights_init()

    def forward(self, input):
        return self.main(input)

    def make_layers(self, cfg, image_channels, latent_size, kernel_size):
        layers = []

        conv2d = nn.ConvTranspose2d( latent_size, cfg[0], kernel_size, 1, 0, bias=False)
        layers += [ conv2d, nn.BatchNorm2d(cfg[0]), nn.ReLU(True) ]

        in_channels = cfg[0]
        for v in cfg[1:]:
            conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size, 2, 1, bias=False)
            layers += [ conv2d, nn.BatchNorm2d(v), nn.ReLU(True) ]
            in_channels = v

        conv2d = nn.ConvTranspose2d(cfg[-1], image_channels, kernel_size, 2, 1, bias=False)
        layers += [ conv2d, nn.Tanh() ]
        
        return nn.Sequential(*layers)

    def weights_init(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)    


class Discriminator(nn.Module):
    def __init__(self, cfg, image_channels, kernel_size):
        super(Discriminator, self).__init__()
        self.main = self.make_layers(cfg, kernel_size, image_channels)
        
        """
        self.main = nn.Sequential(
            # input is (image_channels) x 64 x 64
            nn.Conv2d(image_channels, num_channels, kernel_size, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_channels) x 32 x 32
            nn.Conv2d(num_channels, num_channels * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_channels*2) x 16 x 16
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_channels*4) x 8 x 8
            nn.Conv2d(num_channels * 4, num_channels * 8, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_channels*8) x 4 x 4
            nn.Conv2d(num_channels * 8, 1, kernel_size, 1, 0, bias=False),
            nn.Sigmoid()
        )
        """

        self.weights_init()

    def forward(self, input):
        return self.main(input)

    def make_layers(self, cfg, kernel_size, image_channels):
        layers = []
        in_channels = image_channels
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size, 2, 1, bias=False)
            layers += [ conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(0.2, inplace=True) ]
            in_channels = v

        conv2d = nn.Conv2d(cfg[-1], 1, kernel_size, 1, 0, bias=False)
        layers += [ conv2d, nn.Sigmoid() ]
      
        return nn.Sequential(*layers)

    def weights_init(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)    
