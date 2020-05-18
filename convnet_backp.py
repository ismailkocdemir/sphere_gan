import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        #self.main = self.make_layers(cfg)
         
         _linear = nn.Linear(128, 6*6*512, bias=False)
        _reshape = View((-1,512, 6, 6))

        self.weights_init()

    def forward(self, input):
        return self.main(input)

    def make_layers(self, cfg):
        layers = []
        _linear = nn.Linear(128, 6*6*512, bias=False)
        _reshape = View((-1,512, 6, 6))
        layers += [ _linear, _reshape, nn.BatchNorm2d(cfg[0]), nn.ReLU(True) ]

        in_channels = 512
        for v in cfg[1:]:
            conv2d = nn.ConvTranspose2d(in_channels, v, 4, 2, 1, bias=False)
            layers += [ conv2d, nn.BatchNorm2d(v), nn.ReLU(True) ]
            in_channels = v

        conv2d = nn.ConvTranspose2d(cfg[-1], 3, 3, 1, 1, bias=False)
        layers += [ conv2d,  nn.Tanh() ]
        
        return nn.Sequential(*layers)

    def weights_init(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)    


class Discriminator(nn.Module):
    def __init__(self, cfg,):
        super(Discriminator, self).__init__()
        self.main = self.make_layers(cfg)
        self.weights_init()

    def forward(self, input):
        return self.main(input)

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i,v in enumerate(cfg):
            conv2d_1 = nn.Conv2d(in_channels, v, 3, 1, 1, bias=False)
            conv2d_2 = nn.Conv2d(v, v, 4, 2, 2, bias=False)
            
            curr_size = 48/(2**(i))
            ln_1 = nn.LayerNorm([v,curr_size,curr_size])
            curr_size /= 2
            ln_2 = nn.LayerNorm([v,curr_size,curr_size])

            layers += [ conv2d_1, ln_1, nn.LeakyReLU(0.2, inplace=True), conv2d_2, ln_2, nn.LeakyReLU(0.2, inplace=True) ]
            in_channels = v

        #_flatten = Flatten()
        _inverse_projection = GeometricBlock(pool=False)
        layers.append(_inverse_projection)
    
        return nn.Sequential(*layers)

    def weights_init(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)    


class GeometricBlock(nn.Module):
    def __init__(self, pool=False):
        super().__init__()
        self.pool = pool

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
        u = nn.Linear(u.size()[-1], 1024, bias=False)

        # Inverse Projection
        u_hat = 2*u / (torch.pow(torch.norm(u, dim=1), 2) + 1)
        v = (torch.pow(torch.norm(u, dim=1), 2) -1 )/(torch.pow(torch.norm(u, dim=1), 2) + 1)
        return torch.cat((u_hat, v), dim=1)


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

