import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Linear(128, 6*6*self.dim, bias=False)
        self.reshape = View((-1,self.dim, 6, 6))
        self.bn = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU(True)

        self.conv1 = nn.ConvTranspose2d(self.dim, 256, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(v)
        self.relu1 = nn.ReLU(True)
        
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(v)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(v)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()

        self.weights_init()

    def forward(self, input):
        output = self.ln1(input)
        output = self.reshape(output)
        output = self.bn(output)
        output = self.relu(output)

        output = self.conv1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.tanh(output)

        return output

    def weights_init(self,):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)

class Discriminator(nn.Module):
    def __init__(self, cfg,):
        super(Discriminator, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.ln1_1 = nn.LayerNorm([64,48,48])
        self.lrelu1_1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv1_2 = nn.Conv2d(64, 64, 4, 2, 2, bias=False)
        self.ln1_2 = nn.LayerNorm([64,24,24])
        self.lrelu1_2 = nn.LeakyReLU(0.2, inplace=True)


        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.ln2_1 = nn.LayerNorm([128,24,24])
        self.lrelu2_1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2_2 = nn.Conv2d(128, 128, 4, 2, 2, bias=False)
        self.ln2_2 = nn.LayerNorm([128,12,12])
        self.lrelu2_2 = nn.LeakyReLU(0.2, inplace=True)


        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.ln3_1 = nn.LayerNorm([256,12,12])
        self.lrelu3_1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3_2 = nn.Conv2d(256, 256, 4, 2, 2, bias=False)
        self.ln3_2 = nn.LayerNorm([256,6,6])
        self.lrelu3_2 = nn.LeakyReLU(0.2, inplace=True)


        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.gb = GeometricBlock()
        
        self.weights_init()

    def forward(self, input):
        
        output = self.conv1_1(input)
        output = self.ln1_1(output)
        output = self.lrelu1_1(output)
        
        output = self.conv1_2(output)
        output = self.ln1_2(output)
        output = self.lrelu1_2(output)


        output = self.conv2_1(input)
        output = self.ln2_1(output)
        output = self.lrelu2_1(output)
        
        output = self.conv2_2(output)
        output = self.ln2_2(output)
        output = self.lrelu2_2(output)


        output = self.conv3_1(input)
        output = self.ln3_1(output)
        output = self.lrelu3_1(output)
        
        output = self.conv3_2(output)
        output = self.ln3_2(output)
        output = self.lrelu3_2(output)

        output = self.conv4(output)
        output = self.gb(output)


    def weights_init(self,):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)


class GeometricBlock(nn.Module):
    def __init__(self, pool=True):
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

