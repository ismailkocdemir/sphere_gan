from torch import nn
from torch.autograd import grad
import torch

from utils import *

class conv3x3(nn.Module):
    def __init__(self, input_dim, output_dim = None, bias = False):
        super(conv3x3, self).__init__()
        
        if output_dim == None:
            output_dim = input_dim

        self.conv = nn.Conv2d(input_dim, output_dim, 3, stride=1, padding=1, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ResidualBlockDownSample(nn.Module):
    def __init__(self, input_dim,  size=64):
        super(ResidualBlock, self).__init__()
        half_size = size//2
        self.avg_pool1 = nn.AdaptiveAvgPool2d((half_size,half_size))
        self.conv_shortcut = Conv2d(input_dim, input_dim, kernel_size = 1)

        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()

        self.ln1 = nn.LayerNorm([input_dim, size, size])
        self.ln2 = nn.LayerNorm([input_dim, size, size])
        
        self.conv_1 = conv3x3(input_dim, input_dim, bias = False)
        self.conv_2 = conv3x3(input_dim, input_dim, bias = False)
        
        self.avg_pool2 = nn.AdaptiveAvgPool2d((half_size,half_size))
        
        
    def forward(self, input):
        shortcut = self.avg_pool1(input)
        shortcut = self.conv_shortcut(shortcut)

        output = self.ln1(input)
        output = self.relu1(output)
        output = self.conv_1(output)

        output = self.avg_pool(output)

        output = self.ln2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ResidualBlockUpSample(nn.Module):
    def __init__(self, input_dim, size=64):
        super(ResidualBlock, self).__init__()
        
        self.upsample1 = nn.UpSample(scale_factor=2)
        self.conv_shortcut = Conv2d(input_dim, input_dim, kernel_size = 1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)
        
        self.conv_1 = conv3x3(input_dim, input_dim, bias = False)
        self.conv_2 = conv3x3(input_dim, input_dim, bias = False)        
        self.upsample2 = nn.UpSample(scale_factor=2)
        
        
    def forward(self, input):
        shortcut = self.upsample1(input)
        shortcut = self.conv_shortcut(shortcut)

        output = self.bn1(input)
        output = self.relu1(output)
        output = self.conv_1(output)

        output = self.upsample2(output)

        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, size):
        super(ResidualBlock, self).__init__()
    
        self.conv_shortcut = Conv2d(input_dim, input_dim, kernel_size = 1)

        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()

        self.ln1 = nn.LayerNorm([input_dim, size, size])
        self.ln2 = nn.LayerNorm([input_dim, size, size])
        
        self.conv_1 = conv3x3(input_dim, input_dim, bias = False)
        self.conv_2 = conv3x3(input_dim, input_dim, bias = False)        
        
        
    def forward(self, input):
        shortcut = self.conv_shortcut(shortcut)

        output = self.ln1(input)
        output = self.relu1(output)
        output = self.conv_1(output)

        output = self.ln2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output



class GeneratorResNet(nn.Module):
    def __init__(self, dim=256, output_dim=3*64*64):
        super(Generator, self).__init__()

        self.dim = dim
        
        #self.ln1 = nn.Linear(128, 3*3*512, bias=False)
        #self.reshape = View((-1, 512, 3, 3))
        #self.conv1 = conv3x3(512, self.dim, 3)
        
        self.ln1 = nn.Linear(128, 3*3*self.dim, bias=False)
        self.reshape = View((-1, self.dim, 3, 3))
        
        self.rb1 = ResidualBlockUpSample(self.dim, size=3)
        self.rb2 = ResidualBlockUpSample(self.dim, size=6)
        self.rb3 = ResidualBlockUpSample(self.dim, size=12)
        self.rb4 = ResidualBlockUpSample(self.dim, size=24)
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = conv3x3(self.dim, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input) #self.ln1(input.contiguous())
        output = self.reshape(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        #output = output.view(-1, 3 * self.dim * self.dim)
        return output

class DiscriminatorResNet(nn.Module):
    def __init__(self, dim=256):
        super(Discriminator, self).__init__()

        self.dim = dim

        self.conv1 = conv3x3(3, self.dim, 3)
        self.rb1 = ResidualBlockDownSample(self.dim, size=48)
        self.rb2 = ResidualBlockDownSample(self.dim, size=24)
        self.rb3 = ResidualBlockDownSample(self.dim, size=12)
        self.rb4 = ResidualBlockDownSample(self.dim, size=6)
        self.rb5 = ResidualBlock(self.dim, size=3)

        self.gb = GeometricBlock(dim=self.dim, pool=True)


    def forward(self, input):
        output = input
        
        #output = output.view(-1, 3, self.dim, self.dim)
        
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.rb5(output)
        output = self.gb(output)
        
        return output