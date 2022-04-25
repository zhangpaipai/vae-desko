import torch
import numpy as np
import torch.nn as nn
from variant import *
import torch.nn.functional as F
from torch.distributions import Normal

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算卷积后的图片尺寸
def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
    
    return (size + 2*padding - kernel_size) // stride + 1


w = 400
h = 600
convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
intermediate_dim = 32

class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()
    def forward(self,z_mean,z_log_var):
        sigma = (z_log_var/2).exp()
        normal = Normal(z_mean, sigma)
        
        phi = normal.rsample()
        log_prob = normal.log_prob(phi)

        return phi, log_prob


class VaeEncoder(nn.Module):
    def __init__(self, args):
        super(VaeEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)   # Feature is set to 16, 16 channels in vision processing
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # Using conv2d_size_out for 3 times as 3 convolution net is designed in NN
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_dim  = convw * convh * 32
        self.Dense=nn.Linear(linear_input_dim, intermediate_dim)
        # 输出隐空间维
        self.z_mean=nn.Linear(intermediate_dim, args['latent_dim'])
        # 输出隐空间维
        self.z_log_var=nn.Linear(intermediate_dim, args['latent_dim'])
        self.sample=Sample()
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))       
        x = x.view(x.size(0), -1)
        o = F.relu(self.Dense(x))
        # 隐空间均值、log方差 
        z_mean = self.z_mean(o)
        z_log_var = self.z_log_var(o)
        # 重采样计算 phi, log_prob
        phi, log_prob = self.sample(z_mean,z_log_var)
        
        return phi, log_prob, z_mean, z_log_var


class VaeDecoder(nn.Module):
    def __init__(self, args):
        super(VaeDecoder, self).__init__()
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_dim  = convw * convh * 32
        self.Dense=nn.Linear(args['latent_dim'], intermediate_dim)
        self.out=nn.Linear(intermediate_dim, linear_input_dim)
        self.conv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,z):
        o=F.relu(self.Dense(z))
        o=self.out(o).view(o.size(0), 32, convw, convh)
        o=F.relu(self.conv1(o))
        o=F.relu(self.conv2(o))
        o=self.sigmoid(self.conv3(o))
        return o


class Vae(nn.Module):
    def __init__(self, args):
        super(Vae, self).__init__()
        self.encoder=VaeEncoder(args)
        self.decoder=VaeDecoder(args)
    def forward(self,x):
        phi, log_prob, mean, log_var=self.encoder(x)
        return self.decoder(phi), mean, log_var

