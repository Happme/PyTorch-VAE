# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器：将输入映射到潜在空间
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # 均值
        self.fc22 = nn.Linear(400, latent_dim)  # 方差

        # 解码器：将潜在空间映射回图像空间
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x.view(-1, 28 * 28)))
        return self.fc21(h1), self.fc22(h1)  # 返回均值和对数方差

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # 方差的平方根
        eps = torch.randn_like(std)  # 随机噪声
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)).view(-1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

