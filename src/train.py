# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from src.model import VAE

def loss_function(recon_x, x, mu, logvar):
    # 重构损失（使用BCE损失）
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')

    # KL散度
    # 计算KL散度：0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # mu是均值，logvar是对数方差
    # std = exp(0.5 * logvar)，然后通过标准差来生成潜在变量z
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失：重构损失 + KL散度
    return BCE + KL

def train(model, train_loader, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')

    # 保存模型参数
    torch.save(model.state_dict(), 'vae_state_dict.pth')

def main():
    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载MNIST数据集
    transform = transforms.ToTensor()
    # 下载的数据的二进制文件，非标准图像数据
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    
    # 初始化模型
    latent_dim = 20
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练模型
    num_epochs = 10
    train(model, train_loader, optimizer, num_epochs)


if __name__ == '__main__':
    main()
