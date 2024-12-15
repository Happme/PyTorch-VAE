# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from src.model import VAE

def main():
    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent_dim = 20
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load('vae_state_dict.pth', weights_only=True))

    # 随机数据
    z = torch.randn(1, latent_dim).to(device)

    # 要放在加载模型和参数之后
    model.eval()

    with torch.no_grad():       # 推理过程中不更新梯度
        generate_img = model.decode(z)
        print(f'generate_img size: {generate_img.shape}')
        plt.imshow(generate_img[0].cpu().numpy(), cmap='gray')
        plt.savefig('generate_img.png')
    

if __name__ == '__main__':
    main()
