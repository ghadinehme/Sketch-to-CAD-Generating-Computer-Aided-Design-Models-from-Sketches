from tqdm import tqdm
from dataset.cad_dataset import get_dataloader
from config import ConfigAE
from utils import ensure_dir
from trainer import TrainerAE
from trainer.loss import CADLoss
from trainer.accuracy import CADAccuracy
import torch
import numpy as np
import os
import h5py
#from cadlib.macro import EOS_IDX4

import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import ast
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt

from prettytable import PrettyTable

torch.cuda.empty_cache()


class CNNEncoderVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CNNEncoderVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # Linear layers for mean and log variance
        self.fc_mean = nn.Linear(8192, latent_dim)
        self.fc_logvar = nn.Linear(8192, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, x):
        # Encoder forward pass
        features = self.encoder(x)
        batch_size = features.size(0)
        features = features.view(batch_size, -1)  # Flatten the features
        return features, 1, 2
        
        # Calculate mean and log variance
        mean = self.fc_mean(features)
        logvar = self.fc_logvar(features)
        
        # Reparameterize to sample from the latent space
        z = self.reparameterize(mean, logvar)
        

        return z, mean, logvar


class CNNDecoderVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CNNDecoderVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Decoder architecture
        self.fc_mean = nn.Linear(latent_dim, 256)  # Linear layer for the mean
        self.fc_logvar = nn.Linear(latent_dim, 256)  # Linear layer for the log variance
        # self.bottleneck = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ReLU(),)
        self.fc2 = nn.Linear(latent_dim, 8192)  # Reverse the dimensionality reduction from encoder
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(512, 256, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Use sigmoid activation to constrain output between 0 and 1
        )

    def forward(self, z, mean, logvar):
        # Decode from the latent space z
    
        x = z.view(-1, 512, 4, 4)  # Reshape to match the last feature map size in the encoder
        x = self.decoder(x)
        return x, mean, logvar

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = CNNEncoderVAE(latent_dim)
        self.decoder = CNNDecoderVAE(latent_dim)

    def forward(self, x):
        # Encode input to obtain mean, logvar, and latent representation
        z, mean, logvar = self.encoder(x)
        
        # Decode from the latent representation
        reconstructed_x, _, _ = self.decoder(z, mean, logvar)
        
        return reconstructed_x, mean, logvar
    
