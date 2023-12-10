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
import optuna

from prettytable import PrettyTable

torch.cuda.empty_cache()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print("Total Trainable Params: ", total_params)
    return total_params

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

shuffled_df = pd.read_csv('selected_data.csv')
# remove representations with length = 7
shuffled_df = shuffled_df[shuffled_df["rep_len"] == 7]
# print("Computing lengths of representations...")
# lengths = [len(ast.literal_eval(el)) for el in shuffled_df["Rep"]]
# shuffled_df["rep_len"] = lengths
# print("Done.")
# print("Selecting representations with length < 32...")
# selected_df = shuffled_df[shuffled_df["rep_len"] < 32]
# # selected_df = selected_df[["Name", "Rep"]]
# shuffled_df = selected_df
# # save shuffled_df to csv
# shuffled_df.to_csv('new_data.csv', index=False)
n_datasample = len(shuffled_df)
print("Number of training samples:", n_datasample)


cfg = ConfigAE('test')
cfg.batch_size = 64
cfg.ckpt = '1000'

l = 256
seq_len = 7
batch_size = 128
# Split Data
# test_data = shuffled_df[:32*l].reset_index(drop=True)
# val_data = shuffled_df[32*l:64*l].reset_index(drop=True)
# train_data = shuffled_df[64*l:649*l].reset_index(drop=True)


# # Small Dataset to check Overfitting capacity of model
test_data = shuffled_df[:l].reset_index(drop=True)
val_data = shuffled_df[l:l+2*l].reset_index(drop=True)
train_data = shuffled_df[l+2*l:l+3*l].reset_index(drop=True)

IMG_DIR = 'pics/'

# Get ground truth labels from DeepCAD outputa
class sketchDataset(Dataset):
    # Takes a pandas dataframe input of image filenames under 'Name' and labels under 'Rep'
    def __init__(self, df):
        self.X = df['Name']
        self.y = df['Rep']
        # transform normalizes and prepares image for pretraind VGG network (
        # norm µ and sigma from data used for VGG training)
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(240),
            T.ToTensor(),
            T.Lambda(lambda tensor: ((-tensor + 1)>0).float()),
            # T.Normalize(mean=np.mean([0.485, 0.456, 0.406]), std=np.mean([0.229, 0.224, 0.225]))
        ])

    def __len__(self):
        # Denotes the total number of samples
        return len(self.X)

    def __getitem__(self, index):
        # Generates one sample of data
        imageFileName = self.X[index]
        label = ast.literal_eval(self.y[index])
        pad_token = [3] + 16 * [-1]
        label_pad = label + (seq_len - len(label)) * [pad_token]
        y_label = torch.tensor(label_pad)
        #label_pad_type = torch.nn.functional.one_hot(y_label[:, 0], num_classes=6)
        #label_pad_param = torch.nn.functional.one_hot(y_label[:, 1:] + 1, num_classes=257)

        # print(imageFileName
        imageFileName = "".join(["0" for i in range(8-len(str(imageFileName)))]) + str(imageFileName)
        image = Image.open(IMG_DIR + imageFileName + '.png')
        image = image.convert('L')
        #plt.imshow(image) # check to ensure image is loaded
        #plt.show()
        image = self.transform(image)

        def expand(array, a):
            # Define a kernel for convolution that checks for neighbors
            # Define a kernel for convolution that checks for neighbors
            kernel = torch.tensor([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], dtype=torch.float32)

            # Apply convolution to the input tensor
            convolved = torch.nn.functional.conv2d(array.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)

            # Set positions with values greater than 0 to 1
            return ((convolved > 0).float()[0]-image)*a + image
        
        alpha = [0.7**j for j in range(6)]
        for i in range(len(alpha)):
            image = expand(image, alpha[i])
        return image, y_label[:, 0], y_label[:, 1:]

traindata = sketchDataset(train_data)
testdata = sketchDataset(test_data)
valdata = sketchDataset(val_data)


def show_images(images, nmax=16):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=4).permute(1, 2, 0))

def show_batch(dl, nmax=16):
    for images in dl:
        show_images(images[0], nmax)
        print(images[0].shape)
        break

    
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
        features = F.elu(features)
        m = self.fc_mean(features)
        h = self.fc_logvar(features)
        v = F.softplus(h) + 1e-8
        return features, m, v


class CNNDecoderVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CNNDecoderVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # self.fc2 = nn.Linear(latent_dim, 8192)  # Reverse the dimensionality reduction from encoder
        # self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
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
        epsilon = torch.randn_like(mean)
        z = mean + torch.sqrt(logvar) * epsilon
        
        # Decode from the latent representation
        reconstructed_x, _, _ = self.decoder(z, mean, logvar)
        
        return reconstructed_x, mean, logvar  


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_cmd, n_params, d_model, nhead, num_encoder_layers, num_decoder_layers, weight_decay):
        super(TransformerAutoencoder, self).__init__()

        self.n_cmd = n_cmd
        self.n_params = n_params
        self.d_model = d_model

        # Transformer
        # transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=8192, nhead=nhead)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        # self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_decoder_layers)

        # Linear layer to get class probabilities
        self.fc0 = nn.Linear(8192, self.d_model)
        self.fc = nn.ModuleList([nn.Linear(8192, self.d_model) for _ in range(seq_len-1)])
        self.decoder_cmd = nn.Linear(d_model, n_cmd)
        self.decoder_params = nn.Linear(d_model, n_params*16, weight_decay)




    def forward(self, z):
        # Transformer encoding and decoding
        # encoded = self.transformer_encoder(z)
        # z = self.fc0(z).view(1, -1, self.d_model)
        # y = torch.tensor([[[0.0,0.0,1.0]+[0.0]*(self.d_model-3) for i in range(batch_size)] for j in range(60)]).to(device)
        arr = []
        u = F.relu(self.fc0(z).view(1, -1, self.d_model))
        arr.append(u)
        for i in range(seq_len-1):
            u = F.relu(self.fc[i](z).view(1, -1, self.d_model))
            arr.append(u)
        y = torch.cat(arr, dim=0)
        decoded = self.transformer_decoder(y, y)

        # Class prediction
        # decoded = decoded.permute(1, 0, 2)
        cmd = self.decoder_cmd(decoded)
        params = self.decoder_params(decoded)
        # Reshape output to original input shape: [batch_size, sequence_length, 16, num_classes]
        out = (cmd.view(seq_len, -1, self.n_cmd).permute(1,0,2), params.view(seq_len, -1, 16, self.n_params).permute(1,0,2,3))
        # Calculate L1 regularization loss for linear layers
        l1_loss = torch.tensor(0.0).to(z.device)  # Initialize with 0
        l1_loss += torch.norm(self.fc0.weight, p=1)
        for fc_layer in self.fc:
            l1_loss += torch.norm(fc_layer.weight, p=1)  # L1 norm of weights

        return out, l1_loss


#use optuna to find best hyperparameters
def objective(trial):
    batch_size = trial.suggest_int("batch_size", 16, 128, log = True)
    train_dataloader = DataLoader(traindata, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(testdata, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(valdata, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Define the model
    alpha = trial.suggest_float("alpha", 1, 5)
    beta = trial.suggest_float("beta", 1, 5)
    gamma = trial.suggest_float("gamma", 1, 5)
    d_model = trial.suggest_int("d_model", 256, 1024, step = 256)
    nhead = trial.suggest_int("nhead", 8, 16, step = 8)
    num_decoder_layers = trial.suggest_int("num_encoder_layers", 2, 32)
    model = TransformerAutoencoder(n_cmd = 6, n_params = 257, d_model=d_model, nhead=nhead, num_encoder_layers=4, num_decoder_layers=num_decoder_layers, weight_decay=0)
    encoder = VAE(8192)
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-3))
    optimizer2 = optim.Adam(encoder.parameters(), lr=trial.suggest_float("lr2", 1e-5, 1e-3))
    # continue
    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)  # Adjust step_size and gamma as desired
    scheduler2 = StepLR(optimizer2, step_size=10, gamma=0.8)
    # continue
    # Freeze the decoder parameters
    for param in model.parameters():
        param.requires_grad = True
    # continue
    for param in encoder.parameters():
        param.requires_grad = True
    # continue
    criterion = nn.BCELoss()
    # Training loop
    num_epochs = 5
    model.to(device)
    encoder.to(device)
    model.train()
    loss_func = CADLoss(cfg).cuda()

    for epoch in range(num_epochs):
        data_loader = tqdm(train_dataloader, total=len(train_dataloader))

        model.train()
        train_loss = 0.0
        train_acc_cmd = 0
        train_acc_args = 0
        train_total = 0
        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
        loss_4 = 0
        for images, labels_cmd, labels_param in data_loader:
            images, labels_cmd, labels_param = images.to(device), labels_cmd.to(device), labels_param.to(device)
            optimizer.zero_grad()
            x_rec, m, v = encoder(images)
            epsilon = torch.randn_like(m)
            z = m + torch.sqrt(v) * epsilon
            outputs, l1_loss = model(z)
            output = {}
            output["tgt_commands"], output["tgt_args"] = labels_cmd, labels_param
            output["command_logits"], output["args_logits"] = outputs
            loss_dict = loss_func(output)
            rec = criterion(x_rec, images)
            loss_1 += loss_dict['loss_cmd']
            loss_2 += loss_dict['loss_args']
            loss_3 += rec
            loss_4 += l1_loss
            loss = alpha*loss_dict['loss_cmd'] + beta*loss_dict['loss_args'] + gamma*rec
            data_loader.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss Cmd: {loss_dict["loss_cmd"]:.4f}, Loss Arg: {loss_dict["loss_args"]:.4f}, Rec: {rec:.4f}')
            loss.backward()
            optimizer.step()
            optimizer2.step()
            train_loss += loss.item()

            train_total += 1
            # Step the learning rate scheduler
            scheduler.step()
            scheduler2.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_total = 0
    with torch.no_grad():
        for images, labels_cmd, labels_param in val_dataloader:
            images, labels_cmd, labels_param = images.to(device), labels_cmd.to(device), labels_param.to(device)
            x_rec, m, v = encoder(images)
            epsilon = torch.randn_like(m)
            z = m + torch.sqrt(v) * epsilon
            outputs, l1_loss = model(z)
            output = {}
            output["tgt_commands"], output["tgt_args"] = labels_cmd, labels_param
            output["command_logits"], output["args_logits"] = outputs
            loss_dict = loss_func(output)
            rec = criterion(x_rec, images)

            loss = loss_dict['loss_cmd'] + loss_dict['loss_args'] + rec
            val_loss += loss.item()

            val_total += 1

    # Print training and validation loss and accuracy for each epoch
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    loss_1 /= len(train_dataloader)
    loss_2 /= len(train_dataloader)
    loss_3 /= len(train_dataloader)
    loss_4 /= len(train_dataloader)
    print(loss_1)
    print(loss_2)
    print(loss_3)
    print(loss_4)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    

    return val_loss

def optimize_hyperparameters():
    study = optuna.create_study(direction='minimize')  # Minimize validation loss
    study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

    best_params = study.best_params
    best_loss = study.best_value

    print(f"Best hyperparameters: {best_params}")
    print(f"Best validation loss: {best_loss}")

if __name__ == "__main__":
    optimize_hyperparameters()
