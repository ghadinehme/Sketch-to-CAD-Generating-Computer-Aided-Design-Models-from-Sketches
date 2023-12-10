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
from sketchAE import VAE
from CADAE import TransformerAutoencoder
import pandas as pd
import ast
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt
import json

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

shuffled_df = pd.read_csv('data.csv')

shuffled_df.head()

n_datasample = len(shuffled_df)
print("Number of training samples:", n_datasample)


# Split Data
# test_data = shuffled_df[:32*256].reset_index(drop=True)
# val_data = shuffled_df[32*256:64*256].reset_index(drop=True)
# train_data = shuffled_df[64*256:649*256].reset_index(drop=True)



# Small Dataset to check Overfitting capacity of model
test_data = shuffled_df[:64].reset_index(drop=True)
val_data = shuffled_df[64:64+2*64].reset_index(drop=True)
train_data = shuffled_df[64+2*64:64+3*64].reset_index(drop=True)

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
        label_pad = label + (60 - len(label)) * [pad_token]
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

batch_size = 64
train_dataloader = DataLoader(traindata, batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(testdata, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(valdata, batch_size, shuffle=True, num_workers=2, pin_memory=True)

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

show_batch(train_dataloader)
    
  
model = VAE(8192)
model.load_state_dict(torch.load("reconstruct/vae_sketch_19.pth"))

# class Bottleneck(nn.Module):
#     def __init__(self):
#         super(Bottleneck, self).__init__()
#         n_dim = 8192
#         h_dim = 8192
#         z_dim = 512
#         self.bottleneck = nn.Sequential(
#             nn.Linear(n_dim, 60*512),
#             nn.Tanh(),
#         )

#     def forward(self, z):
#         return self.bottleneck(z).reshape(-1, 60, 512)

# bottleneck = Bottleneck()

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to unsqueeze the input tensor to add the time step dimension (sequence length = 8192)
        x = x.unsqueeze(1)
        
        # Forward pass through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Reshape the output to match the desired output shape (batch_size, 60, 512)
        out = self.linear(out)
        
        return out.reshape(-1,60, 512)

# Define the input shape and create the model
input_size = 8192
hidden_size = 4096
num_layers = 1
output_size = 512*60

bottleneck = LSTMModel(input_size, hidden_size, num_layers, output_size)

count_parameters(model)

cfg = ConfigAE('test')
cfg.batch_size = 32
cfg.ckpt = '1000'



# Example usage
decoder = TransformerAutoencoder(n_cmd = 6, n_params = 257, d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=4)
# decoder.load_state_dict(torch.load("cadAE/cad_vae15.pth"))

def decode(x, d):
    decoded = d.transformer_decoder(x, x)

    # Class prediction
    cmd = d.decoder_cmd(decoded)
    params = d.decoder_params(decoded)
    # Reshape output to original input shape: [batch_size, sequence_length, 16, num_classes]
    out = (cmd.view(-1, 60, d.n_cmd), params.view(-1, 60, 16, d.n_params))
    return out


count_parameters(bottleneck)

print("Number of training samples:", n_datasample)

# Freeze the decoder parameters
for param in bottleneck.parameters():
    param.requires_grad = True

for param in decoder.parameters():
    param.requires_grad = True

for param in model.encoder.parameters():
    param.requires_grad = True

for param in model.decoder.parameters():
    param.requires_grad = False



# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=2, gamma=0.8)  # Adjust step_size and gamma as desired


# Training loop
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
decoder.to(device)
bottleneck.to(device)
model.train()
loss_func = CADLoss(cfg).cuda()
losses = [(10,10)]

acc_func = CADAccuracy(cfg).cuda()
accs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_acc_cmd = 0
    train_acc_args = 0
    train_total = 0
    loss_1 = 0
    loss_2 = 0
    for images, labels_cmd, labels_param in train_dataloader:
        images, labels_cmd, labels_param = images.to(device), labels_cmd.to(device), labels_param.to(device)
        optimizer.zero_grad()
        outputs = decode(bottleneck(model.encoder(images)[0]), decoder)
        output = {}
        output["tgt_commands"], output["tgt_args"] = labels_cmd, labels_param
        output["command_logits"], output["args_logits"] = outputs
        loss_dict = loss_func(output)
        loss_1 += loss_dict['loss_cmd']
        loss_2 += loss_dict['loss_args']
        loss = loss_dict['loss_cmd'] + loss_dict['loss_args']
        loss.backward()
        # max_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        train_loss += loss.item()

        acc_dict = acc_func(output)
        train_acc_cmd += acc_dict["acc_cmd"] * 100
        train_acc_args += acc_dict["acc_args"] * 100
        train_total += 1
    

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_acc_cmd = 0
    val_acc_args = 0
    val_total = 0
    with torch.no_grad():
        for images, labels_cmd, labels_param in val_dataloader:
            images, labels_cmd, labels_param = images.to(device), labels_cmd.to(device), labels_param.to(device)
            # outputs = model(images)
            # gt = encoder(labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2))
            # loss = criterion(outputs, gt[0])
            outputs = decode(bottleneck(model.encoder(images)[0]), decoder)
            output = {}
            output["tgt_commands"], output["tgt_args"] = labels_cmd, labels_param
            output["command_logits"], output["args_logits"] = outputs
            #gt = encoder(labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2))
            #loss = criterion(outputs, gt[0])
            loss_dict = loss_func(output)
            loss = loss_dict['loss_cmd'] + loss_dict['loss_args']
            val_loss += loss.item()

            acc_dict = acc_func(output)
            val_acc_cmd += acc_dict["acc_cmd"] * 100
            val_acc_args += acc_dict["acc_args"] * 100
            val_total += 1

    # Print training and validation loss and accuracy for each epoch
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    loss_1 /= len(train_dataloader)
    loss_2 /= len(train_dataloader)
    val_acc_cmd /= val_total
    val_acc_args /= val_total
    train_acc_cmd /= train_total
    train_acc_args /= train_total
    print(loss_1)
    print(loss_2)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy CmdSS: {train_acc_cmd:.4f}, Train Accuracy Args: {train_acc_args:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy Cmd: {val_acc_cmd:.4f}, Val Accuracy Args: {val_acc_args:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(outputs[0].shape, outputs[1].shape)
    # Step the learning rate scheduler
    scheduler.step()
    losses.append((train_loss, val_loss))

    accs.append((train_acc_cmd.item(), train_acc_args.item(), val_acc_cmd.item(), val_acc_args.item()))
    if epoch%5==0:
        torch.save(model.state_dict(), 'encoder_cad'+str(epoch)+'.pth')
        torch.save(bottleneck.state_dict(), 'bottleneck_cad'+str(epoch)+'.pth')

    with open("acc.txt", 'w') as file:
        for item in accs:
            file.write(str(item) + '\n')
    with open("loss.txt", 'w') as file:
            for item in losses:
                file.write(str(item) + '\n')

    with open("my_dict.txt", 'w') as file:
            file.write(str(list(outputs[0].argmax(-1))) + '\n')
    with open("my_dict.txt", 'w') as file:
            file.write(str(list(labels_cmd.permute(1, 0))) + '\n')

model.eval()
test_loss = 0.0
acc_cmd = 0
acc_args = 0
test_total = 0
with torch.no_grad():
    for images, labels_cmd, labels_param in test_dataloader:
        images, labels_cmd, labels_param = images.to(device), labels_cmd.to(device), labels_param.to(device)
        # outputs = model(images)
        # gt = encoder(labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2))
        # loss = criterion(outputs, gt[0])
        outputs = decode(bottleneck(model.encoder(images)[0]), decoder)
        output = {}
        output["tgt_commands"], output["tgt_args"] = labels_cmd, labels_param
        output["command_logits"], output["args_logits"] = outputs
        #gt = encoder(labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2))
        #loss = criterion(outputs, gt[0])
        loss_dict = loss_func(output)
        loss = loss_dict['loss_cmd'] + loss_dict['loss_args']
        test_loss += loss.item()

        # Calculate accuracy
        acc_dict = acc_func(output)
        acc_cmd += acc_dict["acc_cmd"]
        acc_args += acc_dict["acc_args"]
        test_total += 1

test_loss /= len(test_dataloader)
test_accuracy = acc_cmd / test_total * 100
test_accuracy_1 = acc_args / test_total * 100
print(f"Test Loss: {test_loss:.4f}, Test Accuracy Cmd: {test_accuracy:.2f}%, Test Accuracy Args: {test_accuracy_1:.2f}%")
