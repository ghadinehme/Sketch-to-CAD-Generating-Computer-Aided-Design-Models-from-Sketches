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
import math
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

cfg = ConfigAE('test')
cfg.batch_size = 32
cfg.ckpt = '1000'


# Split Data
test_data = shuffled_df[:32*256].reset_index(drop=True)
val_data = shuffled_df[32*256:64*256].reset_index(drop=True)
train_data = shuffled_df[64*256:649*256].reset_index(drop=True)

# Small Dataset to check Overfitting capacity of model
# test_data = shuffled_df[:256].reset_index(drop=True)
# val_data = shuffled_df[256:2*256].reset_index(drop=True)
# train_data = shuffled_df[2*256:3*256].reset_index(drop=True)

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
            # T.Lambda(lambda tensor: 1-((-tensor + 1)==0).float()),
            T.Lambda(lambda tensor: 1-tensor),
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
        return image, y_label[:, 0], y_label[:, 1:]

traindata = sketchDataset(train_data)
testdata = sketchDataset(test_data)
valdata = sketchDataset(val_data)

batch_size = 256
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

# show_batch(train_dataloader)


class TransformerEncoder(nn.Module):
    def __init__(self, input_channels, image_size, patch_size, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()

        num_patches = (image_size // patch_size) ** 2
        patch_dim = input_channels * patch_size ** 2

        self.patch_embedding = nn.Conv2d(input_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = PositionalEncoding(embedding_dim, num_patches)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(51200, 256)

        self.tanh = nn.Tanh()

    def forward(self, input):
        embedded_input = self.patch_embedding(input)
        batch_size, _, _, _ = embedded_input.size()
        embedded_input = embedded_input.permute(0, 2, 3, 1).view(batch_size, -1, embedded_input.size(1))

        positional_encoded_input = self.positional_encoding(embedded_input)

        output = positional_encoded_input
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)

        return self.tanh(self.fc(output.reshape(256, -1)))

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Multi-head self-attention
        x = self.layer_norm1(x)
        x = x.permute(1, 0, 2)
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.dropout(x)
        x = residual + x

        # Feed forward network
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x




# create network and training agent
tr_agent = TrainerAE(cfg)

# load from checkpoint if provided
tr_agent.load_ckpt(cfg.ckpt)


embedding_dim = 512
num_heads = 8
hidden_dim = 512
num_layers = 8


model = TransformerEncoder(1, 240, 24, embedding_dim, num_heads, hidden_dim, num_layers)


# model.load_state_dict(torch.load('cnnencoder.pth'))


decoder = tr_agent.net.decoder
encoder = tr_agent.net.encoder

count_parameters(decoder)

bottleneck = tr_agent.net.bottleneck

count_parameters(bottleneck)

count_parameters(model)

print("Number of training samples:", n_datasample)

criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as desired

# Freeze the decoder parameters
for param in encoder.parameters():
    param.requires_grad = False

for param in decoder.parameters():
    param.requires_grad = False

# Training loop
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
loss_func = CADLoss(cfg).cuda()
losses = []

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
        outputs = decoder(model(images))
        output = {}
        output["tgt_commands"], output["tgt_args"] = labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2)
        output["command_logits"], output["args_logits"] = outputs
        #gt = encoder(labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2))
        #loss = criterion(outputs, gt[0])
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
            outputs = decoder(model(images))
            output = {}
            output["tgt_commands"], output["tgt_args"] = labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2)
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

    # Step the learning rate scheduler
    scheduler.step()
    losses.append((train_loss, val_loss))

    accs.append((train_acc_cmd.item(), train_acc_args.item(), val_acc_cmd.item(), val_acc_args.item()))

    torch.save(model.state_dict(), 'transformerencoder_'+str(epoch)+'.pth')

    with open("acc.txt", 'w') as file:
        for item in accs:
            file.write(str(item) + '\n')
    with open("loss.txt", 'w') as file:
            for item in losses:
                file.write(str(item) + '\n')

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
        outputs = decoder(model(images))
        output = {}
        output["tgt_commands"], output["tgt_args"] = labels_cmd.permute(1, 0), labels_param.permute(1, 0, 2)
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



torch.save(model.state_dict(), 'transformerencoder.pth')

np.savetxt('output.txt', outputs[0].argmax(dim = -1).to('cpu').numpy())
np.savetxt('labels.txt', labels_cmd.permute(1, 0).to('cpu').numpy())

