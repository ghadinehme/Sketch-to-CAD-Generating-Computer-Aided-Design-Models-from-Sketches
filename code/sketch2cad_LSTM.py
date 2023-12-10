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
            T.Lambda(lambda tensor: -tensor + 1),
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

show_batch(train_dataloader)

class CADModel(nn.Module):
    def __init__(self):
        super(CADModel, self).__init__()
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
        
        #self.fc1 = nn.Linear(224 * 224, 2048)
        self.fc1 = nn.Linear(8192, 512)
        
        
        # Decoder architecture
        self.lstm = nn.LSTMCell(512, hidden_size=512)
        self.decoder = nn.ModuleList([nn.LSTMCell(512, hidden_size=512) for _ in range(59)])
        self.fc_cmd = nn.Linear(512, 6)  # Output layer for step prediction
        self.fc_param = nn.Linear(512, 257 * 16)

    def forward(self, x, hidden):
        # Encoder forward pass
        features = self.encoder(x)
        batch_size = features.size(0)
        features = features.view(batch_size, -1)  # Flatten the features
        
        # Decoder forward pass
        fc2_output = F.relu(self.fc1(features))
        #output = torch.cat((tensor1.unsqueeze(1), tensor2.unsqueeze(1)), dim=1)
        c, h = self.lstm(fc2_output, hidden)
        output = (
            self.fc_cmd(F.relu(c)).unsqueeze(1), self.fc_param(F.relu(c)).unsqueeze(1)
        )
        
    
        for layer in self.decoder:
            c, h = layer(fc2_output, (c, h))
            output = (torch.cat((output[0], self.fc_cmd(F.relu(c)).unsqueeze(1)), dim=1), 
                      torch.cat((output[1], self.fc_param(F.relu(c)).unsqueeze(1)), dim=1))
        
        # Step prediction
        output = (
            torch.softmax(output[0].view(-1, 60, 6), dim=-1).permute(1, 0, 2),
            torch.softmax(output[1].view(-1, 60, 16, 257), dim=-1).permute(1, 0, 2, 3)
        )
        
        return output, (h, c)

# create network and training agent
tr_agent = TrainerAE(cfg)

# load from checkpoint if provided
tr_agent.load_ckpt(cfg.ckpt)

model = CADModel()

# model.load_state_dict(torch.load('lstm_model.pth'))


decoder = tr_agent.net.decoder
encoder = tr_agent.net.encoder

count_parameters(decoder)

bottleneck = tr_agent.net.bottleneck

count_parameters(bottleneck)

count_parameters(model)

print("Number of training samples:", n_datasample)

# out = decoder(torch.tensor([[1.0 for i in range(256)]]).to(device))

# print(out[0].shape, out[1].shape)

# print(out[0].argmax(-1), out[1].argmax(-1))
# Define the loss function
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.005)


# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # Adjust step_size and gamma as desired

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
        hidden = (torch.zeros(images.size(0), 512).to(device),  # Initialize hidden state
                  torch.zeros(images.size(0), 512).to(device))  # Initialize cell state
        outputs, hidden = model(images, hidden)
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
            hidden = (torch.zeros(images.size(0), 512).to(device),  # Initialize hidden state
                  torch.zeros(images.size(0), 512).to(device))  # Initialize cell state
            outputs, hidden = model(images, hidden)
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

    torch.save(model.state_dict(), 'lstm_model_'+str(epoch)+'.pth')

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



torch.save(model.state_dict(), 'lstm_model.pth')

np.savetxt('output.txt', outputs[0].argmax(dim = -1).to('cpu').numpy())
np.savetxt('labels.txt', labels_cmd.permute(1, 0).to('cpu').numpy())

