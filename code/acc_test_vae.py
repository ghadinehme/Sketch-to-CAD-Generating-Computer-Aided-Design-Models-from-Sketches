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
from cadlib.macro import EOS_IDX

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

# shuffled_df = pd.read_csv('new_data.csv')
shuffled_df = pd.read_csv('selected_data.csv')
# remove representations with length = 7
shuffled_df = shuffled_df[shuffled_df["rep_len"] > 7]
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
cfg.batch_size = 32
cfg.ckpt = '1000'

l = n_datasample//649
seq_len = 32
batch_size = 128
# Split Data
test_data = shuffled_df[:585*l].reset_index(drop=True)
# val_data = shuffled_df[585*l:617*l].reset_index(drop=True)
# test_data = shuffled_df[617*l:649*l].reset_index(drop=True)
# test_data = shuffled_df[617*l:620*l].reset_index(drop=True)

# Small Dataset to check Overfitting capacity of model
# test_data = shuffled_df[:l].reset_index(drop=True)
# val_data = shuffled_df[l:l+2*l].reset_index(drop=True)
# train_data = shuffled_df[l+2*l:l+5*l].reset_index(drop=True)

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

# traindata = sketchDataset(train_data)
testdata = sketchDataset(test_data)
# valdata = sketchDataset(val_data)


# train_dataloader = DataLoader(traindata, batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(testdata, batch_size, shuffle=True, num_workers=2, pin_memory=True)
# val_dataloader = DataLoader(valdata, batch_size, shuffle=True, num_workers=2, pin_memory=True)



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
        # self.fc_mean = nn.Linear(8192, latent_dim)
        # self.fc_logvar = nn.Linear(8192, latent_dim)



    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, x):
        # Encoder forward pass
        features = self.encoder(x)
        batch_size = features.size(0)
        features = features.view(batch_size, -1)  # Flatten the features
        features = F.relu(features)
        return features, 1, 2
        # m = self.fc_mean(features)
        # h = self.fc_logvar(features)
        # v = F.softplus(h) + 1e-8
        # return features, m, v


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = CNNEncoderVAE(latent_dim)

    def forward(self, x):
        # Encode input to obtain mean, logvar, and latent representation
        z, mean, logvar = self.encoder(x)
        
        return z, mean, logvar  

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
        self.fc = nn.ModuleList([nn.Linear(8192, self.d_model) for _ in range(seq_len)])
        self.decoder_cmd = nn.Linear(d_model, n_cmd)
        self.decoder_params = nn.Linear(d_model, n_params*16, weight_decay)

    def forward(self, z):
        # Transformer encoding and decoding
        # encoded = self.transformer_encoder(z)
        arr = []
        for i in range(seq_len):
            u = F.relu(self.fc[i](z).view(1, -1, self.d_model))
            arr.append(u)
        y = torch.cat(arr, dim=0)
        z = F.relu(self.fc0(z).view(1, -1, self.d_model))
        decoded = self.transformer_decoder(y, z)

        # Class prediction
        # decoded = decoded.permute(1, 0, 2)
        cmd = self.decoder_cmd(decoded)
        params = self.decoder_params(decoded)
        
        cmd = cmd.view(seq_len, -1, self.n_cmd).permute(1,0,2)
        params = params.view(seq_len, -1, 16, self.n_params).permute(1,0,2,3)

        # cmd[:, 0, :] = 0.0
        # cmd[:, 0, 4] = 1.0

        # # Once a 3 appears in cmd, all the remaining elements must be 3
        # for t in range(1, seq_len):
        #     has_3 = (cmd[:,t,3] >= cmd[:,t,:].max(-1).values).float()
        #     cmd[:, t:, :] = cmd[:, t:, :] * (1 - has_3).unsqueeze(-1).unsqueeze(-1).repeat(1, seq_len-t, 6)
        #     cmd[:, t:, 3] = cmd[:, t:, 3] * has_3.unsqueeze(-1).repeat(1, seq_len-t)

        cmd = F.log_softmax(cmd, dim=-1)
        params = F.log_softmax(params, dim=-1)

        # Reshape output to original input shape: [batch_size, sequence_length, 16, num_classes]
        out = (cmd, params)
        # Calculate L1 regularization loss for linear layers
        l1_loss = torch.tensor(0.0).to(z.device)  # Initialize with 0
        l1_loss += torch.norm(self.fc0.weight, p=1)
        l1_loss += torch.norm(self.decoder_cmd.weight, p=1)
        l1_loss += torch.norm(self.decoder_params.weight, p=1)
        for fc_layer in self.fc:
            l1_loss += torch.norm(fc_layer.weight, p=1)  # L1 norm of weights

        return out, l1_loss

# Example usage
model = TransformerAutoencoder(n_cmd = 6, n_params = 257, d_model=512, nhead=32, num_encoder_layers=4, num_decoder_layers=4, weight_decay=0)
model.load_state_dict(torch.load("vae_dec.pth"))
encoder = VAE(8192)
encoder.load_state_dict(torch.load("vae_enc.pth"))

cfg = ConfigAE('test')
tr_agent = TrainerAE(cfg)
test_loss = 0.0
acc_cmd = 0
acc_args = 0
test_total = 0

accuracies = np.array([0]*16)
number = np.array([0]*16)

amazing = []
loss_func = CADLoss(cfg).cuda()

acc_func = CADAccuracy(cfg).cuda()

fully_correct = 0
pbar = tqdm(test_dataloader)
for i, data in enumerate(pbar):
    images, commands, args = data
    batch_size = images.shape[0]
    gt_vec = torch.cat([commands.unsqueeze(-1), args], dim=-1).squeeze(1).detach().cpu().numpy()
    commands_ = gt_vec[:, :, 0]
    with torch.no_grad():
        outputs, _ = model(encoder(images)[0])
        output = {}
        output["command_logits"], output["args_logits"] = outputs
        output["tgt_commands"], output["tgt_args"] = commands, args

        gt_cmds = commands
        cmds = outputs[0].argmax(-1)
        gt_params = args
        params = outputs[1].argmax(-1)
        for l in range(batch_size):
            check = 0
            check_n = 0
            
            for i, el in enumerate(gt_cmds[l]):
                check_n += 1
                if el == cmds[l][i]:
                    check += 1
                if el == 0:
                    check_n +=2
                    for j in [0,1]:
                        number[j] += 1
                        if abs(gt_params[l][i][j] - params[l][i][j])<5:
                            accuracies[j] += 1
                            check += 1
                if el == 1:
                    check_n +=4
                    for j in [0,1,2,3]:
                        number[j] += 1
                        if abs(gt_params[l][i][j] - params[l][i][j])<5:
                            accuracies[j] += 1
                            check += 1
                if el == 2:
                    check_n +=3
                    for j in [0,1,4]:
                        number[j] += 1
                        if abs(gt_params[l][i][j] - params[l][i][j])<5:
                            accuracies[j] += 1
                            check += 1
                if el == 5:
                    check_n +=11 
                    for j in range(5,16):
                        number[j] += 1
                        if abs(gt_params[l][i][j] - params[l][i][j])<5:
                            accuracies[j] += 1
                            check += 1
            if check == check_n:
                fully_correct += 1
                amazing.append([gt_cmds[l].tolist(), gt_params[l].tolist()])



        # Calculate accuracy
        acc_dict = acc_func(output)
        acc_cmd += acc_dict["acc_cmd"]
        acc_args += acc_dict["acc_args"]
        test_total += 1
        

test_loss /= len(test_dataloader)
test_accuracy = acc_cmd / test_total * 100
test_accuracy_1 = acc_args / test_total * 100

print(accuracies/number)
print(number)

print(fully_correct)

def save_list_to_text_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

save_list_to_text_file(amazing, 'cnn_out.txt')


print(f"Test Loss: {test_loss:.4f}, Test Accuracy Cmd: {test_accuracy:.2f}%, Test Accuracy Args: {test_accuracy_1:.2f}%")

