
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

class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)
  
class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, d_model, seq_len):
        super().__init__()

        self.command_embed = nn.Embedding(6, d_model)

        args_dim = 256 + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * 16, d_model)

        self.pos_encoding = PositionalEncodingLUT(d_model, max_len=seq_len+2)

    def forward(self, commands, args):
        S, N = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL

        src = self.pos_encoding(src)

        return src


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_cmd, n_params, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerAutoencoder, self).__init__()

        self.n_cmd = n_cmd
        self.n_params = n_params
        self.d_model = d_model

        self.embedding = CADEmbedding(d_model, 128)

        # Transformer
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_decoder_layers)

        # Linear layer to get class probabilities
        self.decoder_cmd = nn.Linear(d_model, n_cmd)
        self.decoder_params = nn.Linear(d_model, n_params*16)


    def forward(self, x, y):
        # Reshape and embed input: [batch_size, sequence_length*16, num_classes]
        x = self.embedding(x, y)

        # Transformer encoding and decoding
        encoded = self.transformer_encoder(x)
        decoded = self.transformer_decoder(encoded, encoded)

        # Class prediction
        cmd = self.decoder_cmd(decoded)
        params = self.decoder_params(decoded)
        # Reshape output to original input shape: [batch_size, sequence_length, 16, num_classes]
        out = (cmd.view(-1, 60, self.n_cmd), params.view(-1, 60, 16, self.n_params))
        return out

