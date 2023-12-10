import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask, _get_visibility_mask
from cadlib.macro import CMD_ARGS_MASK


class CADAccuracy(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]
        loss_cmd = (abs(command_logits[padding_mask.bool()].reshape(-1, self.n_commands).argmax(dim = -1) - tgt_commands[padding_mask.bool()].reshape(-1).long())<1).float().sum()
        loss_args = (abs(args_logits[mask.bool()].reshape(-1, self.args_dim).argmax(dim = -1) - (tgt_args[mask.bool()].reshape(-1).long() + 1))<3).float().sum()

        res = {"acc_cmd": loss_cmd/(tgt_commands[padding_mask.bool()].reshape(-1).long().shape[0]), "acc_args": loss_args/(tgt_args[mask.bool()].reshape(-1).long() + 1).shape[0]}
        return res