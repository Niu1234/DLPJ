import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


class DecoderRNNVarlen(nn.Module):
    def __init__(self, in_size=300, rnn_nl=2, rnn_hs=256, fc_hs=128, dp=0.3, num_classes=50):
        super(DecoderRNNVarlen, self).__init__()
        self.rnn_input_size = in_size
        self.rnn_num_layers = rnn_nl
        self.rnn_hidden_size = rnn_hs
        self.fc_hidden_size = fc_hs
        self.num_classes = num_classes
        self.drop_pr = dp

        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,  # input & output will has batch size as 1st dimension. e.g. (batch, time_step, embed_size),
            bidirectional=True
        )

        self.fc1 = nn.Linear(self.rnn_hidden_size * 2, self.fc_hidden_size)
        self.fc2 = nn.Linear(self.fc_hidden_size, self.num_classes)

    def forward(self, x_rnn, x_lengths):
        N, T, n = x_rnn.size()
        for i in range(N):
            if x_lengths[i] < T:
                x_rnn[i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_rnn.device)

        x_lengths[x_lengths > T] = T
        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        # use input of descending length
        packed_x_rnn = torch.nn.utils.rnn.pack_padded_sequence(x_rnn[perm_idx], lengths_ordered.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        packed_rnn_out, (h_n_sorted, h_c_sorted) = self.lstm(packed_x_rnn, None)

        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rnn_out, batch_first=True)
        rnn_out = rnn_out.contiguous()

        # reverse back to original sequence order
        _, unperm_idx = perm_idx.sort(0)
        rnn_out = rnn_out[unperm_idx]

        # FC layers
        x = self.fc1(rnn_out[:, -1, :])  # choose rnn_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_pr, training=self.training)
        x = self.fc2(x)

        return x
