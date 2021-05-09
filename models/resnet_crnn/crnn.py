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


# 2D CNN encoder using ResNet-152 pretrained
class ResNetCNNEncoder(nn.Module):
    def __init__(self, fc1_hs=512, fc2_hs=512, embed_size=256, dp=0.3):
        super(ResNetCNNEncoder, self).__init__()
        self.fc1_hidden_size = fc1_hs
        self.fc2_hidden_size = fc2_hs
        self.cnn_embed_size = embed_size
        self.drop_pr = dp

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc1_hidden_size)
        self.bn1 = nn.BatchNorm1d(self.fc1_hidden_size, momentum=0.01)
        self.fc2 = nn.Linear(self.fc1_hidden_size, self.fc2_hidden_size)
        self.bn2 = nn.BatchNorm1d(self.fc2_hidden_size, momentum=0.01)
        self.fc3 = nn.Linear(self.fc2_hidden_size, self.cnn_embed_size)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_pr, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and batch dim
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  # cnn_embed_seq: shape=(batch, time_step, embed_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, in_size=256, rnn_hs=256, fc_hs=128, num_classes=13, dp=0.3):
        super(DecoderRNN, self).__init__()

        self.rnn_input_size = in_size
        self.rnn_hidden_size = rnn_hs
        self.fc_hidden_size = fc_hs
        self.num_classes = num_classes
        self.drop_pr = dp

        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=3,
            batch_first=True  # input & output will has batch size as 1st dimension. e.g. (batch, time_step, embed_size)
        )

        self.fc1 = nn.Linear(self.rnn_hidden_size, self.fc_hidden_size)
        self.fc2 = nn.Linear(self.fc_hidden_size, self.num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        # None represents zero initial hidden state. rnn_out has shape=(batch, time_step, output_size)
        rnn_out, (h_n, h_c) = self.lstm(x, None)

        # FC layers
        x = self.fc1(rnn_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_pr, training=self.training)
        x = self.fc2(x)

        return x
