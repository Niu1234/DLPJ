import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D(nn.Module):
    @staticmethod
    def _conv3D_output_size(img_size, padding, kernel_size, stride):
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                    np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
        return outshape

    def __init__(self, t_dim, img_h, img_w, num_classes):
        super(CNN3D, self).__init__()

        self.t_dim = t_dim
        self.img_h = img_h
        self.img_w = img_w
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding
        self.fc_hidden1, self.fc_hidden2 = 256, 128
        self.num_classes = num_classes

        self.conv1_outshape = self._conv3D_output_size((self.t_dim, self.img_h, self.img_w),
                                                       self.pd1,
                                                       self.k1,
                                                       self.s1)
        self.conv2_outshape = self._conv3D_output_size(self.conv1_outshape,
                                                       self.pd2,
                                                       self.k2,
                                                       self.s2)

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2], self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # FC 1, 2 and 3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
