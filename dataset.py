import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import cv2

from multiprocessing import Pool


class UCFCrime(data.Dataset):
    def __init__(self, data_dir, fps, transform, frames):
        self.data_dir = data_dir
        self.fps = fps
        self.transform = transform
        self.frames = frames
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.data = []
        self.labels = []
        for i, cls in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, cls)
            video_name_list = sorted(os.listdir(class_dir))
            for vid_name in video_name_list:
                self.data.append(os.path.join(class_dir, vid_name))
                self.labels.append(i)

        # print(self.data)
        # print(self.labels)

    def _load_video(self, path):
        vidcap = cv2.VideoCapture(path)
        # loader = transforms.Compose([transforms.ToTensor()])
        counter = -1
        frame_idx = -1
        vid = []
        while True:
            success, img = vidcap.read()
            counter += 1
            if not success:
                break
            if counter % (30 // self.fps) != 0:
                continue
            frame_idx += 1
            if frame_idx not in self.frames:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            vid.append(img)

        vid = torch.stack(vid, dim=0)
        return vid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_path = self.data[index]
        # Load data
        X = self._load_video(video_path)
        y = torch.LongTensor([self.labels[index]])

        # print(X.shape)
        d, c, h, w = X.shape
        assert d == len(self.frames)
        X = X.permute(1, 0, 2, 3)
        return X, y


class UCFCrimeMEM(data.Dataset):
    def __init__(self, data_dir, fps, transform, frames, c_fst=True):
        self.data_dir = data_dir
        self.fps = fps
        self.transform = transform
        self.frames = frames
        self.channel_first = c_fst

        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.data = []
        self.labels = []

        for cls in self.classes:
            self._load_class(cls)

    def _load_class(self, cls):
        class_dir = os.path.join(self.data_dir, cls)
        video_name_list = sorted(os.listdir(class_dir))
        for vid_idx, vid_name in enumerate(video_name_list):
            if (vid_idx + 1) / len(video_name_list) > 0.8:
                break

            video_path = os.path.join(class_dir, vid_name)
            self.data.append(self._load_video(video_path))
            self.labels.append(self.class_to_idx[cls])
            if vid_idx % 25 == 0:
                print('Loading: {} [{}/{}]'.format(cls, vid_idx + 1, len(video_name_list)))

    def _load_video(self, path):
        vidcap = cv2.VideoCapture(path)
        counter = -1
        frame_idx = -1
        vid = []
        while True:
            success, img = vidcap.read()
            counter += 1
            if not success:
                break
            if counter % (30 // self.fps) != 0:
                continue
            frame_idx += 1
            if frame_idx not in self.frames:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            vid.append(img)

        vid = torch.stack(vid, dim=0)
        return vid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = torch.LongTensor([self.labels[index]])

        # print(X.shape)
        d, c, h, w = X.shape
        assert d == len(self.frames)
        if self.channel_first:
            X = X.permute(1, 0, 2, 3)
        return X, y


class UCFCrimeVarlenMEM(data.Dataset):
    def __init__(self, data_dir, fps, select_frame, transform):
        self.data_dir = data_dir
        self.fps = fps
        self.select_frame = select_frame
        self.frames = np.arange(self.select_frame['begin'], self.select_frame['end'], self.select_frame['skip'])
        self.transform = transform

        # self.img_size = self.transform.__dict__['transforms'][0].__dict__['size']  # get image resize from Transformation
        # self.num_channels = len(self.transform.__dict__['transforms'][2].__dict__['mean'])  # get number of channels from Transformation

        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.data = []
        self.labels = []

        for cls in self.classes:
            self._load_class(cls)

    def _load_class(self, cls):
        class_dir = os.path.join(self.data_dir, cls)
        video_name_list = sorted(os.listdir(class_dir))
        for vid_idx, vid_name in enumerate(video_name_list):
            if (vid_idx + 1) / len(video_name_list) > 0.7:
                break

            video_path = os.path.join(class_dir, vid_name)
            self.data.append(self._load_video(video_path))
            self.labels.append(self.class_to_idx[cls])
            if vid_idx % 25 == 0:
                print('Loading: {} [{}/{}]'.format(cls, vid_idx + 1, len(video_name_list)))

    def _load_video(self, path):
        vidcap = cv2.VideoCapture(path)
        counter = -1
        frame_idx = -1
        vid = []
        while True:
            success, img = vidcap.read()
            counter += 1
            if not success:
                break
            if counter % (30 // self.fps) != 0:
                continue
            frame_idx += 1

            if frame_idx >= self.select_frame['end']:
                break

            if frame_idx not in self.frames:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            vid.append(img)

        vid = torch.stack(vid, dim=0)
        return vid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        d, c, h, w = X.shape

        X_padded = torch.zeros(len(self.frames), c, h, w)
        X_padded[:d, :, :, :] = X

        video_len = torch.LongTensor([d])
        y = torch.LongTensor([self.labels[index]])

        return X_padded, video_len, y
