import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from models.cnn3d.toy import CNN3D
from models.resnet_crnn.crnn import ResNetCNNEncoder, DecoderRNN
from dataset import UCFCrimeMEM


class CNN3DClient:
    def __init__(self):
        # set path
        self.data_path = "/content/drive/MyDrive/Data/Datasets/Anomaly-Detection-Dataset-mp4/Anomaly-Videos"

        # training parameters
        self.num_classes = 13
        self.epochs = 32
        self.batch_size = 8
        self.learning_rate = 1e-3
        self.log_interval = 10
        self.img_h = 240
        self.img_w = 320
        self.fps = 10

        self.begin_frame = 0
        self.end_frame = 28
        self.skip_frame = 1

    @staticmethod
    def train(log_interval, model, device, train_loader, optimizer, epoch):
        model.train()

        losses = []
        scores = []

        counter = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device).view(-1, )

            counter += X.size(0)

            optimizer.zero_grad()
            output = model(X)

            loss = F.cross_entropy(output, y)
            losses.append(loss.item())

            # to compute accuracy
            _, y_pred = torch.max(output, 1)
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            scores.append(step_score)

            loss.backward()
            optimizer.step()

            # show information
            if (i + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                    epoch + 1,
                    counter,
                    len(train_loader.dataset),
                    100. * (i + 1) / len(train_loader),
                    loss.item(),
                    100 * step_score))

        return losses, scores

    @staticmethod
    def validation(model, device, optimizer, test_loader):
        model.eval()

        test_loss = 0
        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device).view(-1, )

                output = model(X)

                loss = F.cross_entropy(output, y, reduction='sum')
                test_loss += loss.item()  # sum up batch loss
                _, y_pred = output.max(1, keepdim=True)  # (y_pred != output) get the index of the max log-probability

                # collect all y and y_pred in all batches
                all_y.extend(y)
                all_y_pred.extend(y_pred)

        test_loss /= len(test_loader.dataset)

        # to compute accuracy
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

        # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100 * test_score))

        # # save Pytorch models of best record
        # torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        # torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))
        # print("Epoch {} model saved!".format(epoch + 1))

        return test_loss, test_score

    def run(self):
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

        # image transformation
        transform = transforms.Compose([transforms.Resize([self.img_h, self.img_w]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])

        selected_frames = np.arange(self.begin_frame, self.end_frame, self.skip_frame).tolist()

        cnn3d = CNN3D(t_dim=len(selected_frames), img_h=self.img_h, img_w=self.img_w, num_classes=self.num_classes).to(device)

        dataset = UCFCrimeMEM(data_dir=self.data_path, fps=self.fps, transform=transform, frames=selected_frames, c_fst=True)
        train_set, valid_set = data.random_split(dataset, lengths=[int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
                                                 generator=torch.Generator().manual_seed(1030))

        train_loader = data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = data.DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.Adam(cnn3d.parameters(), lr=self.learning_rate)

        epoch_train_losses = []
        epoch_train_scores = []
        epoch_test_losses = []
        epoch_test_scores = []

        # start training
        for epoch in range(self.epochs):
            train_losses, train_scores = self.train(self.log_interval, cnn3d, device, train_loader, optimizer, epoch)
            epoch_test_loss, epoch_test_score = self.validation(cnn3d, device, optimizer, valid_loader)

            epoch_train_losses.append(train_losses)
            epoch_train_scores.append(train_scores)
            epoch_test_losses.append(epoch_test_loss)
            epoch_test_scores.append(epoch_test_score)


class CRNNClient:
    def __init__(self):
        # set path
        self.data_dir = "/content/drive/MyDrive/Data/Datasets/Anomaly-Detection-Dataset-mp4/Anomaly-Videos"
        self.model_save_dir = "/content/drive/MyDrive/Colab/CS3033256-ML/ResNetCRNN_ckpt"

        # Architecture
        self.cnn_fc1_hs, self.cnn_fc2_hs = 1024, 768
        self.cnn_embed_size = 512
        self.dropout_prob = 0.0
        self.rnn_hs = 512
        self.rnn_fc_hs = 256

        self.res_img_size = 224
        self.fps = 10

        # training parameters
        self.num_classes = 13
        self.epochs = 120
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.log_interval = 10

        # Select which frame to begin & end in videos
        self.begin_frame, self.end_frame, self.skip_frame = 1, 29, 1

    @staticmethod
    def train(log_interval, model, device, train_loader, optimizer, epoch):
        # set model as training mode
        cnn_encoder, rnn_decoder = model
        cnn_encoder.train()
        rnn_decoder.train()

        losses = []
        scores = []
        counter = 0  # counting total trained sample in one epoch
        for batch_idx, (X, y) in enumerate(train_loader):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            counter += X.size(0)

            optimizer.zero_grad()
            output = rnn_decoder(cnn_encoder(X))  # output has dim = (batch, number of classes)

            loss = F.cross_entropy(output, y)
            losses.append(loss.item())

            # to compute accuracy
            y_pred = torch.max(output, 1)[1]  # y_pred != output
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            scores.append(step_score)  # computed on CPU

            loss.backward()
            optimizer.step()

            # show information
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                    epoch + 1, counter, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                    100 * step_score))

        return losses, scores

    @staticmethod
    def validation(model, device, optimizer, test_loader):
        # set model as testing mode
        cnn_encoder, rnn_decoder = model
        cnn_encoder.eval()
        rnn_decoder.eval()

        test_loss = 0
        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for X, y in test_loader:
                # distribute data to device
                X, y = X.to(device), y.to(device).view(-1, )

                output = rnn_decoder(cnn_encoder(X))

                loss = F.cross_entropy(output, y, reduction='sum')
                test_loss += loss.item()  # sum up batch loss
                y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

                # collect all y and y_pred in all batches
                all_y.extend(y)
                all_y_pred.extend(y_pred)

        test_loss /= len(test_loader.dataset)

        # compute accuracy
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

        # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100 * test_score))

        return test_loss, test_score

    def run(self):
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

        # Data loading parameters
        dl_params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        transform = transforms.Compose([transforms.Resize([self.res_img_size, self.res_img_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        selected_frames = np.arange(self.begin_frame, self.end_frame, self.skip_frame).tolist()

        dataset = UCFCrimeMEM(data_dir=self.data_dir, fps=self.fps, transform=transform, frames=selected_frames, c_fst=False)
        train_test_split = int(len(dataset) * 0.8)
        train_set, valid_set = data.random_split(dataset, lengths=[train_test_split, len(dataset) - train_test_split],
                                                 generator=torch.Generator().manual_seed(1030))

        train_loader = data.DataLoader(train_set, **dl_params)
        valid_loader = data.DataLoader(valid_set, **dl_params)

        cnn_encoder = ResNetCNNEncoder(fc1_hs=self.cnn_fc1_hs, fc2_hs=self.cnn_fc2_hs, embed_size=self.cnn_embed_size,
                                       dp=self.dropout_prob).to(device)
        rnn_decoder = DecoderRNN(in_size=self.cnn_embed_size, rnn_hs=self.rnn_hs, fc_hs=self.rnn_fc_hs, num_classes=self.num_classes,
                                 dp=self.dropout_prob).to(device)

        # Combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                      list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                      list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

        optimizer = torch.optim.Adam(crnn_params, lr=self.learning_rate)

        # record training process
        epoch_train_losses = []
        epoch_train_scores = []
        epoch_test_losses = []
        epoch_test_scores = []

        # start training
        for epoch in range(self.epochs):
            # train, test model
            train_losses, train_scores = self.train(self.log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
            epoch_test_loss, epoch_test_score = self.validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

            # save results
            epoch_train_losses.append(train_losses)
            epoch_train_scores.append(train_scores)
            epoch_test_losses.append(epoch_test_loss)
            epoch_test_scores.append(epoch_test_score)

            # save Pytorch models of best record
            torch.save(cnn_encoder.state_dict(),
                       os.path.join(self.model_save_dir, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save cnn
            torch.save(rnn_decoder.state_dict(),
                       os.path.join(self.model_save_dir, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save rnn
            torch.save(optimizer.state_dict(),
                       os.path.join(self.model_save_dir, 'optimizer_epoch{}.pth'.format(epoch + 1)))  # save optimizer
            print("Epoch {} model saved!".format(epoch + 1))

            # # save all train test results
            # A = np.array(epoch_train_losses)
            # B = np.array(epoch_train_scores)
            # C = np.array(epoch_test_losses)
            # D = np.array(epoch_test_scores)
            # np.save('./CRNN_epoch_training_losses.npy', A)
            # np.save('./CRNN_epoch_training_scores.npy', B)
            # np.save('./CRNN_epoch_test_loss.npy', C)
            # np.save('./CRNN_epoch_test_score.npy', D)

        # plot
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(1, self.epochs + 1), np.array(epoch_train_losses)[:, -1])  # train loss (on epoch end)
        plt.plot(np.arange(1, self.epochs + 1), np.array(epoch_test_losses))  # test loss (on epoch end)
        plt.title("model loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc="upper left")
        # 2nd figure
        plt.subplot(122)
        plt.plot(np.arange(1, self.epochs + 1), np.array(epoch_train_scores)[:, -1])  # train accuracy (on epoch end)
        plt.plot(np.arange(1, self.epochs + 1), np.array(epoch_test_scores))  # test accuracy (on epoch end)
        plt.title("training scores")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc="upper left")
        title = os.path.join(self.model_save_dir, "fig_resnetcrnn.png")
        plt.savefig(title, dpi=600)
        # plt.close(fig)
        plt.show()
