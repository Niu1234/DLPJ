import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import accuracy_score

from models.cnn3d.toy import CNN3D
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
