import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from solution.logger import Logger
from tqdm import tqdm as progressbar
from sklearn import metrics


class DriverModel(nn.Module):
    def __init__(self, logger_dir, input_size=57, seed=1010101):
        self.logger = Logger(logger_dir)
        self._epoch = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        super().__init__()
        hidden_size = 25
        output_size = 1

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        # self.bn1.double()     can convert full model to double.
        # self.bn2.double()

        nn.init.xavier_normal(self.fc2.weight)
        nn.init.xavier_normal(self.fc1.weight)

    @classmethod
    def to_np(cls, x):
        return x.data.cpu().numpy()

    @classmethod
    def to_var(cls, x, use_gpu=True):
        if torch.cuda.is_available() and use_gpu:
            x = x.cuda()
        return Variable(x)

    @classmethod
    def to_tensor(cls, x):
        # noinspection PyUnresolvedReferences
        tensor = torch.from_numpy(x)
        return tensor

    @classmethod
    def show_env_info(cls):
        print('__Python VERSION:', sys.version)
        print('__CUDA VERSION')
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        print("OS: ", sys.platform)
        print("PyTorch: ", torch.__version__)
        print("Numpy: ", np.__version__)
        use_cuda = torch.cuda.is_available()
        print("CUDA is available", use_cuda)

    def forward(self, x):
        out = self.bn1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def predict(self, input_, use_gpu=True):
        if isinstance(input_, np.array):
            input_ = self.to_tensor(input_)
        input_ = self.to_var(input_, use_gpu=use_gpu)
        predictions = self.__call__(input_)
        predictions - self.to_np(predictions)
        return predictions

    def _log_data(self, data_dict):
        for tag, value in data_dict.items():
            self.logger.scalar_summary(tag, value, self._epoch + 1)

    def _log_grads(self):
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, self.to_np(value), self._epoch + 1)
            self.logger.histo_summary(tag + '/grad', self.to_np(value.grad), self._epoch + 1)

    def _log_images(self, info):
        for tag, images in info.items():
            self.logger.image_summary(tag, images, self._epoch + 1)

    def fit(self, optimizer, loss_fn, data_loader, num_epochs, snapshot_interval, verbose=True):
        data_iter = iter(data_loader)
        iter_per_epoch = len(data_loader)
        for e in progressbar(range(num_epochs)):
            self._epoch = e
            if (e + 1) % iter_per_epoch == 0:
                data_iter = iter(data_loader)       # reset iterator
            next_batch = next(data_iter)            # here we assume data type torch.Tensor
            inputs = next_batch["inputs"]
            labels = next_batch["labels"]
            inputs = inputs.float()
            labels = labels.float()
            inputs, labels = self.to_var(inputs), self.to_var(labels)

            optimizer.zero_grad()
            predictions = self.__call__(inputs)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            classes = (predictions.data > 0.5).float()
            pred_y = classes.cpu().numpy().squeeze()
            target_y = self.to_np(labels).squeeze()
            accuracy = metrics.accuracy_score(target_y, pred_y)
            f1_score = metrics.f1_score(target_y, pred_y)
            roc_auc = metrics.roc_auc_score(target_y, pred_y)
            if (e + 1) % snapshot_interval == 0:
                if verbose:
                    print('Step [%d/%d], Loss: %.4f, Acc: %.2f' % (e + 1, num_epochs, loss.data[0], accuracy))
                info = {'loss': loss.data[0], 'accuracy': accuracy, 'f1': f1_score, "ROC AUC": roc_auc}
                self._log_data(info)
                self._log_grads()

                # im_info = {'images': self.to_np(inputs.view(-1, 28, 28)[:10])}
                # self._log_images(im_info)

    def _delete_logger(self):
        # after this method is called one cannot use self.logger any more.
        del self.logger

    def re_init_logger(self, path):
        self.logger = Logger(path)

    def save(self, path):
        self._delete_logger()
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        if isinstance(model, dict):
            model = model['state_dict']
        return model


if __name__ == "__main__":
    from solution.dataset import DriverDataset, ToTensor
    from torch.utils.data import DataLoader

    transformed_dataset = DriverDataset("../data/for_train.csv", transform=ToTensor())
    dataloader = DataLoader(transformed_dataset, batch_size=128, shuffle=False, num_workers=1)

    net = DriverModel("/tmp")
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.BCELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters())
    net.fit(optim, loss_func, dataloader, 50, 5, verbose=True)
    net.save("model.bin")
