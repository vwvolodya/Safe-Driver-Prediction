import sys
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable
from solution.logger import Logger
from tqdm import tqdm as progressbar
from sklearn import metrics


class DriverClassifier(nn.Module):
    def __init__(self, layer_sizes: list, num_classes: int, seed=1010101):

        self._metrics = defaultdict(list)
        self._epoch = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.layers = []
        for i in range(len(layer_sizes) - 1):       # last layer has different activation
            print("layer_size", layer_sizes[i], layer_sizes[i + 1])
            bn = nn.BatchNorm1d(layer_sizes[i])
            fc = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False)
            setattr(self, "bn_%s" % i, bn)
            setattr(self, "fc_%s" % i, fc)
            bn.cuda()
            fc.cuda()
            nn.init.xavier_normal(fc.weight, gain=0.005)
            self.layers.append((bn, fc))

        self.last = nn.Linear(layer_sizes[-1], num_classes)
        nn.init.xavier_normal(self.last.weight, gain=0.005)

        # self.bn1.double()     can convert full model to double.
        # self.bn2.double()

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

    def _avg_metrics(self):
        res = {}
        for k, v in self._metrics.items():
            res[k] = sum(v) / len(v)
        return res

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
        for i, el in enumerate(self.layers):
            bn, fc = el
            if i == 0:
                out = bn(x)
            else:
                out = bn(out)
            out = fc(out)
            out = self.relu(out)
        out = self.last(out)
        out = self.sigmoid(out)
        return out

    def predict(self, input_, use_gpu=True, return_classes=True):
        if isinstance(input_, np.ndarray):
            input_ = self.to_tensor(input_)
        if not isinstance(input_, Variable):
            input_ = self.to_var(input_, use_gpu=use_gpu)
        predictions = self.__call__(input_)
        if not return_classes:
            return predictions
        pred_y = self._get_classes(predictions)
        return pred_y

    def _log_data(self, logger, data_dict):
        for tag, value in data_dict.items():
            logger.scalar_summary(tag, value, self._epoch + 1)

    def _log_grads(self, logger):
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, self.to_np(value), self._epoch + 1)
            logger.histo_summary(tag + '/grad', self.to_np(value.grad), self._epoch + 1)

    def _compute_metrics(self, target_y, pred_y, loss, save=True):
        accuracy = metrics.accuracy_score(target_y, pred_y)
        f1_score = metrics.f1_score(target_y, pred_y)
        recall = metrics.recall_score(target_y, pred_y)
        presicion = metrics.precision_score(target_y, pred_y)
        roc_auc = None
        try:
            roc_auc = metrics.roc_auc_score(target_y, pred_y)
        except ValueError:
            pass
        else:
            self._metrics["roc"].append(roc_auc)
        if save:
            self._metrics["f1"].append(f1_score)
            self._metrics["pres"].append(presicion)
            self._metrics["recall"].append(recall)
            self._metrics["acc"].append(accuracy)
            self._metrics["loss"].append(loss)
        return accuracy, f1_score, roc_auc, presicion, recall

    def _log_and_reset(self, logger, data=None):
        if not data:
            averaged = self._avg_metrics()
            self._log_data(logger, averaged)
            self._log_grads(logger)
            self._metrics = defaultdict(list)
        else:
            self._log_data(logger, data)

    @classmethod
    def _get_classes(cls, predictions):
        classes = (predictions.data > 0.5).float()
        pred_y = classes.cpu().numpy().squeeze()
        return pred_y

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, labels = next_batch["inputs"], next_batch["labels"]
        inputs, labels = inputs.float(), labels.float()
        inputs, labels = cls.to_var(inputs), cls.to_var(labels)
        return inputs, labels

    def evaluate(self, loader):
        iterator = iter(loader)
        iter_per_epoch = len(loader)
        all_predictions = np.array([])
        all_labels = np.array([])
        for i in range(iter_per_epoch):
            inputs, labels = self._get_inputs(iterator)
            pred_y = self.predict(inputs)
            target_y = self.to_np(labels).squeeze()

            all_labels = np.append(all_labels, target_y)
            all_predictions = np.append(all_predictions, pred_y)
        accuracy, f1_score, roc_auc, pres, rec = self._compute_metrics(all_labels, all_predictions, None, save=False)
        return accuracy, f1_score, roc_auc, pres, rec

    def fit(self, optimizer, loss_fn, data_loader, validation_data_loader, num_epochs, logger, verbose=True):
        iter_per_epoch = len(data_loader)
        for e in progressbar(range(num_epochs)):
            self._epoch = e
            data_iter = iter(data_loader)
            for i in range(iter_per_epoch):
                inputs, labels = self._get_inputs(data_iter)

                optimizer.zero_grad()
                predictions = self.predict(inputs, return_classes=False)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()

                pred_y = self._get_classes(predictions)
                target_y = self.to_np(labels).squeeze()

                accuracy, f1_score, roc_auc, pres, rec = self._compute_metrics(target_y, pred_y, loss.data[0])

            val_acc, val_f1, val_roc, val_pres, val_rec = self.evaluate(validation_data_loader)
            if verbose:
                print('Step [%d/%d], Loss: %.4f, Acc: %.2f' % (e + 1, num_epochs, loss.data[0], accuracy))
            self._log_and_reset(logger)
            self._log_and_reset(logger, data={"val/acc": val_acc, "val/f1": val_f1, "val/roc": val_roc,
                                              "val/pres": val_pres, "val/recall": val_rec})
            self.save("../models/model_%s.mdl" % e)

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        """
        Load model
        :param path: string path to file containing model
        :return: instance of this class
        :rtype: DriverClassifier
        """
        model = torch.load(path)
        if isinstance(model, dict):
            model = model['state_dict']
        return model


if __name__ == "__main__":
    from solution.dataset import DriverDataset, ToTensor
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    transformed_dataset = DriverDataset("../data/for_train.csv", scaler=scaler, is_train=True, transform=ToTensor())
    validation_dataset = DriverDataset("../data/for_validation.csv", scaler=scaler, is_train=False, transform=ToTensor())

    dataloader = DataLoader(transformed_dataset, batch_size=32768, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False, num_workers=6)

    main_logger = Logger("../logs")

    input_layer = transformed_dataset.num_features
    net = DriverClassifier([input_layer, input_layer // 3, input_layer // 4, input_layer // 5], 1)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.BCELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.003)
    net.fit(optim, loss_func, dataloader, val_dataloader, 2500, logger=main_logger, verbose=False)
    net.eval()
