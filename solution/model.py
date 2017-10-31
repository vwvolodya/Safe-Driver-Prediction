import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from base.logger import Logger
from base.model import BaseModel
from tqdm import tqdm as progressbar
from sklearn import metrics


class DriverClassifier(BaseModel):
    def __init__(self, layer_sizes: list, num_classes: int, seed=1010101):
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

    def _compute_metrics(self, target_y, pred_y, loss, save=True):
        accuracy = metrics.accuracy_score(target_y, pred_y)
        f1_score = metrics.f1_score(target_y, pred_y)
        recall = metrics.recall_score(target_y, pred_y)
        precision = metrics.precision_score(target_y, pred_y)
        roc_auc = None
        try:
            roc_auc = metrics.roc_auc_score(target_y, pred_y)
        except ValueError:
            pass
        else:
            self._metrics["roc"].append(roc_auc)
        if save:
            self._metrics["f1"].append(f1_score)
            self._metrics["pres"].append(precision)
            self._metrics["recall"].append(recall)
            self._metrics["acc"].append(accuracy)
            self._metrics["loss"].append(loss)
        return accuracy, f1_score, roc_auc, precision, recall

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

    def fit(self, optimizer, loss_fn, data_loader, validation_data_loader, num_epochs, logger, verbose=True):
        iter_per_epoch = len(data_loader)
        for e in progressbar(range(num_epochs)):
            self._epoch = e
            data_iter = iter(data_loader)
            accuracy = None
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
            self.save("models/model_%s.mdl" % e)


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
