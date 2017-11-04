import torch
from torch import nn
from tqdm import tqdm as progressbar
from sklearn import metrics
import numpy as np
from collections import defaultdict
from base.model import BaseModel
from base.logger import Logger

AUTOENCODER = None


def get_input(data):
    cat_input = AUTOENCODER.to_var(data)
    categorical_x = AUTOENCODER.predict_encoder(cat_input)
    categorical_x = AUTOENCODER.to_np(categorical_x).squeeze()
    # x = np.concatenate((numeric_x, categorical_x))
    categorical_x = AUTOENCODER.to_tensor(categorical_x)
    return categorical_x


class FinalModel(BaseModel):
    def __init__(self, layer_sizes, num_classes):
        super().__init__()

        self.activation = nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        gain = nn.init.calculate_gain("tanh")

        self.layers = []
        for i in range(len(layer_sizes) - 1):  # last layer has different activation
            print("layer_size", layer_sizes[i], layer_sizes[i + 1])
            bn = nn.BatchNorm1d(layer_sizes[i])
            fc = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False)
            setattr(self, "bn_%s" % i, bn)
            setattr(self, "fc_%s" % i, fc)
            bn.cuda()
            fc.cuda()
            nn.init.xavier_normal(fc.weight, gain=gain)
            self.layers.append((bn, fc))

        self.last = nn.Linear(layer_sizes[-1], num_classes)
        nn.init.xavier_normal(self.last.weight, gain=gain)

    def forward(self, x):
        for i, el in enumerate(self.layers):
            bn, fc = el
            if i == 0:
                out = bn(x)
            else:
                out = bn(out)
            out = fc(out)
            out = self.activation(out)
        out = self.last(out)
        out = self.sigmoid(out)
        return out

    def predict(self, x, return_classes=False):
        predictions = self.__call__(x)
        classes = None
        if return_classes:
            classes = torch.round(predictions)
            classes = self.to_np(classes).squeeze()
        return predictions, classes

    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        prefix = "val_" if not training else ""
        if predictions_are_classes:
            recall = metrics.recall_score(target_y, pred_y, pos_label=1.0)
            precision = metrics.precision_score(target_y, pred_y, pos_label=1.0)
            # gini = self._gini_normalized(target_y, pred_y)
            result = {"precision": precision, "recall": recall}
        else:
            fpr, tpr, thresholds = metrics.roc_curve(target_y, pred_y, pos_label=1.0)
            auc = metrics.auc(fpr, tpr)
            custom_gini = 2 * auc - 1
            result = {"auc": auc, "custom_gini": custom_gini}

        final = {}
        for k, v in result.items():
            final[prefix + k] = v
        return final

    def evaluate(self, logger, loader, loss_fn=None, switch_to_eval=False, **kwargs):
        # aggregate results from training epoch.
        train_losses = self._predictions.pop("train_loss")
        train_loss = sum(train_losses) / len(train_losses)
        train_metrics_1 = self._compute_metrics(self._predictions["target"], self._predictions["predicted"])
        train_metrics_2 = self._compute_metrics(self._predictions["target"], self._predictions["probs"],
                                                predictions_are_classes=False)
        train_metrics = {"train_loss": train_loss}
        train_metrics.update(train_metrics_1)
        train_metrics.update(train_metrics_2)

        if switch_to_eval:
            self.eval()
        iterator = iter(loader)
        iter_per_epoch = len(loader)
        all_predictions = np.array([])
        all_targets = np.array([])
        all_probs = np.array([])
        losses = []
        for i in range(iter_per_epoch):
            inputs, targets = self._get_inputs(iterator)
            probs, classes = self.predict(inputs, return_classes=True)
            target_y = self.to_np(targets).squeeze()
            if loss_fn:
                loss = loss_fn(probs, targets)
                losses.append(loss.data[0])
            probs = self.to_np(probs).squeeze()
            all_targets = np.append(all_targets, target_y)
            all_probs = np.append(all_probs, probs)
            all_predictions = np.append(all_predictions, classes)
        computed_metrics = self._compute_metrics(all_targets, all_predictions, training=False)
        computed_metrics_1 = self._compute_metrics(all_targets, all_probs, training=False,
                                                   predictions_are_classes=False)

        val_loss = sum(losses) / len(losses)
        computed_metrics.update({"val_loss": val_loss})
        computed_metrics.update(computed_metrics_1)
        if switch_to_eval:
            # switch back to train
            self.train()

        self._log_and_reset(logger, data=train_metrics, log_grads=True)
        self._log_and_reset(logger, data=computed_metrics, log_grads=False)

        self._predictions = defaultdict(list)
        return computed_metrics

    def fit(self, optim, loss_fn, data_loader, validation_data_loader, num_epochs, logger):
        for e in progressbar(range(num_epochs)):
            iter_per_epoch = len(data_loader)
            self._epoch = e
            data_iter = iter(data_loader)
            for i in range(iter_per_epoch):
                inputs, targets = self._get_inputs(data_iter)

                predictions, classes = self.predict(inputs, return_classes=True)

                optim.zero_grad()
                loss = loss_fn(predictions, targets)
                loss.backward()
                optim.step()

                probs = self.to_np(predictions).squeeze()
                target_y = self.to_np(targets).squeeze()

                self._accumulate_results(target_y, classes, loss=loss.data[0], probs=probs)

            self.evaluate(logger, validation_data_loader, loss_fn=loss_fn, return_classes=True)
            self.save("./models/final_%s.mdl" % e)

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, targets = next_batch["inputs"], next_batch["targets"]
        categorical = get_input(next_batch["categorical"])
        inputs = torch.cat((inputs, categorical), 1)

        inputs, targets = cls.to_var(inputs), cls.to_var(targets)
        return inputs, targets


if __name__ == "__main__":
    from auto_encoder.final_dataset import FinalDataset, ToTensor
    from auto_encoder.model import Autoencoder
    from torch.utils.data import DataLoader

    top = None
    augment = 5
    train_batch_size = 8192
    test_batch_size = 2048
    AUTOENCODER = Autoencoder.load("ready/autoenc_22.mdl")

    train_ds = FinalDataset("../data/for_train.csv", is_train=True, transform=ToTensor(), top=top, augment=augment)
    val_ds = FinalDataset("../data/for_test.csv", is_train=False, top=top, transform=ToTensor())

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False, num_workers=6)

    main_logger = Logger("../logs")

    net = FinalModel([36, 20, 10], 1)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.BCELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.fit(optim, loss_func, train_loader, val_loader, 150, logger=main_logger)

