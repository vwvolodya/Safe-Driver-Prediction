import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from base.logger import Logger
from base.model import BaseModel
from tqdm import tqdm as progressbar
from sklearn import metrics
from collections import defaultdict


class DriverClassifier(BaseModel):
    def __init__(self, layer_sizes: list, num_classes: int, seed=1010101, gain=1):
        super().__init__(seed=seed)

        self.selu = nn.SELU(inplace=True)
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
            nn.init.xavier_normal(fc.weight, gain=gain)
            self.layers.append((bn, fc))

        self.last = nn.Linear(layer_sizes[-1], num_classes)
        nn.init.xavier_normal(self.last.weight, gain=gain)

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
            out = self.selu(out)
        out = self.last(out)
        out = self.sigmoid(out)
        return out

    def predict(self, input_, use_gpu=True, return_classes=True):
        if isinstance(input_, np.ndarray):
            input_ = self.to_tensor(input_)
        if not isinstance(input_, Variable):
            input_ = self.to_var(input_, use_gpu=use_gpu)

        predictions = self.__call__(input_)
        classes = None
        if return_classes:
            classes = self._get_classes(predictions)
        return predictions, classes

    @classmethod
    def _gini(cls, actual, pred, cmpcol=0, sortcol=1):
        assert (len(actual) == len(pred))
        all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses

        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)

    @classmethod
    def _gini_normalized(cls, a, p):
        return cls._gini(a, p) / cls._gini(a, a)

    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        prefix = "val_" if not training else ""
        if predictions_are_classes:
            f1_score = metrics.f1_score(target_y, pred_y, pos_label=1.0)
            recall = metrics.recall_score(target_y, pred_y, pos_label=1.0)
            precision = metrics.precision_score(target_y, pred_y, pos_label=1.0)
            gini = self._gini_normalized(target_y, pred_y)
            result = {"f1": f1_score, "precision": precision, "recall": recall, "gini": gini}
        else:
            fpr, tpr, thresholds = metrics.roc_curve(target_y, pred_y, pos_label=1.0)
            auc = metrics.auc(fpr, tpr)
            custom_gini = 2 * auc - 1
            result = {"auc": auc, "custom_gini": custom_gini}

        final = {}
        for k, v in result.items():
            final[prefix + k] = v
        return final

    @classmethod
    def _get_classes(cls, predictions):
        classes = (predictions.data > 0.5).float()
        pred_y = classes.cpu().numpy().squeeze()
        return pred_y

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, labels = next_batch["inputs"], next_batch["targets"]
        inputs, labels = cls.to_var(inputs), cls.to_var(labels)
        return inputs, labels

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

    def fit(self, optimizer, loss_fn, data_loader, validation_data_loader, num_epochs, logger):
        iter_per_epoch = len(data_loader)
        for e in progressbar(range(num_epochs)):
            self._epoch = e
            data_iter = iter(data_loader)

            for i in range(iter_per_epoch):
                inputs, labels = self._get_inputs(data_iter)

                optimizer.zero_grad()
                predictions, pred_y = self.predict(inputs, return_classes=True)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()

                # pred_y = self._get_classes(predictions)
                probs = self.to_np(predictions).squeeze()
                target_y = self.to_np(labels).squeeze()

                self._accumulate_results(target_y, pred_y, loss=loss.data[0], probs=probs)

            self.evaluate(logger, validation_data_loader, loss_fn=loss_fn, switch_to_eval=True)
            self.save("models/model_%s.mdl" % e)


if __name__ == "__main__":
    from solution.dataset import DriverDataset, ToTensor
    from torch.utils.data import DataLoader

    top = None

    transformed_dataset = DriverDataset("../data/for_train.csv", is_train=True, transform=ToTensor(), top=top)
    validation_dataset = DriverDataset("../data/for_validation.csv", is_train=False, transform=ToTensor(), top=top)

    dataloader = DataLoader(transformed_dataset, batch_size=4096, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(validation_dataset, batch_size=2048, shuffle=False, num_workers=6)

    main_logger = Logger("../logs")

    input_layer = transformed_dataset.num_features
    net = DriverClassifier([input_layer, 50, 25, 10, 5], 1)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.BCELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.0003)
    net.fit(optim, loss_func, dataloader, val_dataloader, 50, logger=main_logger)
    print()
