import torch
from torch import nn
from tqdm import tqdm as progressbar
from sklearn import metrics
import numpy as np
from collections import defaultdict
from base.model import BaseModel
from base.logger import Logger


class FinalModel(BaseModel):
    def __init__(self, input_features, hidden, output):
        super().__init__()

        self._tanh = nn.Tanh()
        self._sigmoid = nn.Sigmoid()
        fc1 = nn.Linear(input_features, hidden, bias=True)
        nn.init.xavier_normal(fc1.weight)
        fc2 = nn.Linear(hidden, output)
        nn.init.xavier_normal(fc2.weight)
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        y = self.fc1(x)
        y = self._tanh(y)
        y = self.fc2(y)
        y = self._sigmoid(y)
        return y

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
        inputs, targets = cls.to_var(inputs), cls.to_var(targets)
        return inputs, targets


if __name__ == "__main__":
    from auto_encoder.final_dataset import FinalDataset, ToTensor
    from torch.utils.data import DataLoader
    top = None
    train_batch_size = 4096
    test_batch_size = 2048

    train_dataset = FinalDataset("train.npy", is_train=True, transform=ToTensor(), top=top)
    validation_dataset = FinalDataset("test.npy", is_train=False, top=top, transform=ToTensor())

    dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(validation_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    main_logger = Logger("../logs")

    net = FinalModel(35, 15, 1)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.BCELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.fit(optim, loss_func, dataloader, val_dataloader, 150, logger=main_logger)

