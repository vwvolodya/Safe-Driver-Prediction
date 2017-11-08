import torch
from torch import nn
from tqdm import tqdm as progressbar
from sklearn import metrics
from base.model import BaseModel
from base.logger import Logger
import numpy as np
from collections import defaultdict


class Autoencoder(BaseModel):
    def __init__(self, input_size, hidden_size, encoder_features, num_classes=1, model_prefix="", classifier=False):
        super().__init__()
        self.model_prefix = model_prefix
        self.is_classifier = classifier
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(encoder_features, num_classes)
        nn.init.xavier_uniform(self.out.weight, gain=0.01)

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.xavier_uniform(self.fc1.weight, gain=0.01)

        self.fc2 = nn.Linear(hidden_size, encoder_features, bias=True)
        nn.init.xavier_uniform(self.fc2.weight, gain=0.01)

        self.fc_02 = nn.Linear(encoder_features, hidden_size, bias=False)
        self.fc_02.weight = self.fc2.weight

        self.fc_01 = nn.Linear(hidden_size, input_size, bias=False)
        self.fc_01.weight = self.fc1.weight

        self.activation = nn.Tanh()
        self._encoder_shape = True

    def classifier(self, x):
        # x here is the result of encoder's output
        probs = self.out(x)
        probs = self.sigmoid(probs)
        return probs

    def encoder(self, x):
        if self._encoder_shape is False:
            self.fc1.weight.data.t_()
            self.fc2.weight.data.t_()
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        self._encoder_shape = True
        return out

    def decoder(self, x):
        if self._encoder_shape is True:
            self.fc_01.weight.data.t_()
            self.fc_02.weight.data.t_()
        out = self.fc_02(x)
        out = self.activation(out)
        out = self.fc_01(out)
        out = self.activation(out)
        self._encoder_shape = False
        return out

    def forward(self, x):
        y = self.encoder(x)
        if self.is_classifier is True:
            y = self.classifier(y)
        else:
            y = self.decoder(y)
        return y

    @classmethod
    def _get_classes(cls, predictions):
        classes = (predictions.data > 0.5).float()
        pred_y = classes.cpu().numpy().squeeze()
        return pred_y

    def predict(self, x, return_classes=False, **kwargs):
        predictions = self.__call__(x)
        classes = None
        if self.is_classifier and return_classes:
            classes = self._get_classes(predictions)
        return predictions, classes

    def _compute_metrics_clf(self, target_y, pred_y, predictions_are_classes=True, training=True):
        prefix = "val_" if not training else ""
        if predictions_are_classes:
            recall = metrics.recall_score(target_y, pred_y, pos_label=1.0)
            precision = metrics.precision_score(target_y, pred_y, pos_label=1.0)
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

    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        return {}

    def evaluate_clf(self, logger, loader, loss_fn=None, switch_to_eval=False, **kwargs):
        # aggregate results from training epoch.
        train_losses = self._predictions.pop("train_loss")
        train_loss = sum(train_losses) / len(train_losses)
        train_metrics_1 = self._compute_metrics_clf(self._predictions["target"], self._predictions["predicted"])
        train_metrics_2 = self._compute_metrics_clf(self._predictions["target"], self._predictions["probs"],
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
        computed_metrics = self._compute_metrics_clf(all_targets, all_predictions, training=False)
        computed_metrics_1 = self._compute_metrics_clf(all_targets, all_probs, training=False,
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

    def evaluate(self, logger, loader, loss_fn, switch_to_eval=False, **kwargs):
        # aggregate results from training epoch.
        train_losses = self._predictions.pop("train_loss")
        train_loss = sum(train_losses) / len(train_losses)
        train_metrics = {"train_loss": train_loss}

        if switch_to_eval:
            self.eval()
        iterator = iter(loader)
        iter_per_epoch = len(loader)

        losses = []
        for i in range(iter_per_epoch):
            inputs, targets = self._get_inputs(iterator)
            probs, _ = self.predict(inputs)

            loss = loss_fn(probs, targets)
            losses.append(loss.data[0])

        val_loss = sum(losses) / len(losses)
        computed_metrics = {"val_loss": val_loss}
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

                self._accumulate_results(self.to_np(targets).squeeze(),
                                         classes,
                                         loss=loss.data[0],
                                         probs=self.to_np(predictions).squeeze())

            if self.is_classifier:
                self.evaluate_clf(logger, validation_data_loader, loss_fn, switch_to_eval=True)
                self.save("./models/clf_%s.mdl" % str(e + 1))
            else:
                self.evaluate(logger, validation_data_loader, loss_fn, switch_to_eval=True)
                self.save("./models/%sautoenc_%s.mdl" % (self.model_prefix, str(e + 1)))

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, targets = next_batch["inputs"], next_batch["targets"]
        inputs, targets = cls.to_var(inputs), cls.to_var(targets)
        return inputs, targets


if __name__ == "__main__":
    from auto_encoder.dataset import AutoEncoderDataset, ToTensor
    from torch.utils.data import DataLoader
    top = 100
    val_top = 100
    train_batch_size = 10
    test_batch_size = 10
    is_classifier = True

    main_logger = Logger("../logs")

    if is_classifier:
        train_ds = AutoEncoderDataset("../data/for_train.csv", is_train=True, transform=ToTensor(), top=top,
                                      for_classifier=True, augment=10, duplicate_numeric_features=True)
        val_ds = AutoEncoderDataset("../data/for_test.csv", is_train=True, top=val_top, transform=ToTensor(),
                                    for_classifier=True, duplicate_numeric_features=True)
        net = Autoencoder.load("./models/autoenc_64.mdl")
        net.is_classifier = True
        loss_func = torch.nn.BCELoss()
    else:
        train_ds = AutoEncoderDataset("../data/one-hot-train.csv", is_train=True, transform=ToTensor(), top=top,
                                      noise_rate=0.5, remove_positive=True, duplicate_numeric_features=True)
        val_ds = AutoEncoderDataset("../data/train_pos.csv", is_train=False, top=val_top, transform=ToTensor(),
                                    remove_positive=False, noise_rate=None, duplicate_numeric_features=True)

        input_layer = train_ds.num_features
        net = Autoencoder(input_layer, int(input_layer * 1.3), 25)

        loss_func = torch.nn.MSELoss()

    net.show_env_info()
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=False, num_workers=6)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False, num_workers=6)

    if torch.cuda.is_available():
        net.cuda()
        loss_func.cuda()

    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.fit(optim, loss_func, train_loader, val_loader, 250, logger=main_logger)
