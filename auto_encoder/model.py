import torch
from torch import nn
from tqdm import tqdm as progressbar
from sklearn import metrics
from base.model import BaseModel
from base.logger import Logger
import numpy as np
from collections import defaultdict


class Autoencoder(BaseModel):
    def __init__(self, input_size, hidden_size, encoder_features, model_prefix="", use_batch_norm=False):
        super().__init__()
        self._model_prefix = model_prefix
        # gain = nn.init.calculate_gain("tanh")
        self._use_batch_norm = use_batch_norm
        if use_batch_norm:
            print("Will use BatchNorm for encoder.")
            self.bn = nn.BatchNorm1d(input_size)

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

    def encoder(self, x):
        if self._encoder_shape is False:
            self.fc1.weight.data.t_()
            self.fc2.weight.data.t_()
        if self._use_batch_norm:
            x = self.bn(x)
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
        y = self.decoder(y)
        return y

    def predict_encoder(self, x):
        y = self.encoder(x)
        return y

    def predict(self, x, **kwargs):
        predictions = self.__call__(x)
        return predictions

    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        return {}

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
            probs = self.predict(inputs)

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

                predictions = self.predict(inputs)

                optim.zero_grad()
                loss = loss_fn(predictions, targets)
                loss.backward()
                optim.step()

                self._accumulate_results(targets, predictions, loss=loss.data[0])

            self.evaluate(logger, validation_data_loader, loss_fn, switch_to_eval=True)
            self.save("./models/%sautoenc_%s.mdl" % (self._model_prefix, e + 1))

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, targets = next_batch["inputs"], next_batch["targets"]
        inputs, targets = cls.to_var(inputs), cls.to_var(targets)
        return inputs, targets


if __name__ == "__main__":
    from auto_encoder.dataset import AutoEncoderDataset, ToTensor
    from torch.utils.data import DataLoader
    top = None
    val_top = None
    train_batch_size = 8192
    test_batch_size = 4096

    train_ds = AutoEncoderDataset("../data/one-hot-train.csv", is_train=True, transform=ToTensor(), top=top,
                                  noise_rate=0.2, use_categorical=False)
    val_ds = AutoEncoderDataset("../data/train_pos.csv", is_train=False, top=val_top, transform=ToTensor(),
                                remove_positive=False, use_categorical=False)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=False, num_workers=6)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False, num_workers=6)

    main_logger = Logger("../logs")

    input_layer = train_ds.num_features
    net = Autoencoder(input_layer, int(input_layer * 1.4), 10, model_prefix="numeric_", use_batch_norm=True)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.SmoothL1Loss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.fit(optim, loss_func, train_loader, val_loader, 100, logger=main_logger)
