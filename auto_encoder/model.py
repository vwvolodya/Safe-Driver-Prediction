import torch
from torch import nn
from tqdm import tqdm as progressbar
from sklearn import metrics
from base.model import BaseModel
from base.logger import Logger
import numpy as np
from collections import defaultdict


class Autoencoder(BaseModel):
    def __init__(self, input_size, hidden_size, encoder_features):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.xavier_normal(self.fc1.weight)

        self.fc2 = nn.Linear(hidden_size, encoder_features, bias=True)
        nn.init.xavier_normal(self.fc2.weight)

        self.fc_02 = nn.Linear(encoder_features, hidden_size, bias=True)
        self.fc_02.weight = self.fc2.weight

        self.fc_01 = nn.Linear(hidden_size, input_size, bias=True)
        self.fc_01.weight = self.fc1.weight

        self.activation = nn.Tanh()
        self._encoder_shape = True

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
        # No activation on last layer.
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

    def evaluate(self, logger, loader, loss_fn=None, switch_to_eval=False, **kwargs):
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

            if loss_fn:
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

            self.evaluate(logger, validation_data_loader, loss_fn=loss_fn, switch_to_eval=True)
            self.save("./models/autoenc_%s.mdl" % e)

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

    train_ds = AutoEncoderDataset("../data/for_train_processed.csv", is_train=True, transform=ToTensor(), top=top)
    validation_ds = AutoEncoderDataset("../data/for_test_processed.csv", is_train=False, top=top, transform=ToTensor())

    dataloader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(validation_ds, batch_size=512, shuffle=False, num_workers=1)

    main_logger = Logger("../logs")

    input_layer = train_ds.num_features
    net = Autoencoder(input_layer, int(input_layer * 1.4), 35)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.MSELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.SGD(net.parameters(), lr=0.00005)
    net.fit(optim, loss_func, dataloader, val_dataloader, 100, logger=main_logger)

