import torch
from torch import nn
from tqdm import tqdm as progressbar
from sklearn import metrics
from base.model import BaseModel
from base.logger import Logger


class Autoencoder(BaseModel):
    def __init__(self, layer_sizes, encoder_features):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):  # last layer has different activation
            print("layer_size", layer_sizes[i], layer_sizes[i + 1])
            fc = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True)
            nn.init.xavier_normal(fc.weight)
            layers.append(fc)
            layers.append(nn.Tanh())

        print("Last layer_size", layer_sizes[-1], encoder_features)
        last = nn.Linear(layer_sizes[-1], encoder_features)
        nn.init.xavier_normal(last.weight)
        layers.append(last)
        layers.append(nn.Tanh())

        decoder_layers = []
        print("Reversed first layer", encoder_features, layer_sizes[-1])
        first = nn.Linear(encoder_features, layer_sizes[-1])
        nn.init.xavier_normal(first.weight)
        decoder_layers.append(first)
        decoder_layers.append(nn.Tanh())
        for i in range(1, len(layer_sizes)):
            print("reverse layer sizes", layer_sizes[-i], layer_sizes[-(i + 1)])
            layer = nn.Linear(layer_sizes[-i], layer_sizes[-(i+1)])
            nn.init.xavier_normal(layer.weight)
            decoder_layers.append(layer)
            if i != len(layer_sizes) - 1:
                decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        predictions = self.__call__(x)
        return predictions, None

    def fit(self, optim, loss_fn, data_loader, validation_data_loader, num_epochs, logger, verbose=True):
        for e in progressbar(range(num_epochs)):
            iter_per_epoch = len(data_loader)
            self._epoch = e
            data_iter = iter(data_loader)
            for i in range(iter_per_epoch):
                inputs, targets = self._get_inputs(data_iter)

                predictions, _ = self.predict(inputs)

                optim.zero_grad()
                loss = loss_fn(predictions, targets)
                loss.backward()
                optim.step()

            averaged_losses = self.evaluate(validation_data_loader, loss_fn=loss_fn)
            self._log_and_reset(logger)
            self._log_and_reset(logger, data={"loss": loss.data[0], "val_loss": averaged_losses,
                                              # "f1": f1_score, "precision": precision, "recall": recall
                                              })
            self.save("./models/autoenc_%s.mdl" % e)

    def _compute_metrics(self, target_y, pred_y, losses, save=True):
        averaged_losses = sum(losses) / len(losses)
        # f1_score = metrics.f1_score(target_y, pred_y)
        # recall = metrics.recall_score(target_y, pred_y)
        # precision = metrics.precision_score(target_y, pred_y)
        if save:
            # self._metrics["f1"].append(f1_score)
            # self._metrics["pres"].append(precision)
            # self._metrics["recall"].append(recall)
            self._metrics["val_loss"].append(averaged_losses)
        return averaged_losses

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, targets = next_batch["inputs"], next_batch["targets"]
        inputs, targets = cls.to_var(inputs), cls.to_var(targets)
        return inputs, targets


if __name__ == "__main__":
    from solution1.dataset import DriverDataset, ToTensor
    from torch.utils.data import DataLoader
    top = None

    transformed_dataset = DriverDataset("../data/for_train.csv",  is_train=True, transform=ToTensor(), top=top)
    validation_dataset = DriverDataset("../data/for_validation.csv", is_train=False, top=top,
                                       transform=ToTensor())

    dataloader = DataLoader(transformed_dataset, batch_size=4096, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(validation_dataset, batch_size=1024, shuffle=False, num_workers=1)

    main_logger = Logger("../logs")

    input_layer = transformed_dataset.num_features
    net = Autoencoder([input_layer, input_layer // 2, input_layer // 4], 10)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.MSELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.0002)
    net.fit(optim, loss_func, dataloader, val_dataloader, 50, logger=main_logger, verbose=False)

