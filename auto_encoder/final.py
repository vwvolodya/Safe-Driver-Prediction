import torch
from torch import nn
from tqdm import tqdm as progressbar
from sklearn import metrics
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
        if return_classes:
            predictions = torch.round(predictions)
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

            averaged_losses, f1_score, recall, precision = self.evaluate(validation_data_loader, loss_fn=loss_fn,
                                                                         return_classes=True)
            self._log_and_reset(logger)
            self._log_and_reset(logger, data={"loss": loss.data[0], "val_loss": averaged_losses,
                                              "f1": f1_score, "precision": precision, "recall": recall
                                              })
            self.save("./models/final_%s.mdl" % e)

    def _compute_metrics(self, target_y, pred_y, losses, save=True):
        averaged_losses = sum(losses) / len(losses)
        f1_score = metrics.f1_score(target_y, pred_y)
        recall = metrics.recall_score(target_y, pred_y)
        precision = metrics.precision_score(target_y, pred_y)
        if save:
            self._metrics["f1"].append(f1_score)
            self._metrics["pres"].append(precision)
            self._metrics["recall"].append(recall)
            self._metrics["val_loss"].append(averaged_losses)
        return averaged_losses, f1_score, recall, precision

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

    train_dataset = FinalDataset("train.npy", is_train=True, transform=ToTensor(), top=top)
    validation_dataset = FinalDataset("validation.npy", is_train=False, top=top, transform=ToTensor())

    dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(validation_dataset, batch_size=4096, shuffle=False, num_workers=1)

    main_logger = Logger("../logs")

    net = FinalModel(11, 6, 1)
    net.show_env_info()
    if torch.cuda.is_available():
        net.cuda()

    loss_func = torch.nn.BCELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.0002)
    net.fit(optim, loss_func, dataloader, val_dataloader, 150, logger=main_logger, verbose=False)

