import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm as progressbar
from sklearn import metrics
from base.model import BaseModel
from base.logger import Logger
import numpy as np
import pandas as pd
from collections import defaultdict


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1-label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0),
                2)
        )

        return loss_contrastive


class SiameseNet(BaseModel):
    def __init__(self, input_size, hidden_size, hidden_size_1, encoder_features,
                 num_classes=1, model_prefix="", classifier=False):
        super().__init__()
        self.model_prefix = model_prefix
        self.is_classifier = classifier
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(encoder_features, num_classes)
        nn.init.xavier_uniform(self.out.weight, gain=0.01)

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        nn.init.xavier_normal(self.fc1.weight, gain=0.01)

        self.fc2 = nn.Linear(hidden_size, hidden_size_1, bias=False)
        nn.init.xavier_normal(self.fc2.weight, gain=0.01)

        self.fc3 = nn.Linear(hidden_size_1, encoder_features, bias=False)
        nn.init.xavier_normal(self.fc3.weight, gain=0.01)

        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size_1)

        self.activation = nn.Tanh()

    def classifier(self, x):
        # x here is the result of encoder's output
        probs = self.out(x)
        probs = self.sigmoid(probs)
        return probs

    def encoder(self, x):
        out = self.bn1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.bn2(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.bn3(out)
        out = self.fc3(out)
        out = self.activation(out)
        return out

    def forward(self, x):
        y = self.encoder(x)
        if self.is_classifier is True:
            y = self.classifier(y)
        return y

    @classmethod
    def _get_classes(cls, predictions):
        classes = (predictions.data > 0.5).float()
        pred_y = classes.cpu().numpy().squeeze()
        return pred_y

    def predict(self, x, return_classes=False):
        predictions = self.__call__(x)
        classes = None
        if self.is_classifier and return_classes:
            classes = self._get_classes(predictions)
        return predictions, classes

    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        if not self.is_classifier:
            return {}
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

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)     # here we assume data type torch.Tensor
        input0, input1, label = next_batch["input0"], next_batch["input1"], next_batch["label"]
        input0, input1, label = cls.to_var(input0), cls.to_var(input1), cls.to_var(label)
        return input0, input1, label

    @classmethod
    def _get_clf_inputs(cls, iterator):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, labels = next_batch["inputs"], next_batch["targets"]
        inputs, labels = cls.to_var(inputs), cls.to_var(labels)
        return inputs, labels

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
            input0, input1, label = self._get_inputs(iterator)
            prob0, _ = self.predict(input0, return_classes=False)
            prob1, _ = self.predict(input1, return_classes=False)

            loss = loss_fn(prob0, prob1, label)
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
                input0, input1, label = self._get_inputs(data_iter)

                prediction0, _ = self.predict(input0, return_classes=False)
                prediction1, _ = self.predict(input1, return_classes=False)

                optim.zero_grad()
                loss = loss_fn(prediction0, prediction1, label)
                loss.backward()
                optim.step()

                self._accumulate_results(None, None, loss=loss.data[0])

            self.evaluate(logger, validation_data_loader, loss_fn, switch_to_eval=True)
            self.save("./models/%ss_enc_%s.mdl" % (self.model_prefix, str(e + 1)))

    def evaluate_clf(self, logger, loader, loss_fn=None, switch_to_eval=False, **kwargs):
        # aggregate results from training epoch.
        train_losses = self._predictions.pop("train_loss")
        train_loss = sum(train_losses) / len(train_losses)
        train_metrics_1 = self._compute_metrics(self._predictions["target"], self._predictions["predicted"])
        train_metrics_2 = self._compute_metrics(self._predictions["target"], self._predictions["probs"],
                                                predictions_are_classes=False)
        train_metrics = {"clf_train_loss": train_loss}
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
            inputs, targets = self._get_clf_inputs(iterator)
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
        computed_metrics.update({"clf_val_loss": val_loss})
        computed_metrics.update(computed_metrics_1)
        if switch_to_eval:
            # switch back to train
            self.train()

        self._log_and_reset(logger, data=train_metrics, log_grads=True)
        self._log_and_reset(logger, data=computed_metrics, log_grads=False)

        self._predictions = defaultdict(list)
        return computed_metrics

    def fit_classifier(self, optim, loss_fn, data_loader, validation_data_loader, num_epochs, logger, prev_epoch=0):
        for e in progressbar(range(num_epochs)):
            iter_per_epoch = len(data_loader)
            self._epoch = prev_epoch + e
            data_iter = iter(data_loader)
            for i in range(iter_per_epoch):
                inputs, labels = self._get_clf_inputs(data_iter)

                predictions, classes = self.predict(inputs, return_classes=True)

                optim.zero_grad()
                loss = loss_fn(predictions, labels)
                loss.backward()
                optim.step()

                self._accumulate_results(self.to_np(labels).squeeze(),
                                         classes,
                                         loss=loss.data[0],
                                         probs=self.to_np(predictions).squeeze())
            self.evaluate_clf(logger, validation_data_loader, loss_fn, switch_to_eval=True)
            self.save("./models/clf_%s.mdl" % str(e + 1))

    def gather(self, data_loader):
        all_predictions = []
        all_targets = np.array([])

        iter_per_epoch = len(data_loader)
        data_iter = iter(data_loader)
        for i in range(iter_per_epoch):
            inputs, labels = self._get_clf_inputs(data_iter)
            predictions, _ = self.predict(inputs, return_classes=False)

            target_y = self.to_np(labels).squeeze()
            encoding = self.to_np(predictions)
            all_targets = np.append(all_targets, target_y)
            all_predictions.append(encoding)

        all_predictions = np.row_stack(all_predictions)
        df = pd.DataFrame(np.column_stack((all_predictions, all_targets)))
        print(df.describe())
        df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    from siamese_nn.s_dataset import SiameseDataset, ToTensor
    from torch.utils.data import DataLoader

    top = 500000
    val_top = 60000
    train_batch_size = 8192
    test_batch_size = 4096

    main_logger = Logger("../logs")

    train_ds = SiameseDataset("../data/for_train.csv", top, is_train=True, transform=ToTensor(),
                              duplicate_numeric_features=True)
    # train_ds_clf = SiameseDataset("../data/for_train.csv", top, is_train=True, transform=ToTensor(),
    #                               top=top, for_classifier=True)
    val_ds = SiameseDataset("../data/for_test.csv", val_top, is_train=False, transform=ToTensor(),
                            duplicate_numeric_features=True)
    # val_ds_clf = SiameseDataset("../data/for_test.csv", val_top, is_train=True, transform=ToTensor(),
    #                             top=val_top, for_classifier=True)

    input_layer = train_ds.num_features
    net = SiameseNet(input_layer, int(input_layer * 1.3), 32, 3)
    #
    loss_func = ContrastiveLoss(margin=0.5)

    net.show_env_info()
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, num_workers=12, pin_memory=True)
    # train_loader_clf = DataLoader(train_ds_clf, batch_size=train_batch_size, num_workers=12, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=6, pin_memory=True)
    # val_loader_clf = DataLoader(val_ds_clf, batch_size=test_batch_size, num_workers=6, pin_memory=True)

    if torch.cuda.is_available():
        net.cuda()
        loss_func.cuda()

    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.fit(optim, loss_func, train_loader, val_loader, 250, logger=main_logger)

    # print("Asking...")
    # value = input("Please select epoch number   ")
    # print("Going to use epoch %s for classifier   " % value)
    # net.load("./models/s_enc_%s.mdl" % value)
    # net.is_classifier = False
    # net.cuda()
    #
    # ds = SiameseDataset("../data/for_test.csv", val_top, is_train=True, transform=ToTensor(),
    #                     duplicate_numeric_features=True, for_classifier=True, top=val_top)
    # loader = DataLoader(ds, batch_size=test_batch_size, num_workers=6, pin_memory=True)
    # net.eval()
    # net.gather(loader)
    #
    # clf_loss = nn.BCELoss()
    # if torch.cuda.is_available():
    #     clf_loss.cuda()
    # opt = torch.optim.Adam([net.out.weight, net.out.bias], lr=0.0001)
    # net.fit_classifier(opt, clf_loss, train_loader_clf, val_loader_clf, 50, logger=main_logger, prev_epoch=60)
