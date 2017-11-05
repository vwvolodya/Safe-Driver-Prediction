import torch
from torch.autograd import Variable
from tqdm import tqdm as progressbar
from auto_encoder.model import Autoencoder
from auto_encoder.dataset import AutoEncoderDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(x, y, title):
    """
    Plots scatterplot
    """
    x1 = range(x.shape[0])
    plt.clf()
    plt.title(title)
    plt.scatter(x1, x, marker='o', c=y, s=20, edgecolor='k')
    plt.show()


def to_var(x, use_gpu=True):
    if torch.cuda.is_available() and use_gpu:
        x = x.cuda()
    return Variable(x)


def _get_inputs(iterator):
    next_batch = next(iterator)  # here we assume data type torch.Tensor
    inputs, targets = next_batch["inputs"], next_batch["targets"]
    y = next_batch["y"]
    inputs, targets = to_var(inputs), to_var(targets)
    return inputs, targets, y


def calc_mse(data_loader, model, loss_fn):
    iter_per_epoch = len(data_loader)
    data_iter = iter(data_loader)
    losses = []
    for i in progressbar(range(iter_per_epoch)):
        inputs, targets, y = _get_inputs(data_iter)

        predictions, _ = model.predict(inputs)
        features = model.predict_encoder(inputs)
        np_features = features.data.cpu().numpy()
        # code specific for batch_size = 1
        loss = loss_fn(predictions, targets)
        loss1 = loss.data.cpu().numpy()
        y = y.numpy()
        first_part = [i for i in np_features[0]]
        second = first_part + [loss1[0], y[0][0]]
        losses.append(second)
    return losses


def get_encoder_repr(data_loader, model):
    iter_per_epoch = len(data_loader)
    data_iter = iter(data_loader)
    result = []
    for i in progressbar(range(iter_per_epoch)):
        inputs, targets, y = _get_inputs(data_iter)
        # targets and inputs here should be the same
        outputs = model.predict(targets)
        np_outputs = outputs.data.cpu().numpy()
        features = model.predict_encoder(targets)
        np_features = features.data.cpu().numpy()
        y = y.numpy()
        first_part = [i for i in np_features[0]]
        second = first_part + [y[0][0]]
        result.append(second)
    return result


def gather_positive(path):
    data = pd.read_csv(path)
    true_rows = data["target"] == 1
    data = data[true_rows]
    data.to_csv("../data/train_pos.csv", index=False)


if __name__ == "__main__":
    gather_positive("../data/one-hot-train.csv")
    # from auto_encoder.dataset import ToTensor
    # from torch.utils.data import DataLoader
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # # sns.set(style="white")
    #
    # top = None
    #
    # train_ds = AutoEncoderDataset("../data/for_train.csv", is_train=False, transform=ToTensor(), top=top)
    # test_dataset = AutoEncoderDataset("../data/for_test.csv", is_train=False, transform=ToTensor(), top=top)
    #
    # dataloader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=12)
    # val_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    #
    # autoenc = Autoencoder.load("ready/adam_autoenc_20.mdl")
    # autoenc.eval()
    #
    # train_features = get_encoder_repr(dataloader, autoenc)
    # test_features = get_encoder_repr(val_dataloader, autoenc)
    # np.save("train", train_features)
    # np.save("test", test_features)

