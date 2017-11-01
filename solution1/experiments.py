import torch
from torch.autograd import Variable
from tqdm import tqdm as progressbar
from solution1.model import Autoencoder
from solution1.dataset import DriverDataset
import numpy as np
import pandas as pd


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
        loss = loss_fn(predictions, targets)
        loss1 = loss.data.cpu().numpy()
        y = y.numpy()
        losses.append([loss1[0], y[0][0]])
    return losses


if __name__ == "__main__":
    from solution1.dataset import ToTensor
    from torch.utils.data import DataLoader
    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.set(style="white")

    top = None

    transformed_dataset = DriverDataset("../data/for_train.csv", is_train=False, transform=ToTensor(), top=top)
    test_dataset = DriverDataset("../data/for_test.csv", is_train=False, transform=ToTensor(), top=top)
    validation_dataset = DriverDataset("../data/for_validation.csv", is_train=False, top=top,
                                       transform=ToTensor())
    dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=12)
    val_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    autoenc = Autoencoder.load("models/autoenc_100.mdl")
    autoenc.eval()

    loss_func = torch.nn.MSELoss()
    if torch.cuda.is_available():
        loss_func.cuda()
    train_losses = calc_mse(dataloader, autoenc, loss_func)
    val_losses = calc_mse(val_dataloader, autoenc, loss_func)
    # test_losses = calc_mse(test_dataloader, autoenc, loss_func)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    # test_losses = np.array(test_losses)

    print(train_losses.shape)
    print(val_losses.shape)
    # print(test_losses.shape)

    np.save("train", train_losses)
    # np.save("test.npa", test_losses)
    np.save("validation", val_losses)

    train_df = pd.DataFrame(train_losses, columns=["mse", "y"])
    train_df["y"] = train_df["y"].astype("object")

    val_df = pd.DataFrame(val_losses, columns=["mse", "y"])
    val_df["y"] = val_df["y"].astype("object")

    plot_data(train_df["mse"], train_df["y"], "Train")
    plot_data(val_df["mse"], val_df["y"], "Val")
