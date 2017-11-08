from auto_encoder.model import Autoencoder
from auto_encoder.final import FinalModel
from auto_encoder.dataset import AutoEncoderDataset, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm as progressbar
import pandas as pd
import torch


test_file = "../data/prediction/one-hot-test.csv"


test_dataset = AutoEncoderDataset(test_file, transform=ToTensor(), is_train=False, inference_only=True, top=None,
                                  for_classifier=True, duplicate_numeric_features=True)
dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=1)
print(len(test_dataset))

net = Autoencoder.load("./models/clf_39.mdl")
net.is_classifier = True
net.eval()


def predictions(model, loader):
    iterator = iter(loader)
    iter_per_epoch = len(loader)
    result = {}
    for i in progressbar(range(iter_per_epoch)):
        next_batch = next(iterator)  # here we assume data type torch.Tensor
        inputs, ids = next_batch["inputs"], next_batch["id"]
        inputs = model.to_var(inputs)
        ids_np = ids.numpy().squeeze().tolist()
        ids_np = [int(i) for i in ids_np]
        probs, _ = model.predict(inputs, return_classes=False)
        probs = model.to_np(probs).squeeze()
        probs = probs.tolist()
        chunk = dict(zip(ids_np, probs))

        result.update(chunk)
    return result


preds = predictions(net, dataloader)
new_df = pd.DataFrame(list(preds.items()), columns=["id", "target"])
new_df.to_csv("../data/prediction/predicted.csv", float_format='%.4f', index=False)
print(new_df.shape)
print()
