from mlp.dataset import DriverDataset, ToTensor
from mlp.model import DriverClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm as progressbar
import pandas as pd


test_file = "../data/prediction/test.csv"

test_dataset = DriverDataset(test_file, transform=ToTensor(),is_train=False, inference_only=True, top=None)
print(test_dataset.shape)
net = DriverClassifier.load("./models/model_12.mdl")
net.eval()

dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=1)


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
