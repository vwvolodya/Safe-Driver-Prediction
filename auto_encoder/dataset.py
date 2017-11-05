import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from base.dataset import ToTensor
# from torchvision import transforms, utils


class AutoEncoderDataset(Dataset):
    def __init__(self, path, noise_rate=0.5, is_train=True, transform=None, inference_only=False,
                 top=None, remove_positive=True, use_categorical=True):
        self.transform = transform
        self._inference_only = inference_only
        self.noise_rate = noise_rate
        mean_file = "mean.pkl"
        data = pd.read_csv(path)
        existing_columns = list(data.columns)
        excluded = {"id", "target"}
        if use_categorical:
            print("Will use only categorical data")
            needed_columns = [i for i in existing_columns if "_cat" in i or "_bin" in i]
        else:
            print("will use only numeric data")
            needed_columns = [i for i in existing_columns if "_cat" not in i and "_bin" not in i and i not in excluded]
        print("Diff", set(needed_columns) - set(existing_columns))
        if remove_positive:
            print("Will remove all positives.")
            false_rows = data["target"] != 1
            data = data[false_rows]

        needed_data = data[needed_columns]
        if not use_categorical:
            print("Replacing all -1 with NaN")
            needed_data = needed_data.replace(-1, np.NaN)
        if os.path.exists(mean_file):
            print("Will LOAD mean vector now.")
            with open(mean_file, "rb") as f:
                mean_vector = pickle.load(f)
        else:
            print("Will calculate mean")
            mean_vector = needed_data.mean()
            with open(mean_file, "wb") as f:
                pickle.dump(mean_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

        needed_data = needed_data.fillna(mean_vector)

        # self.x = data.iloc[:, 1: -1].as_matrix()
        self.x = needed_data.as_matrix()
        self.is_train = is_train
        if self.is_train:
            print("Will use target column")
            self.target_column = "target"
            self.y = data[self.target_column].as_matrix()
        if top:
            "will use only top..."
            self.x = self.x[:top, :]        # get only top N samples
        self.num_features = self.x.shape[1]
        self.shape = self.x.shape
        self.__print_stats(data)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :]
        if self.is_train:
            y = np.array([self.y[idx]])
            # higher noise rate actually results in less nodes set to 0.
            mask = np.random.binomial(1, 1 - self.noise_rate, x.shape[0])
            x = x * mask
        else:
            y = np.array([0])
        item = {"inputs": x, "targets": x, "y": y}

        if self.transform:
            item = self.transform(item)
        return item

    def __print_stats(self, data):
        if self.is_train:
            positive = data[self.target_column]
            all_pos = sum(positive)
            all_items = len(self)
            percentage = 1.0 * all_pos / all_items
            print("There are %s positive examples" % percentage)
            print("Total number of examples is %s" % all_items)
        print("#################################")


if __name__ == "__main__":
    transformed_dataset = AutoEncoderDataset("../data/one-hot-train.csv", transform=ToTensor())
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['inputs'].size(), sample['targets'].size())
        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['inputs'].size(), sample_batched['targets'].size())
        if i_batch == 3:
            break
