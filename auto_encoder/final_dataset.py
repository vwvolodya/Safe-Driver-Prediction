import pandas as pd
import numpy as np
import os, pickle
from torch.utils.data import Dataset, DataLoader
from base.dataset import ToTensor
# from torchvision import transforms, utils


class FinalDataset(Dataset):
    def __init__(self, path, is_train=True, inference_only=False, transform=None, top=None, augment=None):
        self.transform = transform
        self.inference_only = inference_only
        mean_file = "mean.pkl"
        data = pd.read_csv(path)
        self.target_column = "target"
        if not inference_only and augment:
            # augment data to change balance.
            true_rows = data[self.target_column] == 1
            slice_ = data[true_rows]
            data = data.append([slice_] * augment, ignore_index=True)
            data = data.sample(frac=1)
        existing_columns = list(data.columns)
        excluded = {"id", "target"}

        cat_columns = [i for i in existing_columns if "_cat" in i or "_bin" in i]
        print("will use only numeric data")
        val_columns = [i for i in existing_columns if "_cat" not in i and "_bin" not in i and i not in excluded]

        val_data = data[val_columns]
        cat_data = data[cat_columns]
        print("Replacing all -1 with NaN")
        val_data = val_data.replace(-1, np.NaN)
        self.ids = data["id"].as_matrix()
        if os.path.exists(mean_file):
            print("Will LOAD mean vector now.")
            with open(mean_file, "rb") as f:
                mean_vector = pickle.load(f)
        else:
            raise Exception("No mean file found!!!")

        val_data = val_data.fillna(mean_vector)

        # self.x = data.iloc[:, 1: -1].as_matrix()
        self.numeric_x = val_data.as_matrix()
        self.cat_x = cat_data.as_matrix()
        self.is_train = is_train
        if self.is_train:
            print("Will use target column")
            self.target_column = "target"
            self.y = data[self.target_column].as_matrix()
        if top:
            "will use only top..."
            self.numeric_x = self.numeric_x[:top, :]  # get only top N samples
        self.__print_stats(data)

    def __len__(self):
        return self.numeric_x.shape[0]

    def __getitem__(self, idx):
        numeric_x = self.numeric_x[idx, :]
        cat_vector = self.cat_x[idx, :]

        if not self.inference_only:
            y = self.y[idx]
        else:
            y = 0
        item = {"numeric": numeric_x, "targets": np.array([y]), "categorical": cat_vector}
        if self.inference_only:
            item["id"] = np.array([self.ids[idx]])

        if self.transform:
            item = self.transform(item)
        return item

    def __print_stats(self, data):
        if self.inference_only:
            return
        positive = data[self.target_column]
        all_pos = sum(positive)
        all_items = len(self)
        percentage = 1.0 * all_pos / all_items
        print("There are %s positive examples" % percentage)
        print("Total number of examples is %s" % all_items)


if __name__ == "__main__":
    top = None

    transformed_dataset = FinalDataset("../data/for_train.csv", transform=ToTensor(), top=top, is_train=True, augment=3)
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['numeric'].size(), sample['targets'].size(), sample["categorical"].size())
        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['numeric'].size(), sample_batched['targets'].size())
        if i_batch == 3:
            break
