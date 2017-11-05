import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from base.dataset import ToTensor
# from torchvision import transforms, utils


class FinalDataset(Dataset):
    def __init__(self, path, is_train=True, inference_only=False, transform=None, top=None, augment=None):
        self.transform = transform
        self.inference_only = inference_only
        self.is_train = is_train
        df = pd.read_csv(path)
        self.target_column = "target"
        existing_columns = list(df.columns)
        excluded_cols = {"id", self.target_column}
        cat_bin_cols = [i for i in existing_columns if "_cat" in i or "_bin" in i]
        numeric_cols = list(set(existing_columns) - excluded_cols - set(cat_bin_cols))

        if not inference_only and augment:
            # augment data to change balance.
            true_rows = df[self.target_column] == 1
            slice_ = df[true_rows]
            df = df.append([slice_] * augment, ignore_index=True)
            df = df.sample(frac=1)
        self.ids = df["id"].as_matrix()
        if not inference_only:
            self.y = df[self.target_column].as_matrix()
        data = df[numeric_cols].as_matrix()
        self.cat_data = df[cat_bin_cols].as_matrix()
        if top:
            data = data[:top, :]
        self.data = data
        self.__print_stats(df)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        numeric_x = self.data[idx, :]
        cat_vector = self.cat_data[idx, :]

        if not self.inference_only:
            y = self.y[idx]
        else:
            y = 0
        item = {"inputs": numeric_x, "targets": np.array([y]), "categorical": cat_vector}
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
        print(i, sample['inputs'].size(), sample['targets'].size())
        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['inputs'].size(), sample_batched['targets'].size())
        if i_batch == 3:
            break
