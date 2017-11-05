import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from base.dataset import ToTensor
import os, pickle
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class TfDriverDataset(Dataset):
    def __init__(self, path, is_train=True, transform=None, inference_only=False, top=None, augment=None):
        self._inference_only = inference_only
        self.transform = transform
        self.target_column = "target"
        data = pd.read_csv(path)
        print("Original", data.shape)
        self.ids = data["id"].as_matrix()
        if augment:
            magic_multiplier = augment  # 26 is because we have 3.5 % of true labels and we want wo make dataset balanced
            # augment data to change balance.
            true_rows = data[self.target_column] == 1
            slice_ = data[true_rows]
            data = data.append([slice_] * magic_multiplier, ignore_index=True)
            data = data.sample(frac=1)
            print("After augmentation shape is ", data.shape)

        mean_file = "mean.pkl"
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

        scaler_filename = "max_scaler.plk"
        if os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
            val_data = scaler.transform(val_data)
        else:
            scaler = MinMaxScaler()
            val_data = scaler.fit_transform(val_data)
            joblib.dump(scaler, scaler_filename)

        self.x = np.column_stack((val_data, cat_data.as_matrix()))
        self.is_train = is_train
        self.num_features = self.x.shape[1]
        if self.is_train:
            print("Will use target column")
            self.y = data[self.target_column].as_matrix()
        if top:
            "will use only top..."
            self.x = self.x[:top, :]  # get only top N samples
        self.__print_stats(data)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :]
        item = {"inputs": x}
        if not self._inference_only:
            y = np.array([self.y[idx]])
            item["targets"] = y
        else:
            item["targets"] = np.array([0])
            item["id"] = np.array([self.ids[idx]])

        if self.transform:
            item = self.transform(item)
        return item

    def __print_stats(self, data):
        if self._inference_only:
            return
        positive = data[self.target_column]
        all_pos = sum(positive)
        all_items = len(self)
        percentage = 1.0 * all_pos / all_items
        print("There are %s positive examples" % percentage)
        print("Total number of examples is %s", all_items)


if __name__ == "__main__":
    transformed_dataset = TfDriverDataset("../data/for_train.csv", transform=ToTensor())
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
