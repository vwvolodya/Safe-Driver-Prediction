import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from torchvision import transforms, utils


class ToTensor:
    def __call__(self, x):
        result = {k: torch.from_numpy(v) for k, v in x.items()}
        return result


class DriverDataset(Dataset):
    def __init__(self, path, transform=None, inference_only=False):
        self.transform = transform
        self._inference_only = inference_only
        self.data = pd.read_csv(path)
        self.array = self.data.as_matrix()
        self.names = "id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin," \
                     "ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin," \
                     "ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01," \
                     "ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat," \
                     "ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11," \
                     "ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04," \
                     "ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12," \
                     "ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin," \
                     "ps_calc_19_bin,ps_calc_20_bin".split(",")
        self.mapping = {name: i + 1 for i, name in enumerate(self.names)}
        self.target_column = "target"
        self.exclude_columns = {"id", "target"}

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.array[idx, :]
        target_idx = self.mapping[self.target_column]
        feature = []
        for col in self.names:
            if col in self.exclude_columns:
                continue
            index = self.mapping[col]
            feature.append(row[index])
        x = np.array(feature)
        item = {"inputs": x}
        if not self._inference_only:
            y = np.array([row[target_idx]])
            item["labels"] = y

        if self.transform:
            item = self.transform(item)
        return item


if __name__ == "__main__":
    transformed_dataset = DriverDataset("../data/for_train.csv", transform=ToTensor())
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['inputs'].size(), sample['labels'].size())
        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['inputs'].size(), sample_batched['labels'].size())
        if i_batch == 3:
            break
