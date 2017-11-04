import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from base.dataset import ToTensor


class TfDriverDataset(Dataset):
    def __init__(self, path, is_train=True, transform=None, inference_only=False, top=None):
        self.transform = transform
        self._inference_only = inference_only
        self.names = "id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin," \
                     "ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin," \
                     "ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01," \
                     "ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat," \
                     "ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11," \
                     "ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04," \
                     "ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12," \
                     "ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin," \
                     "ps_calc_19_bin,ps_calc_20_bin".split(",")
        self.target_column = "target"
        self.exclude_columns = {"id", "target"}     # "ps_ind_14", "ps_car_11_cat"}
        used_columns = list(set(self.names) - self.exclude_columns)

        data = pd.read_csv(path)
        print("Original", data.shape)
        self.ids = data["id"].as_matrix()
        if is_train:
            magic_multiplier = 5  # 26 is because we have 3.5 % of true labels and we want wo make dataset balanced
            # augment data to change balance.
            true_rows = data[self.target_column] == 1
            slice = data[true_rows]
            data = data.append([slice] * magic_multiplier, ignore_index=True)
            data = data.sample(frac=1)

        x = data[used_columns].as_matrix()
        if not inference_only:
            self.y = data[self.target_column].as_matrix()

        self.x = x
        if top:
            self.x = self.x[:top, :]
        self.num_features = self.x.shape[1]
        self.shape = self.x.shape
        if not inference_only:
            self.__print_stats(data)

    def __len__(self):
        return self.shape[0]

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
        positive = data[self.target_column]
        all_pos = sum(positive)
        all_items = len(self)
        percentage = 1.0 * all_pos / all_items
        print("There are %s positive examples" % percentage)
        print("Total number of examples is %s", all_items)


if __name__ == "__main__":
    transformed_dataset = TfDriverDataset("../data/for_train_tf.csv", transform=ToTensor())
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
