import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from base.dataset import ToTensor
# from torchvision import transforms, utils


class DriverDataset(Dataset):
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
        # ps_ind_14 = (ps_ind_10 + ps_ind_11 + ps_ind_12 + ps_ind_13)
        # ps_car_11_cat  has 104 classes(!)

        self.exclude_columns = {"id", "target", "ps_ind_14", "ps_car_11_cat"}

        scaling_ind = [
            "ps_ind_01", "ps_ind_03", "ps_ind_14", "ps_ind_15"
        ]
        categorical_ind = [
            "ps_ind_02_cat", "ps_ind_04_cat", "ps_ind_05_cat"
        ]
        # total 18 ind features. 3 categorical, 11 binary , 4 numerical

        # reg 3 features. ps_reg_03 may need scaling. values like 1.49. 3 numerical features

        categorical_car = [
            "ps_car_01_cat", "ps_car_02_cat", "ps_car_03_cat", "ps_car_04_cat", "ps_car_05_cat", "ps_car_06_cat",
            "ps_car_07_cat", "ps_car_08_cat", "ps_car_09_cat", "ps_car_10_cat",
            "ps_car_11_cat",    # this feature has the most categories. ( > 100 )
        ]
        scaling_car = [
            "ps_car_11", "ps_car_15"
        ]
        # total 16 features / 5 numeric 11 categorical
        scaling_calc = [
            "ps_calc_04", "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08", "ps_calc_09", "ps_calc_10",
            "ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14"
        ]
        # total 20 features. / 14 numeric, 6 binary

        self.mapping = {name: i + 1 for i, name in enumerate(self.names)}
        self.target_column = "target"
        self.category_columns = [i for i in self.names if "cat" in i and i not in self.exclude_columns]
        self.binary_columns = [i for i in self.names if "bin" in i]
        self.columns_for_scaling = list(set(self.names) - self.exclude_columns - set(self.category_columns) - \
                                   set(self.binary_columns))

        data = pd.read_csv(path)
        # take only negative
        if is_train:
            false_rows = data[self.target_column] != 1
            data = data[false_rows]
            # data = data[self.target_column != 1]
        scaled = data[self.columns_for_scaling]

        new = pd.DataFrame(data[self.category_columns], dtype='object')
        categorical = pd.get_dummies(new)
        binary = data[self.binary_columns]
        categorical_matrix = categorical.as_matrix()
        binary_matrix = binary.as_matrix()
        self.x = np.column_stack((categorical_matrix, binary_matrix, scaled.as_matrix()))
        if top:
            self.x = self.x[:top, :]        # get only top N samples
        self.num_features = self.x.shape[1]
        self.shape = self.x.shape
        self.__print_stats(data)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :]
        item = {"inputs": x, "targets": x}

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
    transformed_dataset = DriverDataset("../data/for_train.csv", transform=ToTensor())
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
