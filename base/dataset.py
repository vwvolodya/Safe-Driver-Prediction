import numpy as np
import os, pickle
import torch
from torch.utils.data import Dataset
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class BaseDataset(Dataset):
    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    @classmethod
    def augment_dataframe(cls, data, multiplier, target_column="target"):
        true_rows = data[target_column] == 1
        slice_ = data[true_rows]
        data = data.append([slice_] * multiplier, ignore_index=True)
        data = data.sample(frac=1)
        print("After augmentation shape is ", data.shape)
        return data

    @classmethod
    def remove_positive(cls, data):
        print("Will remove all positives.")
        false_rows = data["target"] != 1
        data = data[false_rows]
        print("New shape is ", data.shape)
        return data

    @classmethod
    def replace_na(cls, data, mean_file, na_value=-1):
        print("Replacing all -1 with NaN")
        needed_data = data.replace(na_value, np.NaN)
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
        return needed_data

    @classmethod
    def scale(cls, data, scaler_filename, scale_type="min-max"):
        scaler_class_mapping = {
            "min-max": MinMaxScaler,
            "std": StandardScaler,
            "robust": RobustScaler
        }
        if os.path.exists(scaler_filename):
            print("Loading scaler from file")
            scaler = joblib.load(scaler_filename)
            val_data = scaler.transform(data)
        else:
            scaler = scaler_class_mapping[scale_type]()
            val_data = scaler.fit_transform(data)
            print("Saving scaler to file")
            joblib.dump(scaler, scaler_filename)
        return val_data

    @classmethod
    def get_categorical_df(cls, data):
        existing_columns = list(data.columns)
        needed_columns = [i for i in existing_columns if "_cat" in i or "_bin" in i]
        needed_data = data[needed_columns]
        return needed_data

    @classmethod
    def get_numeric_df(cls, data, excluded):
        existing_columns = list(data.columns)
        print("will use only numeric data")
        needed_columns = [i for i in existing_columns if "_cat" not in i and "_bin" not in i and i not in excluded]
        needed_data = data[needed_columns]
        return needed_data

    def final_stuff(self, data, top=None):
        if self.is_train:
            print("Will use target column")
            self.y = data[self.target_column].as_matrix()
        if top:
            "will use only top..."
            self.x = self.x[:top, :]        # get only top N samples
        self.num_features = self.x.shape[1]
        self.shape = self.x.shape
        self.__print_stats(data)

    def __print_stats(self, data):
        if self.inference_only:
            return
        positive = data[self.target_column]
        all_pos = sum(positive)
        all_items = len(self)
        percentage = 1.0 * all_pos / all_items
        print("There are %s positive examples" % percentage)
        print("Total number of examples is %s" % all_items)
        print("#################################")


class ToTensor:
    def __call__(self, x):
        result = {k: torch.from_numpy(v).float() for k, v in x.items()}
        return result
