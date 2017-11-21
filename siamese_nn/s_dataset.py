import numpy as np
import pandas as pd
from base.dataset import ToTensor, BaseDataset
from random import choice


class SiameseDataset(BaseDataset):
    def __init__(self, path, length, inference_only=False, transform=None, is_train=False, for_classifier=False,
                 duplicate_numeric_features=False, top=None):

        self.transform = transform
        self.inference_only = inference_only
        self.for_classifier = for_classifier
        self.target_column = "target"
        self.is_train = is_train
        mean_file = "mean.pkl"
        excluded = {"id", "target"}

        data = pd.read_csv(path)
        self.ids = data["id"].as_matrix()

        if not for_classifier:
            pos_data = self.get_special_label_data(data, label=1)
            neg_data = self.get_special_label_data(data, label=0)

            pos_categorical = self.get_categorical_df(pos_data)
            neg_categorical = self.get_categorical_df(neg_data)

            pos_numeric = self.get_numeric_df(pos_data, excluded)
            neg_numeric = self.get_numeric_df(neg_data, excluded)

            pos_numeric = self.replace_na(pos_numeric, mean_file)
            neg_numeric = self.replace_na(neg_numeric, mean_file)

            if duplicate_numeric_features:
                self.pos_x = np.column_stack((pos_numeric, pos_numeric, pos_numeric, pos_categorical.as_matrix()))
                self.neg_x = np.column_stack((neg_numeric, neg_numeric, neg_numeric, neg_categorical.as_matrix()))
            else:
                self.pos_x = np.column_stack((pos_numeric, pos_categorical.as_matrix()))
                self.neg_x = np.column_stack((neg_numeric, neg_categorical.as_matrix()))
            self.pos_range = range(len(self.pos_x))
            self.neg_range = range(len(self.neg_x))

            self.length = length
            self.num_features = self.pos_x.shape[1]
        else:
            categorical = self.get_categorical_df(data)
            numeric = self.get_numeric_df(data, excluded)

            numeric = self.replace_na(numeric, mean_file)
            if duplicate_numeric_features:
                self.x = np.column_stack((numeric, numeric, numeric, categorical.as_matrix()))
            else:
                self.x = np.column_stack((numeric, categorical.as_matrix()))
            self.final_stuff(data, top=top)
            self.length = self.x.shape[0]
        # self.final_stuff(data, top=top)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.for_classifier:
            original_x = self.x[idx, :]
            item = {"inputs": original_x}
            if self.inference_only:
                y = np.array([0])
                item["id"] = np.array([self.ids[idx]])
            else:
                y = np.array([self.y[idx]])
            item["targets"] = y
        else:
            same = choice([0, 1])
            positive = choice([0, 1]) == 1
            if positive:
                idx_0 = choice(self.pos_range)
                x_0 = self.pos_x[idx_0, :]
                if same == 0:
                    idx_1 = choice(self.pos_range)
                    x_1 = self.pos_x[idx_1, :]
                else:
                    idx_1 = choice(self.neg_range)
                    x_1 = self.neg_x[idx_1, :]
            else:
                idx_0 = choice(self.neg_range)
                x_0 = self.neg_x[idx_0, :]
                if same == 0:
                    idx_1 = choice(self.pos_range)
                    x_1 = self.pos_x[idx_1, :]
                else:
                    idx_1 = choice(self.neg_range)
                    x_1 = self.neg_x[idx_1, :]

            item = {"input0": x_0, "input1": x_1, "label": np.array([same])}

        if self.transform:
            item = self.transform(item)
        return item


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    for_classifier = True
    train_ds = SiameseDataset("../data/for_train.csv", 10, transform=ToTensor(), top=10, for_classifier=for_classifier,
                              is_train=True)
    for i in range(len(train_ds)):
        sample = train_ds[i]
        if not for_classifier:
            print(i, sample['input0'].size(), sample['input1'].size(), sample["label"].size())
        else:
            print(i, sample["inputs"].size(), sample["targets"].size())
        if i == 3:
            break

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)
    for i_batch, sample_batched in enumerate(loader):
        if not for_classifier:
            print(i_batch, sample_batched['input0'].size(), sample_batched['input1'].size(),
                  sample_batched["label"].size())
        else:
            print(i_batch, sample_batched["inputs"].size(), sample_batched["targets"].size())
        if i_batch == 3:
            break
