import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from base.dataset import ToTensor, BaseDataset


class AutoEncoderDataset(BaseDataset):
    def __init__(self, path, noise_rate=None, is_train=True, transform=None, inference_only=False,
                 top=None, remove_positive=False, augment=None):
        self.transform = transform
        self.inference_only = inference_only
        self.noise_rate = noise_rate
        self.target_column = "target"
        self.is_train = is_train
        mean_file = "mean.pkl"
        excluded = {"id", "target"}

        data = pd.read_csv(path)

        if augment:
            data = self.augment_dataframe(data, augment, target_column=self.target_column)

        if remove_positive:
            data = self.remove_positive(data)

        categorical = self.get_categorical_df(data)
        numeric = self.get_numeric_df(data, excluded)

        numeric = self.replace_na(numeric, mean_file)
        numeric = self.scale(numeric, "max_scaler.pkl")

        self.x = np.column_stack((numeric, categorical.as_matrix()))
        self.final_stuff(data, top=top)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :]
        if self.noise_rate:
            # higher noise rate actually results in less nodes set to 0.
            mask = np.random.binomial(1, 1 - self.noise_rate, x.shape[0])
            x = x * mask

        if not self.inference_only:
            y = np.array([self.y[idx]])
        else:
            y = np.array([0])
        item = {"inputs": x, "targets": x, "y": y}

        if self.transform:
            item = self.transform(item)
        return item


if __name__ == "__main__":
    transformed_dataset = AutoEncoderDataset("../data/for_train.csv", transform=ToTensor(), augment=2)
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['inputs'].size(), sample['targets'].size(), sample["y"].size())
        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['inputs'].size(), sample_batched['targets'].size())
        if i_batch == 3:
            break
