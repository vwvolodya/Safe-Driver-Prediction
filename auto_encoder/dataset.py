import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from base.dataset import ToTensor
# from torchvision import transforms, utils


class AutoEncoderDataset(Dataset):
    def __init__(self, path, noise_rate=0.5, is_train=True, transform=None, inference_only=False, top=None):
        self.transform = transform
        self._inference_only = inference_only
        self.noise_rate = noise_rate
        data = pd.read_csv(path)
        cols = data.columns

        self.x = data.iloc[:, 1: -1].as_matrix()
        self.is_train = is_train
        self.target_column = cols[-1]
        self.y = data[self.target_column].as_matrix()
        if top:
            self.x = self.x[:top, :]        # get only top N samples
        self.num_features = self.x.shape[1]
        self.shape = self.x.shape
        self.__print_stats(data)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :]
        y = np.array([self.y[idx]])
        if self.is_train:
            mask = np.random.binomial(1, self.noise_rate, x.shape[0])
            x = x * mask
        item = {"inputs": x, "targets": x, "y": y}

        if self.transform:
            item = self.transform(item)
        return item

    def __print_stats(self, data):
        positive = data[self.target_column]
        all_pos = sum(positive)
        all_items = len(self)
        percentage = 1.0 * all_pos / all_items
        print("There are %s positive examples" % percentage)
        print("Total number of examples is %s" % all_items)


if __name__ == "__main__":
    transformed_dataset = AutoEncoderDataset("../data/for_train_processed.csv", transform=ToTensor())
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
