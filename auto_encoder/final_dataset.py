import numpy as np
from torch.utils.data import Dataset, DataLoader
from base.dataset import ToTensor
# from torchvision import transforms, utils


class FinalDataset(Dataset):
    def __init__(self, path, is_train=True, transform=None, top=None):
        self.transform = transform
        data = np.load(path)
        if top:
            data = data[:top, :]
        self.data = data
        self.__print_stats(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx, : -1]
        y = self.data[idx, -1]
        item = {"inputs": x, "targets": np.array([y])}

        if self.transform:
            item = self.transform(item)
        return item

    def __print_stats(self, data):
        all_pos = sum(data[:, -1])
        all_items = len(self)
        percentage = 1.0 * all_pos / all_items
        print("There are %s positive examples" % percentage)
        print("Total number of examples is %s", all_items)


if __name__ == "__main__":
    transformed_dataset = FinalDataset("train.npy", transform=ToTensor())
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
