from solution.dataset import DriverDataset, ToTensor
from solution.model import DriverClassifier
from torch.utils.data import DataLoader


test_dataset = DriverDataset("../data/for_test.csv", transform=ToTensor())
net = DriverClassifier.load("../models/model_1.mdl")

dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

res = net.evaluate(dataloader)

print(res)
