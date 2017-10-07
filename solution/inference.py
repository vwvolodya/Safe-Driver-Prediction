from solution.model import DriverClassifier, DriverDataset, ToTensor


test_dataset = DriverDataset("../data/for_train.csv", transform=ToTensor())
net = DriverClassifier.load("models/model_1.mdl")

