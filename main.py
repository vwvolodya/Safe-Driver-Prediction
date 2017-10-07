import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data/train.csv")
train, other = train_test_split(data, test_size=0.2, random_state=101101)
validation, test = train_test_split(other, test_size=0.5, random_state=10101)

names = train.columns.values.tolist()
scaler = MinMaxScaler()
# test[['ps_ind_01', 'ps_ind_03']] = scaler.fit_transform(test[['ps_ind_01', 'ps_ind_03']])

r = test.iloc[[1]]
print(data.describe())
print()
# train.to_csv("data/for_train.csv")
# test.to_csv("data/for_test.csv")
# validation.to_csv("data/for_validation.csv")
