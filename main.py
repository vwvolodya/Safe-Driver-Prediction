import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

data = pd.read_csv("data/train.csv")

cat1 = "ind"
cat2 = "reg"
cat3 = "car"
cat4 = "calc"

train, other = train_test_split(data, test_size=0.2, random_state=101101)
validation, test = train_test_split(other, test_size=0.5, random_state=10101)

names = data.columns.values.tolist()
# scaler = MinMaxScaler()
# test[['ps_ind_01', 'ps_ind_03']] = scaler.fit_transform(test[['ps_ind_01', 'ps_ind_03']])

filtered = [i for i in names if cat1 in i]
# print(data.describe())

def plot_corr(data):
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plot.get_figure().savefig("output.png")

def pca_func(data):
    pca = PCA(n_components=50)
    excluded = {"target", "id"}
    filterd = [i for i in names if i not in excluded]
    features = pd.DataFrame(data[filterd])
    pca.fit(features.as_matrix())
    pca_score = pca.explained_variance_ratio_
    V = pca.components_

    print("############## PCA score", pca_score)
    print("Features")
    print(V)


def experiments(d, cat, n):
    only = [i for i in names if cat in i and "cat" not in i]
    subset = d[only]
    only_cat = [i for i in d if "cat" in i and cat in i]
    new = pd.DataFrame(d[only_cat], dtype='object')
    categorical = pd.get_dummies(new)
    print(categorical.shape, subset.shape)

    pca = PCA(n_components=n, random_state=101010, svd_solver='full')
    pca.fit(categorical.as_matrix())
    dat = pca.transform(categorical.as_matrix())

    print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    return categorical


experiments(data, cat3, 32)
# experiments(data, cat2)
# experiments(data, cat3)
# experiments(data, cat4)
# pca_func(train)
# train.to_csv("data/for_train.csv")
# test.to_csv("data/for_test.csv")
# validation.to_csv("data/for_validation.csv")
