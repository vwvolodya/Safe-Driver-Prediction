import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

cat1 = "ind"
cat2 = "reg"
cat3 = "car"
cat4 = "calc"


def func(data):
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
        "ps_car_11_cat",  # this feature has the most categories. ( > 100 )
    ]
    scaling_car = [
        "ps_car_11", "ps_car_15"
    ]
    # total 16 features / 5 numeric 11 categorical
    scaling_calc = [
        "ps_calc_04", "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08", "ps_calc_09", "ps_calc_10",
        "ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14"
    ]
    slice1 = data[categorical_ind]
    print(slice1.describe())
    print("\n")
    slice2 = data[categorical_car]
    print(slice2.describe())


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


def pca_func(data, names):
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


def experiments(d, cat, n, names):
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


def prepare(path, is_train=True, top=None):
    names = "id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin," \
                 "ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin," \
                 "ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01," \
                 "ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat," \
                 "ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11," \
                 "ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04," \
                 "ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12," \
                 "ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin," \
                 "ps_calc_19_bin,ps_calc_20_bin".split(",")
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
        "ps_car_11_cat",  # this feature has the most categories. ( > 100 )
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

    mapping = {name: i + 1 for i, name in enumerate(names)}
    target_column = "target"
    exclude_columns = {"id", "target", "ps_ind_14"}
    category_columns = [i for i in names if "cat" in i and i not in exclude_columns]
    binary_columns = [i for i in names if "bin" in i and i not in exclude_columns]
    columns_for_scaling = set(names) - exclude_columns - set(category_columns) - \
                               set(binary_columns)
    columns_for_scaling = list(columns_for_scaling)
    columns_for_scaling.sort()

    data = pd.read_csv(path)
    print("Original", data.shape)
    ids = data["id"].as_matrix()
    if is_train:
        magic_multiplier = 8  # 26 is because we have 3.5 % of true labels and we want wo make dataset balanced
        # augment data to change balance.
        true_rows = data[target_column] == 1
        slice = data[true_rows]
        data = data.append([slice] * magic_multiplier, ignore_index=True)
        data = data.sample(frac=1)

    y = data[target_column].as_matrix()

    new = pd.DataFrame(data[category_columns], dtype='object')
    categorical = pd.get_dummies(new)
    scaled = data[columns_for_scaling]
    if isinstance(scaled, pd.DataFrame):
        scaled_matrix = scaled.as_matrix()
    else:
        scaled_matrix = scaled
    binary = data[binary_columns]
    categorical_matrix = categorical.as_matrix()
    binary_matrix = binary.as_matrix()
    x = np.column_stack((scaled_matrix, binary_matrix, categorical_matrix, y))
    if top:
        x = x[:top, :]
    num_features = x.shape[1]
    shape = x.shape
    print(shape)
    result = pd.DataFrame(x)
    result.to_csv("../data/processed.csv")


def preprocess(path):
    names = "id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin," \
            "ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin," \
            "ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01," \
            "ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat," \
            "ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11," \
            "ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04," \
            "ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12," \
            "ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin," \
            "ps_calc_19_bin,ps_calc_20_bin".split(",")
    categorical = [i for i in names if "cat" in i]
    types = {i:"object" for i in categorical}
    data = pd.read_csv(path, dtype=types)
    new_ = pd.get_dummies(data)
    new_.to_csv("../data/prediction/one-hot-test.csv", index=False)


if __name__ == "__main__":
    # prepare("../data/tf_idf_all.csv", is_train=False)
    data = pd.read_csv("../data/one-hot-train.csv")
    train, other = train_test_split(data, test_size=0.1, random_state=101101)
    train.to_csv("../data/for_train.csv", index=False)
    other.to_csv("../data/for_test.csv", index=False)
    # func(data)
# train, other = train_test_split(data, test_size=0.2, random_state=101101)
# validation, test = train_test_split(other, test_size=0.5, random_state=10101)
#
# names = data.columns.values.tolist()
# scaler = MinMaxScaler()
# test[['ps_ind_01', 'ps_ind_03']] = scaler.fit_transform(test[['ps_ind_01', 'ps_ind_03']])

# filtered = [i for i in names if cat1 in i]
# print(data.describe())
# experiments(data, cat3, 32)
# experiments(data, cat2)
# experiments(data, cat3)
# experiments(data, cat4)
# pca_func(train)
# train.to_csv("data/for_train.csv")
# test.to_csv("data/for_test.csv")
# validation.to_csv("data/for_validation.csv")
