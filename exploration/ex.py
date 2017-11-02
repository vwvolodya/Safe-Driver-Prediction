from sklearn import metrics
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
import pandas as pd
import numpy as np


def load_data(path, negative_only=False, top=None):
    names = "id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin," \
            "ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin," \
            "ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01," \
            "ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat," \
            "ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11," \
            "ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04," \
            "ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12," \
            "ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin," \
            "ps_calc_19_bin,ps_calc_20_bin".split(",")
    exclude_columns = {"id", "target"}#, "ps_ind_14", "ps_car_11_cat"}

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
    category_columns = [i for i in names if "cat" in i and i not in exclude_columns]
    binary_columns = [i for i in names if "bin" in i]
    columns_for_scaling = list(set(names) - exclude_columns - set(category_columns) - \
                                    set(binary_columns))

    data = pd.read_csv(path)
    if negative_only:
        false_rows = data[target_column] != 1
        data = data[false_rows]
    scaled = data[columns_for_scaling]

    new = pd.DataFrame(data[category_columns], dtype='object')
    categorical = pd.get_dummies(new)
    binary = data[binary_columns]
    categorical_matrix = categorical.as_matrix()
    binary_matrix = binary.as_matrix()
    x = np.column_stack((categorical_matrix, binary_matrix, scaled.as_matrix()))
    # x = scaled.as_matrix()
    y = data[target_column].as_matrix()
    if top:
        x = x[:top, :]  # get only top N samples
        y = y[:top, :]
    num_features = x.shape[1]
    shape = x.shape
    print(shape)

    return x, y


if __name__ == "__main__":
    x, y = load_data("../data/for_train.csv", negative_only=True)
    val_x, val_y = load_data("../data/for_validation.csv")
    # clf = NearestCentroid(metric='cosine')
    # clf = svm.SVC(verbose=True)
    clf = svm.OneClassSVM(verbose=True)
    clf.fit(x)

    predicted_y = clf.predict(x)
    f1_train = metrics.f1_score(y, predicted_y)
    print(f1_train)

    predicted_y_val = clf.predict(val_x)
    f1_test = metrics.f1_score(val_y, predicted_y_val)
    print(f1_test)
