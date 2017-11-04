import pandas as pd
import numpy as np
from collections import Counter


def transform_cat_to_tfidf(input_path, output_path, input_test, output_test):
    names = "id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin," \
            "ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin," \
            "ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01," \
            "ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat," \
            "ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11," \
            "ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04," \
            "ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12," \
            "ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin," \
            "ps_calc_19_bin,ps_calc_20_bin".split(",")
    category_columns = [i for i in names if "cat" in i]
    category_columns = {i: "object" for i in category_columns}
    df = pd.read_csv(input_path, dtype=category_columns)
    test_df = pd.read_csv(input_test, dtype=category_columns)
    for cat_col in category_columns:
        cnt = Counter(df[cat_col])
        res = dict(cnt.most_common())
        all = sum(res.values())
        res = {k: v/all for k, v in res.items()}
        func = lambda x: res[x]
        df[cat_col] = df[cat_col].apply(func)
        test_df[cat_col] = test_df[cat_col].apply(func)

        print()

    df.to_csv(output_path, index=False)
    test_df.to_csv(output_test, index=False)


if __name__ == "__main__":
    transform_cat_to_tfidf("../data/train.csv", "../data/tf_idf_all.csv",
                           "../data/prediction/test.csv", "../data/prediction/test_tf_idf.csv")
    print()
