import pandas as pd
import numpy as np
from sklearn.externals import joblib

clf = joblib.load("xg.mdl")
data = pd.read_csv("../data/prediction/one-hot-test.csv")
ids = data["id"].as_matrix()

x = np.load("../data/test.npy")

y_prob = clf.predict_proba(x)
y_prob = y_prob[:, 1]
new_df = pd.DataFrame(list(zip(ids, y_prob)), columns=["id", "target"])
new_df.to_csv("../data/prediction/predicted_xg.csv", index=False)
