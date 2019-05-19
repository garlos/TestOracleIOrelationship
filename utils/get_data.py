import pandas

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from utils import cfg
from sklearn import preprocessing


def get_all():
    # Import DataSet
    dataset = pd.read_csv(cfg.get_dset_path()).values
    print("dataSet shape: {}".format(dataset.shape))

    # Normalize dataSet
    rawX = dataset[:, 1:5].astype('int32')
    scalerX = preprocessing.StandardScaler().fit(rawX)
    x_scaled = scalerX.fit_transform(rawX)
    print("x_scaled shape: {}".format(x_scaled.shape))
    print("finalX: {}".format(x_scaled))

    rawY = dataset[:, 5].astype('int32')
    y_scaled = rawY / rawY.max()
    print("y_scaled shape: {}".format(y_scaled.shape))
    print("finalY: {}".format(rawY))

    # Split dataSet into train/test
    X_train, X_test, y_train, y_test = train_test_split(x_scaled,
                                                        y_scaled,
                                                        test_size=cfg.get_test_size(),
                                                        random_state=cfg.get_random_state())
    return X_train, X_test, y_train, y_test

# get_all()
