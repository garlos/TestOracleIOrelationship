import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from utils import cfg


def get_all():
    # Import DataSet
    dataset = pd.read_csv(cfg.get_dset_path()).values
    # print("dataSet shape: {}".format(dataset.shape))

    # Reshape and normalize dataSet
    rawX = dataset[:, 2:].reshape(dataset.shape[0], 1, 25, 25).astype('float32')
    finalX = rawX / 255.0

    # 2755 is output range.....(max - min)=2755 # by dividing in 2755 we normalize output
    rawY = dataset[:, 1]
    finalY = rawY / 2755.0
    # print("finalY: {}".format(finalY))

    # Split dataSet into train/test
    X_train, X_test, y_train, y_test = train_test_split(finalX,
                                                        finalY,
                                                        test_size=cfg.get_test_size(),
                                                        random_state=cfg.get_random_state())
    return X_train, X_test, y_train, y_test
