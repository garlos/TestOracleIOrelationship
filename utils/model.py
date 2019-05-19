from keras import Sequential
from keras.layers import Dense, Dropout
from tensorflow.python.estimator import keras
from utils import get_data
from keras.utils import np_utils
from utils import cfg


def get_model():
    X_train, X_test, y_train, y_test = get_data.get_all()
    input_dim = X_train.shape[1]

    y_train = np_utils.to_categorical(y_train)
    nb_classes = y_train.shape[1]

    if cfg.get_network_type() == "classification":
        # Here's a Deep Dumb MLP (DDMLP)(classification)
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(nb_classes, activation=cfg.get_last_layer()))

        # Model optimizer
        keras.optimizers.rmsprop(lr=cfg.get_lr(),
                                 beta_1=cfg.get_beta_1(),
                                 beta_2=cfg.get_beta_2(),
                                 epsilon=cfg.get_epsilon())

        # Compile model
        model.compile(loss=cfg.get_loss(),
                      metrics=cfg.get_metrics(),
                      optimizer=cfg.get_optimizer())

        return model

    elif cfg.get_network_type() == "regression":
        # Here's a Deep Dumb MLP (DDMLP)(regression)
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(1, activation=cfg.get_last_layer()))

        # Model optimizer

        keras.optimizers.Adam(lr=cfg.get_lr(),
                              beta_1=cfg.get_beta_1(),
                              beta_2=cfg.get_beta_2(),
                              epsilon=cfg.get_epsilon())

        # Compile model
        model.compile(loss=cfg.get_loss(),
                      metrics=cfg.get_metrics(),
                      optimizer=cfg.get_optimizer())

        return model
