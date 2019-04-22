from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import backend as k
from tensorflow.python.estimator import keras

from utils import cfg


def get_cnn_model():
    model = Sequential()
    k.set_image_dim_ordering('th')

    model.add(Conv2D(30, (3, 3), padding='valid', input_shape=(1, 25, 25), activation='relu'))

    model.add(Conv2D(30, (3, 3), padding='valid', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(15, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(32, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(128, activation='relu'))

    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(Dense(1, activation='linear'))

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
