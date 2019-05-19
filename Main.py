from keras.callbacks import TensorBoard
from tensorflow.python.estimator import keras

import Test
from keras.wrappers.scikit_learn import KerasRegressor
from utils import model, report_value, get_data
from utils import cfg
from utils import plot_chart

# Tensorboard CallBack
# tensorCallBack = TensorBoard(log_dir='./graph',
#                              histogram_freq=1,
#                              write_graph=True,
#                              write_images=True)

# Get train & test data
X_train, X_test, y_train, y_test = get_data.get_all()

# Create model
xModel = KerasRegressor(build_fn=model.get_model)

# Fit model
history = xModel.fit(X_train, y_train,
                     epochs=cfg.get_epochs(),
                     batch_size=cfg.get_batch_size(),
                     verbose=cfg.get_verbose(),
                     validation_split=cfg.get_validation_split(),
                     shuffle=cfg.get_shuffle(),
                     callbacks=[])

# Save model as h5 files
# model.get_cnn_model().save('./h5/trained_model.h5')

# Predict and measure RMSE
pred = Test.check_preds(X_test, xModel)

if cfg.get_network_type() == "regression":
    plot_chart.plot_regression(pred.flatten(), y_test, history)

# Print the inputs and predicted outputs values
report_value.print_values(y_test, pred)
