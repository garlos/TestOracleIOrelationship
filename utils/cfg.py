# Supported network types
classification_network = 'classification'
regression_network = 'regression'

# Default network type
# You can config this variable to use 2 types of networks
network_type = regression_network

# Data set path
dset_path = "./input/DataSet.csv"

learning_rate = 0.05
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

# Model config
classification_loss_function = 'categorical_crossentropy'
regression_loss_function = 'mse'

metrics = ['mae']

classification_optimizer = 'rmsprop'
regression_optimizer = 'adam'

classification_last_layer = 'softmax'
regression_last_layer = 'linear'

# Data Set config
test_size = 0.1
random_state = 1

# Training config
epochs = 40
batch_size = 30
verbose = 1
validation_split = 0.2
shuffle = True


# All getter & setter functions #

def get_network_type():
    if network_type == classification_network:
        return classification_network
    elif network_type == regression_network:
        return regression_network


def get_epochs():
    return epochs


def get_batch_size():
    return batch_size


def get_verbose():
    return verbose


def get_validation_split():
    return validation_split


def get_shuffle():
    return shuffle


def get_test_size():
    return test_size


def get_random_state():
    return random_state


def get_lr():
    return learning_rate


def get_beta_1():
    return beta_1


def get_beta_2():
    return beta_2


def get_epsilon():
    return epsilon


def get_loss():
    if network_type == "classification":
        return classification_loss_function
    if network_type == "regression":
        return regression_loss_function


def get_last_layer():
    if network_type == "classification":
        return classification_last_layer
    if network_type == "regression":
        return regression_last_layer


def get_optimizer():
    if network_type == "classification":
        return classification_optimizer
    if network_type == "regression":
        return regression_optimizer


def get_metrics():
    return metrics


def get_all_hypers():
    return get_epochs()


def get_dset_path():
    return dset_path
