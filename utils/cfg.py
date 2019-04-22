# Optimizer config
learning_rate = 0.01
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

# Model config
loss_function = 'mse'
metrics = ['mae']
optimizer = 'adam'

# Data Set config
test_size = 0.05
random_state = 1

# Training config
epochs = 50
batch_size = 30
verbose = 1
validation_split = 0.2
shuffle = True

# Data set path
dset_path = "./input/dataset.csv"


# All getter & setter functions #

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
    return loss_function


def get_metrics():
    return metrics


def get_optimizer():
    return optimizer


def get_all_hypers():
    return get_epochs()


def get_dset_path():
    return dset_path