import datetime
import matplotlib.pyplot as plt
import pandas as pd
from utils import cfg, model
from datetime import datetime


# Plot regression chart
def plot_regression(pred, y, hist, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')

    plt.text(0.2, 0.6, hyper_params(), fontsize=8)
    # plt.text(2100, 0.6, model_summary(), fontsize=4)

    # plt.text(30, 0.4, "validation_loss: {0}".format(hist.history['val_loss']), fontsize=8)
    # plt.text(30, 0.37, "validation_mae: {0}".format(hist.history['val_mean_absolute_error']), fontsize=8)
    plt.legend()
    plt.savefig("./report/{0}".format(datetime.now().strftime('%Y-%m-%d # %H-%M-%S')), dpi=150)
    plt.show()


# Return Hyper Parameters as string
def hyper_params():
    desc = ("Epochs:  " + str(cfg.get_epochs()) + "         " + "Lr:           " + str(cfg.get_lr()) + "\n" +
            "B_Size:   " + str(cfg.get_batch_size()) + "       " + "Beta_1:    " + str(cfg.get_beta_1()) + "\n" +
            "Verbose: " + str(cfg.get_verbose()) + "        " + "Beta_2:    " + str(cfg.get_beta_2()) + "\n" +
            "V_Split:   " + str(cfg.get_validation_split()) + "     " + "Epsilon:   " + str(cfg.get_epsilon()) + "\n" +
            "Shuffle:   " + str(cfg.get_shuffle()) + "\n" +
            "----------------------\n" +
            "Test_Size: " + str(cfg.get_test_size()) + "\n" +
            "Rnd_State: " + str(cfg.get_random_state()) + "\n" +
            "----------------------\n" +
            "Loss:        " + (cfg.get_loss()) + "\n" +
            "Metrics:    " + (','.join(cfg.get_metrics())) + "\n" +
            "Optimizer: " + (cfg.get_optimizer()) + "\n" +
            "----------------------\n")
    # "Data-set count: " + str(dset_count))
    return desc


# Return model summary as string
def model_summary():
    sumry = []
    model.get_model().summary(line_length=35, print_fn=lambda x: sumry.append(x))
    summary = "\n".join(sumry)
    return summary


# def metrics_val():
#     hist = model.get_cnn_model().history['val_loss']
#     return hist
