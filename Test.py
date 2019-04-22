
def check_preds(x, trained_model):
    # trained_model = model.load
    # Predict and measure RMSE
    pred = trained_model.predict(x)
    # score = k.sqrt(metrics.mae(pred, y))
    # print("Score (RMSE): {}".format(score))
    # print(classification_report(y, to_categorical(predictions)))
    return pred
