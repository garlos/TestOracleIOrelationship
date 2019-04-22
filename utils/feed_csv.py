import os
import numpy as np
from PIL import Image
import pandas as pd


def csv_feeder():
    for f in os.listdir('./img'):
        if f.endswith('.jpg'):
            fln = f.split('j', 1)[0]
            final = img_lbl_to_dataframe('./img/' + f, float(fln[0:len(fln) - 1]))
            print(final)
            with open("./input/dataset.csv", 'a') as file:
                final.to_csv(file, header=False)


def img_lbl_to_dataframe(img, label):
    image = Image.open(img)
    image = image.resize((25, 25), Image.NEAREST)
    image.load()
    img_data = np.asarray(image, dtype="uint8")
    img_data = img_data.mean(axis=2)

    img_dframe = pd.DataFrame(img_data)
    # print(img_dframe)
    img_dframe = pd.DataFrame(img_dframe.values.reshape(1, -1))
    # print(img_dframe)
    img_dframe.insert(0, 'label', label)

    return img_dframe
