from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/ssd")
def ssd_test():
    from keras import backend as K

    import keras.preprocessing.image as kg
    from keras.optimizers import Adam
    import scipy
    from scipy.misc import imread
    import numpy as np
    #from matplotlib import pyplot as plt

    from ssd_keras_1_master.keras_ssd300 import ssd_300
    from ssd_keras_1_master.keras_ssd_loss import SSDLoss
    from ssd_keras_1_master.ssd_box_encode_decode_utils import decode_y, decode_y2

    from coco_utils import get_coco_category_maps

    from analyse_food import test_ssd as ts
    reload(ts)

    image_filename = "./test_images/apples.jpg"
    dataset="COCO"

    results, log, fig, ax = ts.analyse_image(image_filename, dataset,
                                          False, False)

    return log


if __name__ == '__main__':
    app.run(debug=True)
    