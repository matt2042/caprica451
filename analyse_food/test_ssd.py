"""

copy across the notebook for clarity

"""

#############################################################################
# Imports
#############################################################################

from keras import backend as K

import keras.preprocessing.image as kg
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from ssd_keras_1_master.keras_ssd300 import ssd_300
from ssd_keras_1_master.keras_ssd_loss import SSDLoss
from ssd_keras_1_master.ssd_box_encode_decode_utils import decode_y, decode_y2

#############################################################################
# Constants
#############################################################################

IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300

COCO_WEIGHTS_FILENAME = "ssd_keras_1_master/VGG_coco_SSD_300x300_iter_400000.h5__"
VOC_WEIGHTS_FILENAME = ""

MS_COCO_ANNOTATIONS_FILENAME = 'ssd_keras_1_master/datasets/MicrosoftCOCO/annotations/COCO_classes_to_names.pickle'


#############################################################################
# Settings
#############################################################################

# image_filename = "/Users/colinrawlings/Desktop/htc18/refs/ssd_keras-1-master/examples/fish-bike.jpg"
# image_filename = "/Users/colinrawlings/Desktop/htc18/refs/ssd_keras-1-master/examples/apples.jpg"
# dataset = "COCO"


#############################################################################
# definitions
#############################################################################

def get_model(dataset):
    """

    :return: keras_ssd300.ssd300 model
    """

    if dataset == "VOC":
        ssd300_nclasses = 20
        ssd300_scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
        weights_filename = "/Users/colinrawlings/Desktop/htc18/refs/ssd_keras-1-master/VGG_VOC0712_SSD_300x300_iter_120000.h5"
    elif dataset == "COCO":
        ssd300_nclasses = 80
        ssd300_scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        weights_filename = "/Users/colinrawlings/Desktop/htc18/refs/ssd_keras-1-master/VGG_coco_SSD_300x300_iter_400000.h5"
    else:
        raise ValueError("Unrecognised dataset: {}".format(dataset))

    K.clear_session()  # Clear previous models from memory.

    model = ssd_300(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                    n_classes=ssd300_nclasses,
                    l2_regularization=0.0005,
                    scales=ssd300_scales,  # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    limit_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    coords='centroids',
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=True)

    # 2: Load the trained weights into the model.

    # TODO: Set the path of the trained weights.

    model.load_weights(weights_filename, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    return model


#############################################################################

def get_images(image_paths):
    """

    :param image_paths:
    :return:
    """

    original_images = list()  # Store the images here.
    input_images = list()  # Store resized versions of the images here.

    for image_path in image_paths:
        original_images.append(imread(image_path))
        image = kg.load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        image = kg.img_to_array(image)
        input_images.append(image)

    input_images = np.array(input_images)

    return original_images, input_images


#############################################################################

def detect_objects(input_images, model):
    """

    :param input_images:
    :param model:
    :return:
    """

    y_predicted = model.predict(input_images)
    y_predicted_decoded = decode_y(y_predicted,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   input_coords='centroids',
                                   normalize_coords=True,
                                   img_height=IMAGE_HEIGHT,
                                   img_width=IMAGE_WIDTH)

    return y_predicted_decoded


#############################################################################

def get_VOC_names_from_class():
    """

    :return: list()
    """

    classes_to_names = ['background',
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

    return classes_to_names


#############################################################################

def get_COCO_names_from_class():
    """

    :return: list()
    """

    import pickle

    return pickle.load(open(MS_COCO_ANNOTATIONS_FILENAME, 'r'))


#############################################################################

def display_text_results(dataset, original_images, results):
    """

    :param dataset: "COCO", "VOC"
    :param original_images:
    :param results:
    :return: None
    """

    if dataset == "VOC":
        names_from_classes = get_VOC_names_from_class()
    elif dataset == "COCO":
        names_from_classes = get_COCO_names_from_class()
    else:
        raise ValueError("Unrecognised dataset: {}".format(dataset))

    boxes = results[0]
    num_boxes = len(boxes)

    log = ""
    log += "Predicted boxes:\r\n"
    log += 'class\tconf\txmin\tymin\txmax\tymax\r\n'

    for box_number in range(num_boxes):
        box = boxes[box_number]
        class_id = int(box[0])

        name = names_from_classes[class_id]

        log += "{}\t{:.2f}\t{:.0f}\t\t{:.0f}\t\t{:.0f}\t\t{:.0f}\r\n".format(name, box[1], box[2], box[3], box[4], box[5])

    return log


#############################################################################

def display_graphical_results(dataset, original_images, results,
                              fig_width_in=5, fig_height_in=4):
    """

    :param dataset: "COCO", "VOC"
    :param original_images:
    :param results:
    :return:
    """

    if dataset == "VOC":
        names_from_classes = get_VOC_names_from_class()
    elif dataset == "COCO":
        names_from_classes = get_COCO_names_from_class()
    else:
        raise ValueError("Unrecognised dataset: {}".format(dataset))

    boxes = results[0]
    num_boxes = len(boxes)

    colors = plt.cm.hsv(np.linspace(0, 1, num_boxes)).tolist()

    #

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(fig_width_in, fig_height_in)

    ax.imshow(original_images[0])

    for box_number in range(num_boxes):
        box = boxes[box_number]
        class_id = int(box[0])

        name = names_from_classes[class_id]

        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[-4] * original_images[0].shape[1] / IMAGE_WIDTH
        ymin = box[-3] * original_images[0].shape[0] / IMAGE_HEIGHT
        xmax = box[-2] * original_images[0].shape[1] / IMAGE_WIDTH
        ymax = box[-1] * original_images[0].shape[0] / IMAGE_HEIGHT
        color = colors[0]
        label = '{}: {:.2f}'.format(name, box[1])
        ax.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        ax.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 0.65})

    return fig, ax


#############################################################################

def display_results(dataset,
                    original_images, results,
                    text_output=True, graphical_output=True):
    """

    :param dataset: "COCO", "VOC"
    :param original_images:
    :param results:
    :param text_output:
    :param graphical_output:
    :return:
    """

    log = display_text_results(dataset, original_images, results)
    if text_output:
        print(log)

    fig, ax = None, None
    if graphical_output:
        fig, ax = display_graphical_results(dataset, original_images, results)

    return log, fig, ax


#############################################################################

def analyse_image(image_filepath, dataset,
                  text_output=True, graphical_output=True):
    """

    :param image_filepath:
    :param dataset:
    :return:
    """

    model = get_model(dataset)

    original_images, input_images = get_images([image_filepath])

    results = detect_objects(input_images, model)

    log, fig, ax = display_results(dataset, original_images, results,
                                   text_output=text_output, graphical_output=graphical_output)

    return results, log, fig, ax


#############################################################################

if __name__ == "__main__":
    results, log, fig, ax = analyse_image(image_filename, dataset)
