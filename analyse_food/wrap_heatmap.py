"""

wrap heatmaps module ready for correlating output from heatmap and ssd modules

"""

#############################################################################
# Imports
#############################################################################

import matplotlib.pyplot as plt

plt.ion()

from scipy.misc import imread
import numpy as np
from keras.preprocessing import image as image_proc
from keras import backend as K

from keras.applications.resnet50 import ResNet50, preprocess_input
from heatmap import to_heatmap, synset_to_dfs_ids

#############################################################################
# Constants
#############################################################################

IMAGE_NET_DOG_CODE = "n02084071"

IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1280

#############################################################################
# Settings
#############################################################################

# image_filename = "/Users/colinrawlings/Desktop/htc18/refs/ssd_keras-1-master/caprica451/examples/bananas.jpg"

# image_filename = "/Users/colinrawlings/Desktop/htc18/caprica451/examples/banana_rotten.jpg"

image_filename = "/Users/colinrawlings/Desktop/htc18/caprica451/examples/banana_02.jpg"

# image_filename = "/Users/colinrawlings/Desktop/htc18/caprica451/examples/banana_pair.jpg"

object_name = "banana"

threshold = 0.75


#############################################################################
# definitions
#############################################################################

def calculate_heatmap(original_image, new_model, image, ids, preprocessing=None):
    # The quality is reduced.
    # If you have more than 8GB of RAM, you can try to increase it.

    from PIL import Image
    import numpy as np

    x = image_proc.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    if preprocessing is not None:
        x = preprocess_input(x)

    print("prediction starting")

    out = new_model.predict(x)

    print("prediction finished")

    heatmap = out[0]  # Removing batch axis.

    if K.image_data_format() == 'channels_first':
        heatmap = heatmap[ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=0)
    else:
        heatmap = heatmap[:, :, ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=2)

    #  resize back to original dimensions

    pil_heatmap = Image.fromarray(heatmap)

    resized_pil_heatmap = pil_heatmap.resize((original_image.shape[1], original_image.shape[0]),
                                             Image.BICUBIC)

    resized_np_heatmap = np.array(resized_pil_heatmap)

    return resized_np_heatmap


#############################################################################

def analyse_heatmap(threshold, original_image, heatmap):
    """
    :param original_image
    :param heatmap:
    :return: uint8 numpy.array masked_image, mask
    """

    mask = heatmap > threshold

    np_image = np.array(original_image)

    masked_np_image = np.zeros(np.shape(np_image))

    for channel in range(3):
        masked_np_image[:, :, channel] = mask * np_image[:, :, channel]

    masked_np_image = np.asarray(masked_np_image, dtype=np.uint8)

    return masked_np_image, mask


#############################################################################

def display_graphical_results(original_image, heatmap, masked_image):
    """

    :param original_image:
    :param heatmap:
    :param masked_image:
    :return: fig, axs
    """

    fig, axs = plt.subplots(3, 1)

    axs[0].imshow(original_image, interpolation="none")
    axs[0].contour(heatmap, [threshold, 1.1])
    axs[1].imshow(heatmap, interpolation="none")
    axs[1].contour(heatmap, [threshold, 1.1])
    axs[2].imshow(masked_image, interpolation="none")

    return fig, axs


#############################################################################

def calc_masked_image(image_filename, object_name, mask_threshold=0.5):
    """

    :param image_filename:
    :param object_name:
    :param mask_threshold:
    :return: masked_image, mask, fig, axs
    """

    from heatmap.imagenet1000_clsid_to_human import get_imagenet_classes_from_names

    class_ids = get_imagenet_classes_from_names()

    # model

    class_id = class_ids[object_name]
    model = ResNet50()
    new_model = to_heatmap(model)

    # calc

    original_image = imread(image_filename)
    image = image_proc.load_img(image_filename, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    heatmap = calculate_heatmap(original_image, new_model, image, class_id, preprocess_input)

    #

    masked_image, mask = analyse_heatmap(threshold, original_image, heatmap)
    fig, axs = display_graphical_results(original_image, heatmap, masked_image)

    return masked_image, mask, fig, axs


#############################################################################
# main
#############################################################################

if __name__ == "__main__":
    masked_image, mask, fig, axs =  calc_masked_image(image_filename,
                                                      object_name,
                                                      mask_threshold=threshold)
