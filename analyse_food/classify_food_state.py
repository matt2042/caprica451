import matplotlib

matplotlib.use("macosx")

import matplotlib.pyplot as plt

plt.ion()

import wrap_ssd as ws
import wrap_heatmap as wh

import python_helpers as yh

from scipy.misc import imread
import numpy as np

#############################################################################
# Settings
#############################################################################

# image_filename = "/Users/colinrawlings/Desktop/htc18/refs/ssd_keras-1-master/examples/fish-bike.jpg"
# image_filename = "/Users/colinrawlings/Desktop/htc18/refs/ssd_keras-1-master/examples/bananas.jpg"
# image_filename = "/Users/colinrawlings/Desktop/htc18/examples/banana_pair.jpg"
image_filename = "/Users/colinrawlings/Desktop/htc18/caprica451/examples/banana_rotten.jpg"
image_filename = "/Users/colinrawlings/Desktop/htc18/caprica451/examples/banana_02.jpg"



dataset = "COCO"

ssd_threshold = 0.45
mask_threshold = 0.4

#############################################################################
# Calc
#############################################################################

original_image = imread(image_filename)

ssd_results, log, fig, ax = ws.analyse_image(image_filename, dataset, ssd_threshold)

masked_image, mask, fig, axs = wh.calc_masked_image(image_filename,
                                                    ssd_results[0]["name"],
                                                    mask_threshold=mask_threshold)

keras_results = dict(ssd_results=ssd_results,
                     masked_image=masked_image,
                     mask=mask)

yh.save_var("tmp.pickle", keras_results)

keras_results = yh.load_var("tmp.pickle")

ssd_result = keras_results["ssd_results"][0]
masked_image_rgb = np.asarray(keras_results["masked_image"], dtype=float)
mask = np.asarray(keras_results["mask"], dtype=float)


# apply bbox mask

if int(ssd_result["xmin"]) < 0:
    ssd_result["xmin"] = 0

if int(ssd_result["ymin"]) < 0:
    ssd_result["ymin"] = 0

bbox_mask = np.zeros((original_image.shape[0], original_image.shape[1]))
bbox_mask[int(ssd_result["ymin"]):int(ssd_result["ymax"]),
int(ssd_result["xmin"]):int(ssd_result["xmax"])] = 1

for p in range(3):
    masked_image_rgb[:, :, p] = masked_image_rgb[:, :, p] * bbox_mask / 255


plt.imshow(masked_image_rgb)

#
# classify
#

masked_image_hsv = matplotlib.colors.rgb_to_hsv(masked_image_rgb)

fig, axs = plt.subplots(1, 3)

for p in range(3):
    combined_mask = bbox_mask*mask
    axs[p].hist(masked_image_hsv[:, :, p].ravel(), 50, weights=combined_mask.ravel())
