"""Demo for the Mask R-CNN."""

# Copyright (c) 2017 Matterport, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from copy import deepcopy
import os
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import skimage.io

# Root directory of the project
global ROOT_DIR
working_path = os.path.dirname(os.path.realpath(__file__))
parent_path = Path(working_path).parent
correct_path = Path(parent_path).parent.absolute()
ROOT_DIR = os.path.join(correct_path, 'Mask_RCNN')

# Import Mask RCNN
# This import statements are directory specific, so we add a comment to tell
# the flake8 test to ignore the error 'E402 module level import not at top of file'
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils  # noqa: E402
import mrcnn.model as modellib  # noqa: E402
from mrcnn import visualize  # noqa: E402
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))  # To find local version
import coco  # noqa: E402


class InferenceConfig(coco.CocoConfig):

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5


class MaskRCNN_detectron():
    """Detection class to separate model initialization from detection."""

    def __init__(self):
        """Load the model."""
        global ROOT_DIR
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

        # Local path to trained weights file
        # working_path = os.path.dirname(os.path.realpath(__file__))
        # parent_path = Path(working_path).parent.absolute()
        COCO_MODEL_PATH = os.path.join(parent_path, 'models/mask_rcnn_coco.h5')
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        config = InferenceConfig()
        # config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # ## Class Names
        #
        # The model classifies objects returning class IDs, integer values identifying each class.
        # Some datasets assign integer values to their classes and some don't.
        # For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88.
        # The IDs are often sequential, but not always.
        # The COCO dataset has classes associated with class IDs 70 and 72, but not 71.
        #
        # To improve consistency and support training on data from multiple sources,
        # our ```Dataset``` class assigns it's own sequential integer IDs to each class.
        # Example: if you load the COCO dataset using our ```Dataset``` class, the 'person' class
        # would get class ID=1 (like COCO) and the 'teddy bear' class is 78 (different from COCO).
        # Keep that in mind when mapping class IDs to class names.
        #
        # To get the list of class names, load the dataset and use ```class_names``` property:
        #
        # # Load COCO dataset
        # dataset = coco.CocoDataset()
        # dataset.load_coco(COCO_DIR, "train")
        # dataset.prepare()
        #
        # # Print class names
        # print(dataset.class_names)
        #
        # We don't want to require you to download the COCO dataset just to run this demo,
        # so we're including the list of class names below.
        # The index of the class name in the list represent its ID.

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = [
            'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect(
        self,
        target_image: np.ndarray,
        load_image: str = ''
    ) -> Dict:
        """
        Compute the object masks on the image.

        Args
        ----
            target_image: the input image to the model.
            load_image: path to a stored image if to run offline.

        If a path is given, then load_image overrides target_image.

        Returns
        -------
            r: result dictionary

        """
        # Load a random image from the images folder
        if load_image != '':
            target_image = skimage.io.imread(load_image)

        # Run detection
        results = self.model.detect([target_image], verbose=0)

        # Visualize results
        r = results[0]

        # Remove table from detection
        # dining table is item 61
        rmv_class = self.class_names.index('dining table')
        # TODO: do the same for class 'person'
        rmv_index = np.where(r['class_ids'] == rmv_class)
        r['class_ids'] = np.delete(r['class_ids'], rmv_index)
        r['rois'] = np.delete(r['rois'], rmv_index, axis=0)

        r['masks'] = np.delete(r['masks'], rmv_index, axis=2)
        r['scores'] = np.delete(r['scores'], rmv_index)

        names = []
        for idx in r['class_ids']:
            names.append(self.class_names[idx])
        r['names'] = names

        # modify BB to have same convention as gradCAM
        for i, bb in enumerate(r['rois']):
            new_bb = [bb[1], bb[0], bb[3], bb[2]]
            r['rois'][i] = new_bb

        return r

    def print_results_on_image(
        self,
        np_image: np.ndarray,
        detection_dict: Dict,
        axes: Any = None
    ):
        """Print the detection instances on top of the given image."""
        results = deepcopy(detection_dict)
        # modify BB to have maskRCNN convention
        for i, bb in enumerate(results['rois']):
            new_bb = [bb[1], bb[0], bb[3], bb[2]]
            results['rois'][i] = new_bb

        colors = visualize.random_colors(len(self.class_names))
        visualize.display_instances(
            np_image,
            results['rois'],
            results['masks'],
            results['class_ids'],
            self.class_names,
            results['scores'],
            ax=axes,
            colors=colors
        )
