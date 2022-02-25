"""Test script for the Mask R-CNN."""

# Copyright (c) 2022, ABB
# All rights reserved.
#
# Redistribution and use in source and binary forms, with
# or without modification, are permitted provided that
# the following conditions are met:
#
#   * Redistributions of source code must retain the
#     above copyright notice, this list of conditions
#     and the following disclaimer.
#   * Redistributions in binary form must reproduce the
#     above copyright notice, this list of conditions
#     and the following disclaimer in the documentation
#     and/or other materials provided with the
#     distribution.
#   * Neither the name of ABB nor the names of its
#     contributors may be used to endorse or promote
#     products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from copy import copy
import os
from pathlib import Path

import numpy as np
from object_detection import maskRCNN
from object_detection.mask_centroid_heuristic import Parameters, compute_mask_frame
from PIL import Image, ImageDraw
import skimage.io


def compress_bool_array(np_array: np.ndarray) -> bool:
    """Return the AND between the items in the bool array."""
    result = True
    for item in np_array:
        result = result and bool(item)

    return bool(result)


def get_image_path(image_name: str) -> str:
    """Return the path to the test image."""
    # get this file folder 'test'
    working_path = os.path.dirname(os.path.realpath(__file__))
    # get object detection folder
    object_detection_path = Path(working_path).parent.absolute()
    # get object recognition folder
    object_recognition_path = Path(object_detection_path).parent.absolute()
    # get perception_layer folder
    perception_layer_path = Path(object_recognition_path).parent.absolute()
    # get bt_learning --> root folder
    root_path = Path(perception_layer_path).parent.absolute()
    data_directory = os.path.join(root_path, 'hri/disambiguate/disambiguate/data')
    image_path = os.path.join(data_directory, image_name)

    return image_path


def test_maskRCNN():
    source_img = 'current_scene.jpg'
    image_path = get_image_path(source_img)

    target_image = np.array([])

    detectron = maskRCNN.MaskRCNN_detectron()
    output_maskRCNN = detectron.detect(target_image, image_path)

    target_image = skimage.io.imread(image_path)
    detectron.print_results_on_image(target_image, output_maskRCNN)

    assert 'banana' in output_maskRCNN['names']
    assert 'fork' in output_maskRCNN['names']
    assert 'knife' in output_maskRCNN['names']

    oracle_banana_box1 = [247, 288, 393, 452]
    oracle_banana_box2 = [760, 257, 919, 393]

    for i, name in enumerate(output_maskRCNN['names']):
        if name == 'banana':
            box = np.array(output_maskRCNN['rois'][i])
            bool1 = np.isclose(box, oracle_banana_box1, atol=10)
            bool1 = compress_bool_array(bool1)
            bool2 = np.isclose(box, oracle_banana_box2, atol=10)
            bool2 = compress_bool_array(bool2)

            assert bool1 is True or bool2 is True


def test_maskRCNN_rbgTOdepth():
    source_img = 'color_img.jpg'
    image_path = get_image_path(source_img)

    target_image = np.array([])

    detectron = maskRCNN.MaskRCNN_detectron()
    output_maskRCNN = detectron.detect(target_image, image_path)

    target_image = skimage.io.imread(image_path)
    detectron.print_results_on_image(target_image, copy(output_maskRCNN))

    assert 'banana' in output_maskRCNN['names']

    params = Parameters()
    params.normalize = False
    params.debug = True
    params.use_max = True

    points = []
    rotations = []

    classes = output_maskRCNN['names']
    for i, name in enumerate(classes):
        box = list(output_maskRCNN['rois'][i])
        mask = output_maskRCNN['masks'][:, :, i]

        point, rotation = compute_mask_frame(classes, name, box, mask, params)
        points.append(point)
        rotations.append(rotation)
        print(name)
        print(point, rotation)

    pil_image = Image.open(image_path)
    draw = ImageDraw.Draw(pil_image)
    for i, point in enumerate(points):
        corner1 = (point[0] - 5, point[1] - 5)
        corner2 = (point[0] + 5, point[1] + 5)
        draw.ellipse((corner1, corner2), fill='black', width=5)

    target_img = 'color_img_hr.jpg'
    save_path = get_image_path(target_img)
    pil_image.save(save_path)


def test_mask_heuristic():
    # TODO: maybe the method is more optimal if we send in the normalized mask.
    # To normalize a mask it should suffice to take the values in Masks that are
    # within the Bounding Box, since every value in Masks is a pixel of the image.
    source_img = 'current_scene.jpg'
    image_path = get_image_path(source_img)

    target_image = np.array([])

    detectron = maskRCNN.MaskRCNN_detectron()
    output_maskRCNN = detectron.detect(target_image, image_path)

    params = Parameters()
    params.normalize = False
    params.debug = True
    params.use_max = True

    points = []
    rotations = []

    classes = output_maskRCNN['names']
    for i, name in enumerate(output_maskRCNN['names']):
        box = list(output_maskRCNN['rois'][i])
        mask = output_maskRCNN['masks'][:, :, i]

        point, rotation = compute_mask_frame(classes, name, box, mask, params)
        points.append(point)
        rotations.append(rotation)
        print(point, rotation)


    pil_image = Image.open(image_path)
    draw = ImageDraw.Draw(pil_image)
    for i, point in enumerate(points):
        corner1 = (point[0] - 5, point[1] - 5)
        corner2 = (point[0] + 5, point[1] + 5)
        draw.ellipse((corner1, corner2), fill='black', width=5)

    target_img = 'current_scene_hr.jpg'
    save_path = get_image_path(target_img)
    pil_image.save(save_path)
