"""
Call the disambiguate function.

This script can be used for standalone testing.
"""

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

import os
from pathlib import Path
import sys
from typing import List

from disambiguate import disambiguate
import numpy as np
from object_detection import maskRCNN
from perception_utils.process_detection import boundingbox_intersection


def compress_bool_array(np_array: np.ndarray) -> bool:
    """Return the AND between the items in the bool array."""
    result = True
    for item in np_array:
        result = result and bool(item)

    return bool(result)


def test_disambiguation(maskRCNN: bool = False):
    source_img = 'current_scene.jpg'
    if len(sys.argv) > 1:
        source_img = str(sys.argv[1])

    target_obj = 'banana'

    ambiguous, bounding_box, disambiguated_obj = disambiguate.disambiguate_scene(
        source_img, target_obj, verbal=False, output_imgs=True)

    print('Ambiguity solved? ' + str(ambiguous))
    print(f'The item {disambiguated_obj} is located in: {bounding_box}.')

    assert disambiguated_obj == target_obj
    # The 'right' banana is located aprox in [750, 262, 921, 392] in the image 'current_scene.jpg'
    # The 'left' banana is located aprox in [241, 288, 395, 448] in the image 'current_scene.jpg'
    oracle_right_box = [750, 262, 921, 392]
    oracle_left_box = [241, 288, 395, 448]
    box_array = np.array(bounding_box)
    # Tolerance in pixels
    bool_array_R = np.isclose(box_array, oracle_right_box, atol=10)
    bool_R = compress_bool_array(bool_array_R)
    bool_array_L = np.isclose(box_array, oracle_left_box, atol=10)
    bool_L = compress_bool_array(bool_array_L)
    assert bool_R is True or bool_L is True

    if maskRCNN:
        # get this file folder
        working_path = os.path.dirname(os.path.realpath(__file__))
        # get disambiguate folder
        parent_path = Path(working_path).parent.absolute()
        results_directory = os.path.join(parent_path, 'disambiguate/data')
        image_path = os.path.join(results_directory, source_img)
        mask_points = call_maskRCNN(disambiguated_obj, image_path, bounding_box)
        print(mask_points)


def call_maskRCNN(
    image_path: str,
    target_obj: str,
    bounding_box: List[int]
) -> List[List[bool]]:
    """Call MaskRCNN to get the masks."""
    target_image = np.array([])
    mask_pts = []

    if target_obj != 'object':
        print('Detecting with MaskRCNN')
        # then we have correspondence in maskRCNN
        highest_intersection_idx = -1
        highest_intersection = -1
        output_maskRCNN = maskRCNN.maskRCNN(target_image, image_path, display_results=True)
        for i, name in enumerate(output_maskRCNN['names']):
            if name == target_obj:
                intersection, _ = boundingbox_intersection(
                    bounding_box, output_maskRCNN['rois'][i])
                if intersection > highest_intersection:
                    highest_intersection_idx = i
                    highest_intersection = intersection

        mask = (output_maskRCNN['masks'][:, :, highest_intersection_idx])

        for y_idx, row in enumerate(mask):
            for x_idx, val in enumerate(row):
                if val:
                    mask_pts.append((x_idx, y_idx))

    return mask_pts


if __name__ == '__main__':
    test_disambiguation(maskRCNN=True)
