"""Test routine for the process detection function."""

# Copyright (c) 2021 Matteo Iovino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
from copy import copy

import numpy as np
import perception_utils.process_detection as det


def test_disambiguate_items():
    names = ['banana', 'knife', 'bottle', 'fork', 'banana']
    class_ids = [47, 44, 40, 43, 47]
    rois = [
        [100, 450, 300, 620],
        [250, 50, 320, 400],
        [570, 580, 650, 780],
        [610, 50, 700, 300],
        [850, 160, 1000, 350]
    ]
    # Masks are completely invented values
    # Remeber that masks are bool matrices with dimension M*N
    # where M is image pixels on X axis and N image pixels on Y axis.

    masks = np.array([
        [[False, False, False], [False, False, False], [False, False, True]],
        [[False, False, False], [False, False, False], [False, True, False]],
        [[False, False, False], [False, False, False], [True, False, False]],
        [[False, False, False], [False, False, True], [False, False, False]],
        [[False, False, False], [False, True, False], [False, False, False]]
    ])
    masks = np.reshape(masks, (3, 3, 5))

    detection_result = {
        'names': names,
        'class_ids': class_ids,
        'rois': rois,
        'masks': masks
    }

    first_detection = True
    detected_objects = {}
    objects_list = []

    # First detection
    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['banana_1', 'knife', 'bottle', 'fork', 'banana_2']
    assert first_detection is False
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['banana_1']['unique'] is False
    assert detected_objects['banana_2']['unique'] is False
    assert detected_objects['knife']['unique'] is True
    assert detected_objects['bottle']['category'] == 'bottle'
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][3]
    assert np.allclose(detected_objects['banana_1']['mask'], detection_result['masks'][:, :, 0])
    assert objects_list == names

    # Second detection --> Nothing should change
    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    assert first_detection is False
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['banana_1']['unique'] is False
    assert detected_objects['banana_2']['unique'] is False
    assert detected_objects['knife']['unique'] is True
    assert detected_objects['bottle']['category'] == 'bottle'
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][3]
    assert detected_objects['banana_1']['bounding_box'] == detection_result['rois'][0]
    assert detected_objects['banana_2']['bounding_box'] == detection_result['rois'][4]
    assert np.allclose(detected_objects['banana_1']['mask'], detection_result['masks'][:, :, 0])
    assert objects_list == names

    # Third detection --> Let's assume the banana 'banana_1' has been disambiguated
    detected_objects['banana'] = copy(detected_objects['banana_1'])
    detected_objects['banana'].__setitem__('unique', True)
    del detected_objects['banana_1']
    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['knife', 'bottle', 'fork', 'banana_2', 'banana']
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['banana']['unique'] is True
    assert detected_objects['banana_2']['unique'] is False
    assert detected_objects['banana']['category'] == 'banana'
    assert detected_objects['banana']['bounding_box'] == detection_result['rois'][0]
    assert detected_objects['knife']['bounding_box'] == detection_result['rois'][1]
    assert detected_objects['bottle']['bounding_box'] == detection_result['rois'][2]
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][3]
    assert detected_objects['banana_2']['bounding_box'] == detection_result['rois'][4]
    assert np.allclose(detected_objects['banana']['mask'], detection_result['masks'][:, :, 0])

    # Now we do the same but let's assume that banana_2 has been disambiguated
    # Reset all the variables, but skip tests for first detection
    first_detection = True
    detected_objects = {}
    objects_list = []

    # First detection
    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    detected_objects['banana'] = copy(detected_objects['banana_2'])
    detected_objects['banana'].__setitem__('unique', True)
    del detected_objects['banana_2']
    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['banana_1', 'knife', 'bottle', 'fork', 'banana']
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['banana']['unique'] is True
    assert detected_objects['banana_1']['unique'] is False
    assert detected_objects['banana']['category'] == 'banana'
    assert detected_objects['banana']['bounding_box'] == detection_result['rois'][4]
    assert detected_objects['knife']['bounding_box'] == detection_result['rois'][1]
    assert detected_objects['bottle']['bounding_box'] == detection_result['rois'][2]
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][3]
    assert detected_objects['banana_1']['bounding_box'] == detection_result['rois'][0]
    assert np.allclose(detected_objects['banana']['mask'], detection_result['masks'][:, :, 4])


def test_remove_items():
    names = ['banana', 'knife', 'bottle', 'fork', 'banana']
    class_ids = [47, 44, 40, 43, 47]
    rois = [
        [100, 450, 300, 620],
        [250, 50, 320, 400],
        [570, 580, 650, 780],
        [610, 50, 700, 300],
        [850, 160, 1000, 350]
    ]
    # Masks are completely invented values
    # Remeber that masks are bool matrices with dimension M*N
    # where M is image pixels on X axis and N image pixels on Y axis.

    masks = np.array([
        [[False, False, False], [False, False, False], [False, False, True]],
        [[False, False, False], [False, False, False], [False, True, False]],
        [[False, False, False], [False, False, False], [True, False, False]],
        [[False, False, False], [False, False, True], [False, False, False]],
        [[False, False, False], [False, True, False], [False, False, False]]
    ])
    masks = np.reshape(masks, (3, 3, 5))

    detection_result = {
        'names': names,
        'class_ids': class_ids,
        'rois': rois,
        'masks': masks
    }
    # Now we do the same but let's assume that banana_1 has been removed after first detection
    # Reset all the variables, but skip tests for first detection
    first_detection = True
    detected_objects = {}
    objects_list = []

    # First detection
    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    # Remove 1 banana from the detection results
    names = ['knife', 'bottle', 'fork', 'banana']
    class_ids = [44, 40, 43, 47]
    rois = [
        [250, 50, 320, 400],
        [570, 580, 650, 780],
        [610, 50, 700, 300],
        [850, 160, 1000, 350]
    ]
    # Masks are completely invented values
    # Remeber that masks are bool matrices with dimension M*N
    # where M is image pixels on X axis and N image pixels on Y axis.

    masks = np.array([
        [[False, False, False], [False, False, False], [False, True, False]],
        [[False, False, False], [False, False, False], [True, False, False]],
        [[False, False, False], [False, False, True], [False, False, False]],
        [[False, False, False], [False, True, False], [False, False, False]]
    ])
    masks = np.reshape(masks, (3, 3, 4))

    detection_result = {
        'names': names,
        'class_ids': class_ids,
        'rois': rois,
        'masks': masks
    }

    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['knife', 'bottle', 'fork', 'banana']
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['banana']['unique'] is True
    assert detected_objects['banana']['category'] == 'banana'
    assert detected_objects['banana']['bounding_box'] == detection_result['rois'][3]
    assert detected_objects['knife']['bounding_box'] == detection_result['rois'][0]
    assert detected_objects['bottle']['bounding_box'] == detection_result['rois'][1]
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][2]
    assert np.allclose(detected_objects['banana']['mask'], detection_result['masks'][:, :, 3])

    # Now we remove another item
    names = ['knife', 'bottle', 'fork']
    class_ids = [44, 40, 43]
    rois = [
        [250, 50, 320, 400],
        [570, 580, 650, 780],
        [610, 50, 700, 300]
    ]
    # Masks are completely invented values
    # Remeber that masks are bool matrices with dimension M*N
    # where M is image pixels on X axis and N image pixels on Y axis.

    masks = np.array([
        [[False, False, False], [False, False, False], [False, True, False]],
        [[False, False, False], [False, False, False], [True, False, False]],
        [[False, False, False], [False, False, True], [False, False, False]]
    ])
    masks = np.reshape(masks, (3, 3, 3))

    detection_result = {
        'names': names,
        'class_ids': class_ids,
        'rois': rois,
        'masks': masks
    }

    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['knife', 'bottle', 'fork']
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['knife']['bounding_box'] == detection_result['rois'][0]
    assert detected_objects['bottle']['bounding_box'] == detection_result['rois'][1]
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][2]


def test_add_items():
    names = ['knife', 'bottle', 'fork']
    class_ids = [44, 40, 43]
    rois = [
        [250, 50, 320, 400],
        [570, 580, 650, 780],
        [610, 50, 700, 300]
    ]
    # Masks are completely invented values
    # Remeber that masks are bool matrices with dimension M*N
    # where M is image pixels on X axis and N image pixels on Y axis.

    masks = np.array([
        [[False, False, False], [False, False, False], [False, True, False]],
        [[False, False, False], [False, False, False], [True, False, False]],
        [[False, False, False], [False, False, True], [False, False, False]]
    ])
    masks = np.reshape(masks, (3, 3, 3))

    detection_result = {
        'names': names,
        'class_ids': class_ids,
        'rois': rois,
        'masks': masks
    }

    first_detection = True
    detected_objects = {}
    objects_list = []

    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['knife', 'bottle', 'fork']
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['knife']['bounding_box'] == detection_result['rois'][0]
    assert detected_objects['bottle']['bounding_box'] == detection_result['rois'][1]
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][2]

    # Add 1 banana from the detection results
    names = ['knife', 'bottle', 'fork', 'banana']
    class_ids = [44, 40, 43, 47]
    rois = [
        [250, 50, 320, 400],
        [570, 580, 650, 780],
        [610, 50, 700, 300],
        [850, 160, 1000, 350]
    ]
    # Masks are completely invented values
    # Remeber that masks are bool matrices with dimension M*N
    # where M is image pixels on X axis and N image pixels on Y axis.

    masks = np.array([
        [[False, False, False], [False, False, False], [False, True, False]],
        [[False, False, False], [False, False, False], [True, False, False]],
        [[False, False, False], [False, False, True], [False, False, False]],
        [[False, False, False], [False, True, False], [False, False, False]]
    ])
    masks = np.reshape(masks, (3, 3, 4))

    detection_result = {
        'names': names,
        'class_ids': class_ids,
        'rois': rois,
        'masks': masks
    }

    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['knife', 'bottle', 'fork', 'banana']
    assert list(detected_objects.keys()) == oracle_keys
    assert detected_objects['banana']['unique'] is True
    assert detected_objects['banana']['category'] == 'banana'
    assert detected_objects['banana']['bounding_box'] == detection_result['rois'][3]
    assert detected_objects['knife']['bounding_box'] == detection_result['rois'][0]
    assert detected_objects['bottle']['bounding_box'] == detection_result['rois'][1]
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][2]
    assert np.allclose(detected_objects['banana']['mask'], detection_result['masks'][:, :, 3])

    # Now add another banana
    names = ['banana', 'knife', 'bottle', 'fork', 'banana']
    class_ids = [47, 44, 40, 43, 47]
    rois = [
        [100, 450, 300, 620],
        [250, 50, 320, 400],
        [570, 580, 650, 780],
        [610, 50, 700, 300],
        [850, 160, 1000, 350]
    ]
    # Masks are completely invented values
    # Remeber that masks are bool matrices with dimension M*N
    # where M is image pixels on X axis and N image pixels on Y axis.

    masks = np.array([
        [[False, False, False], [False, False, False], [False, False, True]],
        [[False, False, False], [False, False, False], [False, True, False]],
        [[False, False, False], [False, False, False], [True, False, False]],
        [[False, False, False], [False, False, True], [False, False, False]],
        [[False, False, False], [False, True, False], [False, False, False]]
    ])
    masks = np.reshape(masks, (3, 3, 5))

    detection_result = {
        'names': names,
        'class_ids': class_ids,
        'rois': rois,
        'masks': masks
    }

    detected_objects, objects_list, first_detection = det.process_maskRCNN_results(
                                                        detection_result,
                                                        detected_objects,
                                                        objects_list,
                                                        first_detection
                                                      )

    oracle_keys = ['banana_1', 'knife', 'bottle', 'fork', 'banana_2']
    assert first_detection is False
    assert collections.Counter(oracle_keys) == collections.Counter(list(detected_objects.keys()))
    assert detected_objects['banana_1']['unique'] is False
    assert detected_objects['banana_2']['unique'] is False
    assert detected_objects['knife']['unique'] is True
    assert detected_objects['bottle']['category'] == 'bottle'
    assert detected_objects['fork']['bounding_box'] == detection_result['rois'][3]
    assert np.allclose(detected_objects['banana_1']['mask'], detection_result['masks'][:, :, 0])
    assert objects_list == names


def test_boundingbox_intersection():
    # test intersection
    bb1 = [100, 100, 500, 600]
    bb2 = [300, 300, 800, 1000]
    oracle_bb = [bb2[0], bb2[1], bb1[2], bb1[3]]
    oracle_area = (bb1[2] - bb2[0])*(bb1[3] - bb2[1])
    area, bb = det.boundingbox_intersection(bb1, bb2)
    assert area == oracle_area
    assert bb == oracle_bb

    # test NO intersection
    bb1 = [100, 100, 500, 600]
    bb2 = [700, 700, 1000, 1000]
    oracle_bb = []
    oracle_area = 0
    area, bb = det.boundingbox_intersection(bb1, bb2)
    assert area == oracle_area
    assert bb == oracle_bb
