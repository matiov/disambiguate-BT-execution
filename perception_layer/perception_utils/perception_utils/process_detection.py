"""
Utility functions to process the results from detection algorithms.

Detection Algorithms supported:
- maskRCNN
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

import collections
from copy import copy
from typing import Dict, List, Tuple

import numpy as np


class ObjectData(object):
    """Fixed Dicitonary representing a detected object."""

    def __init__(self, dictionary):
        self._dictionary = dictionary

    def __setitem__(self, key, item):
        if key not in self._dictionary:
            raise KeyError('The key {} is not defined.'.format(key))
        self._dictionary[key] = item

    def __getitem__(self, key):
        return self._dictionary[key]

    def __tostring__(self):
        string_id = f'ID: {self._dictionary["id"]}\n'
        string_cat = f'Category: {self._dictionary["category"]}\n'
        string_bb = f'Bounding Box: {self._dictionary["bounding_box"]}\n'
        string_unique = f'Unique: {self._dictionary["unique"]}\n'

        string_dict = string_id + string_cat + string_bb + string_unique
        return string_dict


TEMPLATE_OBJ_DICT = {
    'id': 0,
    'category': '',
    'bounding_box': [0, 0, 0, 0],
    'mask': None,
    'unique': True
}


ACCEPTED_CLASSES = [
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors'
]


def process_maskRCNN_results(
    detection_result: Dict,
    detected_objects: Dict = {},
    objects_list: List[str] = [],
    first_detection: bool = False,
    only_update: bool = False
) -> Tuple[Dict, List[str], bool]:
    """
    Process the results form maskRCNN and update the dictionary of the objects.

    Args
    ----
        detection_result: result from the maskRCNN detection algorithm.
        detected_objects: dictionary storing the detected objects.
        objects_list: list storing the categories of the detected objects.
        first_detection: True if this is the first time detection is processed.
        only_update: Run only the update function not to add new objects.

    Returns
    -------
        The detected_objects dicitonary, the objects_list and an update of the first_detection.

    Notes on Data types
    -------------------
        The dictionary 'detection_result' has the following shape:

        - the keys are ['class_ids', 'rois', 'masks', 'scores' 'names'].
        - 'names' is a list with the names of the detected COCO classes with IDs in class_ids.
        - 'rois' is a list of bounding boxes.
        - 'masks' is a list of masks (Dim: PIXELS_X:PIXELS_Y:N).
        - 'scores' is a list of scores.
        --> the list has length N where N is the number of detected elements.

        The dictionary 'detected_objects' has the following shape:
        - the keys are the object categories OR a string concatenating obj category and obj ID.
        - the values are of type ObjectData.

        The dictionary of type ObjectData has the following shape:
        - the keys are ['id', 'category', 'bounding_box', 'mask', 'unique'].
        - 'id' is the ID of the object, 0 if there is only one object of that category.
        - 'category' is a string with the name of the COCO label for the object.
        - 'bounding_box' is an array defining the bounding box [Xmin, Ymin, XMAX, YMAX].
        - 'mask' is the a matrix of booleans with dimension the pixels in the image X, Y.
        - 'unique' is a boolean defining if the object is unique or not.
        --> 'unique' can be True after the disambiguation process!!

    """
    processed_keys = []
    processed_cats = []
    unprocessed_idx = int(1e10)

    # Check if there are the same objects and in the same number as before!
    equality = collections.Counter(objects_list) == collections.Counter(detection_result['names'])
    if not equality:
        print('The detected objects are no more the same!!!')

    objects_in_dict = []
    for dict_item in detected_objects.items():
        # items() returns a tuple (key, value)
        objects_in_dict.append(dict_item[1].__getitem__('category'))

    for i, object_class in enumerate(detection_result['names']):
        # 'Non ragioniam di lor ma, guarda e passa.' Inferno, chant III, verse 51
        if object_class not in ACCEPTED_CLASSES:
            continue
        # Some values can be set just the first time!
        elif first_detection or object_class not in objects_list:
            print(f'Adding class {object_class} to the dictionary.')
            detected_objects, processed_keys = __add_item_to_dict(
                detection_result,
                detected_objects,
                object_class,
                i,
                processed_keys,
                processed_cats
            )
            processed_cats.append(object_class)
        # Check for the case of one more item but from a class already added.
        elif not only_update and\
             (objects_list.count(object_class) < detection_result['names'].count(object_class) or\
             objects_in_dict.count(object_class) < detection_result['names'].count(object_class)):
            print(f'Adding item of class {object_class} to the dictionary.')
            detected_objects, processed_keys = __add_item_to_dict(
                detection_result,
                detected_objects,
                object_class,
                i,
                processed_keys,
                processed_cats
            )
            processed_cats.append(object_class)
            # If we add a new item of existing class, remove the entry for single item.
            # del detected_objects[object_class]
        else:
            # Otherwise we have the same objects as before, but they might have moved!
            # Therefore we need to update existing data, making sure it matches.
            # ---------------------------------------------------------------------------
            # Another option is that an item has been removed from the scene.
            # We will deal with this case later, but we still update information.
            detected_objects, processed_keys, match_found = __update_dictionary(
                object_class, i, detection_result, detected_objects, objects_list, processed_keys)
            if not match_found:
                # Add the index of the object in the unprocessed ones
                unprocessed_idx = i
            else:
                processed_cats.append(object_class)

    for stored_key in list(detected_objects.keys()):
        if stored_key in processed_keys:
            continue
        else:
            # Either an object has moved in the scene or removed from the scene.
            obj_to_update = detected_objects[stored_key]
            try:
                unprocessed_class = detection_result['names'][unprocessed_idx]
                is_unprocessed = obj_to_update.__getitem__('category') == unprocessed_class
            except IndexError:
                is_unprocessed = False
            # Control check
            if is_unprocessed:
                # The hypotesis is that 1 object can change every detection frame
                # So we can update without worrying too much!
                print(f'Updating information for unprocessed object {stored_key}')
                obj_to_update.__setitem__(
                    'bounding_box', detection_result['rois'][unprocessed_idx])
                obj_to_update.__setitem__(
                    'mask', detection_result['masks'][:, :, unprocessed_idx])
            else:
                del detected_objects[stored_key]

    if len(detected_objects.keys()) > 0:
        first_detection = False

    objects_list = copy(detection_result['names'])

    return detected_objects, objects_list, first_detection


def __add_item_to_dict(
    detection_result: Dict,
    detected_objects: Dict,
    obj_class: str,
    obj_idx: int,
    processed_keys: List[str],
    processed_cats: List[str]
) -> Tuple[Dict, List[str]]:
    """
    Add a new item to the dictionary of detected objects.

    Args
    ----
        detection_result: result from the maskRCNN detection algorithm.
        detected_objects: dictionary storing the detected objects.
        obj_class: the class of the object to match.
        obj_index: the index of the object to match in the detected items dictionary.
        processed_keys: the list of processed keys.
        processed_cats: the list of processed categories.

    Processed keys keeps track of the keys that have been processed.
    Since the key in the dictionary doesn't necessarily correspond to the class of the item,
    it is also required to keep track of wchich classes have been processed and how many times.
    This is the reason why processed categories is a parameter of this function.

    Returns
    -------
        The updated detected_objects dictionary and processed list.

    """
    occurrencies = detection_result['names'].count(obj_class)
    data_dictionary = copy(TEMPLATE_OBJ_DICT)
    object_data = ObjectData(data_dictionary)
    object_data.__setitem__('category', obj_class)
    object_data.__setitem__('bounding_box', detection_result['rois'][obj_idx])
    object_data.__setitem__('mask', detection_result['masks'][:, :, obj_idx])
    # Set IDS and uniqueness of objects.
    if occurrencies == 1:
        # The object is unique!
        object_data.__setitem__('id', 0)
        object_data.__setitem__('unique', True)
        object_name = obj_class
    else:
        object_data.__setitem__('unique', False)
        ambiguous_obj_id = processed_cats.count(obj_class) + 1
        object_data.__setitem__('id', ambiguous_obj_id)
        object_name = obj_class + '_' + str(ambiguous_obj_id)

    detected_objects[object_name] = object_data
    # print(f'Adding in key {object_name}:')
    # print(detected_objects[object_name].__tostring__())
    processed_keys.append(object_name)

    return detected_objects, processed_keys


def __update_dictionary(
    obj_class: str,
    obj_index: int,
    detection_result: Dict,
    detected_objects: Dict,
    objects_list: List[str],
    processed: List[str]
) -> Tuple[Dict, List[str], bool]:
    """
    Update the dictionary of detected objects.

    This helper function allows to update correctly the existing items
    with the values from the last detection iteration.
    Only those items whose class occurs more than once are checked for matching.

    Args
    ----
        obj_class: the class of the object to match.
        obj_index: the index of the object to match in the detected items dictionary.
        detection_result: result from the maskRCNN detection algorithm.
        detected_objects: dictionary storing the detected objects.
        objects_list: list of the detected categories.
        processed: the list of processed objects

    Returns
    -------
        The updated detected_objects dictionary and processed list.
        A boolean stating if a match has been found and hence update correct.

    """
    occurencies = objects_list.count(obj_class)
    possible_keys = []
    match_found = False
    if occurencies > 1:
        possible_keys = [str(obj_class + '_') + str(i + 1) for i in range(occurencies)]
    possible_keys.append(obj_class)
    print('Item(s) to update: ' + str(possible_keys))

    bb_from_detection = detection_result['rois'][obj_index]

    highest_intersection_key = ''
    highest_intersection = 0
    # match the bounding box
    for stored_key in possible_keys:
        try:
            bb_to_match = detected_objects[stored_key].__getitem__('bounding_box')
        except KeyError:
            print(f'(Updating.) Key {stored_key} not found, ignoring!')
            continue
        intersection, _ = boundingbox_intersection(bb_to_match, bb_from_detection)
        if intersection > highest_intersection:
            # if there is high intersection it might be that the BB are matching.
            # This means that the object hasn't moved much.
            highest_intersection = intersection
            highest_intersection_key = stored_key

    if highest_intersection > 0:
        # match found, so we can update
        obj_to_update = detected_objects[highest_intersection_key]
        obj_to_update.__setitem__('bounding_box', bb_from_detection)
        obj_to_update.__setitem__('mask', detection_result['masks'][:, :, obj_index])
        match_found = True
        # We can also check if this item is unique:
        # it might be the case that there is an item to remove
        # and if the remaining item(s) from the same class are present in 1 copy,
        # then they are unique! In this case we also change its name.
        if detection_result['names'].count(obj_class) == 1:
            print(f'The item {highest_intersection_key} is now unique.')
            obj_to_update.__setitem__('unique', True)
            # add it to the dictionary as non ambiguous key.
            # The ambiguopus key will be removed later.
            new_object = copy(obj_to_update)
            highest_intersection_key = detection_result['names'][obj_index]
            detected_objects[highest_intersection_key] = new_object
        processed.append(highest_intersection_key)

    return detected_objects, processed, match_found


def boundingbox_intersection(
    box1: List[int],
    box2: List[int]
) -> Tuple[int, List[int]]:
    """
    Compute the intersection of two bounding boxes.

    The bounding boxes have the notation [xmin, ymin, XMAX, YMAX].

    Args
    ----
        box1: the first bounding box.
        box2: the second bounding box.

    Returns
    -------
        area: the area of intersection.
        bounding_box: the resulting intersection bounding_box

    Note
    ----
        The area is expressed in terms of geometric area!
        The area intended as number of pixels, can be computed by the geometric features
        that can be obtained from the bounding box:
        PIXELS = area + perimeter/2 + 1

    """
    xmin_b1 = int(box1[0])
    xMAX_b1 = int(box1[2])
    ymin_b1 = int(box1[1])
    yMAX_b1 = int(box1[3])
    xmin_b2 = int(box2[0])
    xMAX_b2 = int(box2[2])
    ymin_b2 = int(box2[1])
    yMAX_b2 = int(box2[3])

    if (xmin_b2 > xMAX_b1) or (ymin_b2 > yMAX_b1) or (xMAX_b2 < xmin_b1) or (yMAX_b2 < ymin_b1):
        # No intersection
        area = 0
        bounding_box = []
    else:
        # There is intersection
        xmin = max(xmin_b1, xmin_b2)
        ymin = max(ymin_b1, ymin_b2)
        xMAX = min(xMAX_b1, xMAX_b2)
        yMAX = min(yMAX_b1, yMAX_b2)
        bounding_box = [xmin, ymin, xMAX, yMAX]
        # geometric area is width*height
        area = (xMAX - xmin)*(yMAX - ymin)

    return area, bounding_box
