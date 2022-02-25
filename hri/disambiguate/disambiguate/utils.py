"""Auxiliary functions for the main HRI loop."""

# Copyright (c) 2021 Fethiye Irmak Doğan
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

import os
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from scipy.spatial import distance
# import torch


def intersects(
    bound: List[float],
    cluster: List[float]
) -> bool:
    """Determine if a cluster intersects with a bounding box."""
    x0_bound, x1_bound = int((bound[0])), int((bound[1]))
    y0_bound, y1_bound = int((bound[2])), int((bound[3]))
    x0, x1, y0, y1 = int(cluster[0]), int(cluster[1]), int(cluster[2]), int(cluster[3])
    return not (x1_bound < x0 or x0_bound > x1 or y0_bound > y1 or y1_bound < y0)


def closest_non_detectable_obj(
    region: List[int],
    inference_dict: Dict,
    obj_name: str,
    appended: List,
    img_path: str,
    category_index: int
) -> int:
    """Return the index of the closest object to the given region."""
    im = Image.open(img_path)
    im_width, im_height = im.width, im.height
    object_ind = []
    for i in range(inference_dict['num_detections']):
        if category_index[inference_dict['detection_classes'][i]]['name'] == obj_name:
            object_ind.append(i)
    dist = float('inf')
    show_ind = -1
    for ind in object_ind:
        pos = inference_dict['detection_boxes'][ind]
        object_pos = [pos[1]*im_width, pos[3]*im_width, pos[0]*im_height, pos[2]*im_height]
        x_bound_center = (object_pos[0] + object_pos[1])/2
        y_bound_center = (object_pos[2] + object_pos[3])/2
        x_center = (region[0] + region[1])/2
        y_center = (region[2] + region[3])/2
        current_dist = distance.euclidean([x_bound_center, y_bound_center], [x_center, y_center])
        if dist > current_dist and ind not in appended:
            show_ind = ind
            dist = current_dist

    return show_ind


def construct_detection_dict(
    working_path: str,
    source_img_name: str,
    probas: Any,
    bboxes_scaled: Any,
    classes: List[str]
) -> Dict:
    """
    Rearrange the detected items in a dictionary with relevant keys.

    The dictionary has the following keys:
    - detection_masks: List[List[int]] | mask of the detected objects (int = 0 or 1).
    - num_detections: int | number of detected objects in the dictionary.
    - detection_boxes: List[int] | bounding boxes of the objects.
                                   Convention [ymin, xmin, yMAX, xMAX].
    - detection_scores: List[float] | score of the detection accuracy.
    - detection_classes: List[int] | index of the detected objects classes in the class Dict.

    Note:
    ----
        probas and bboxes_scaled are of type torch.tensor,
        not specified due to import problems in Windows.

    """
    image = Image.open(os.path.join(working_path + '/data', source_img_name))
    pickled = os.path.join(working_path, 'object_detection/detection_dictionary.pickle')

    with open(pickled, 'rb') as handle:
        output_dict = pickle.load(handle)
    n_detections = output_dict['num_detections']

    im_width, im_height = image.size
    for i, probability in enumerate(probas):
        if classes[probability.argmax()] == 'dining table' or\
           classes[probability.argmax()] == 'chair':
            continue
        else:
            n_detections += 1
            listed = bboxes_scaled[i].tolist()
            xmin_obj = listed[0]/im_width
            ymin_obj = listed[1]/im_height
            xMAX_obj = listed[2]/im_width
            yMAX_obj = listed[3]/im_height
            output_dict['detection_boxes'][n_detections-1] =\
                [ymin_obj, xmin_obj, yMAX_obj, xMAX_obj]
            output_dict['detection_classes'][n_detections-1] = probability.argmax()
            output_dict['detection_scores'][n_detections-1] = probability.max()

    output_dict['num_detections'] = n_detections

    return output_dict


def find_items_in_expression(nouns: List[str], classes: List[str]) -> List[str]:
    """Find the items in the expression that belong to the labels dataset."""
    items_in_expresison = []
    for class_label in classes:
        for noun_chunk in nouns:
            for noun in noun_chunk.split():
                if class_label == noun or class_label == noun[:-1]:
                    items_in_expresison.append(class_label)

    return items_in_expresison


def get_target_item_name(
        nouns: List[str],
        items_in_expression: List[str],
        inference_dict: Dict,
        category_index: Dict
) -> str:
    """
    Return the name of the target item.

    The name will be an class label among the detected objects or
    an empty string if the name is not present in the possible classes.

    Note:
    ----
        category_index is a dictionary with the convention:
        - 'id': int.
        - 'name': str.

        inference_dict is a dictionary with the convention:
        - detection_masks: List[List[int]] | mask of the detected objects (int = 0 or 1).
        - num_detections: int | number of detected objects in the dictionary.
        - detection_boxes: List[int] | bounding boxes of the objects.
        - detection_scores: List[float] | score of the detection accuracy.
        - detection_classes: List[int] | index of the detected objects classes in the class Dict.

    """
    detected_obj = []
    target_item = ''

    for i in range(inference_dict['num_detections']):
        detected_obj.append(category_index[inference_dict['detection_classes'][i]]['name'])

    for item in items_in_expression:
        for noun in nouns[0].split():
            if (item == noun or item == noun[:-1]) and item in detected_obj:
                target_item = item

    return target_item


def handle_hri(
    region_check_items: List,
    working_path: str,
    source_img: str,
    output_dict: Dict,
    target_item: str,
    category_index: str,
    bounding_boxes: List[int],
    colors: List[Tuple[int]],
    output_imgs: bool
) -> Tuple[List, List, str]:
    """
    Handle HRI items.

    Appended is a list with the idex of the target objects to disambiguate.

    """
    # create used paths
    data_directory = os.path.join(working_path, 'data')
    results_directory = os.path.join(data_directory, 'results')
    img_path = os.path.join(data_directory, source_img)
    res_img_questions = cv2.imread(img_path)
    pil_image = Image.open(img_path)

    candidate_centers = []
    appended = []
    obj_call_name = 'object'

    col = 0
    if len(region_check_items) > 1:
        # more than 1 object has been identified in the scene
        if target_item:
            obj_call_name = target_item
            for item in region_check_items:
                ind_ob = closest_non_detectable_obj(
                    bounding_boxes[item[1]],
                    output_dict,
                    target_item,
                    appended,
                    img_path,
                    category_index
                )
                if ind_ob != -1:
                    candidate_centers, appended, res_img_questions = handle_known_object(
                        candidate_centers,
                        appended,
                        res_img_questions,
                        ind_ob,
                        output_dict,
                        pil_image,
                        colors[col]
                    )
                    col += 1
            show_img_question(results_directory, res_img_questions, output_imgs)
        else:
            obj_call_name = 'object'
            for item in region_check_items:
                candidate_centers, res_img_questions = handle_unknown_object(
                    candidate_centers,
                    res_img_questions,
                    bounding_boxes[item[1]],
                    output_dict, pil_image,
                    colors[col]
                )
                col += 1
            show_img_question(results_directory, res_img_questions, output_imgs)
    else:
        # object has not been identified in the scene
        if target_item:
            obj_call_name = target_item
            for item in bounding_boxes:
                ind_ob = closest_non_detectable_obj(
                    item, output_dict, target_item, appended, img_path, category_index)
                if ind_ob != -1:
                    candidate_centers, appended, res_img_questions = handle_known_object(
                        candidate_centers,
                        appended,
                        res_img_questions,
                        ind_ob,
                        output_dict,
                        pil_image,
                        colors[col]
                    )
                    col += 1
            show_img_question(results_directory, res_img_questions, output_imgs)
        else:
            obj_call_name = 'object'
            for item in bounding_boxes:
                candidate_centers, res_img_questions = handle_unknown_object(
                    candidate_centers,
                    res_img_questions,
                    item,
                    output_dict,
                    pil_image,
                    colors[col]
                )
                col += 1
            show_img_question(results_directory, res_img_questions, output_imgs)

    return candidate_centers, appended, obj_call_name


def show_img_question(
    results_directory: str,
    res_img_questions: np.ndarray,
    output_imgs: bool
) -> None:
    """Show the image with the target objects from the question."""
    cv2.imwrite(os.path.join(results_directory, 'question_regions.jpg'), res_img_questions)
    if output_imgs:
        imm_q = plt.imread(os.path.join(results_directory, 'question_regions.jpg'))
        plt.imshow(imm_q)
        plt.show()


def handle_known_object(
    candidate_centers: List[List[int]],
    appended: List,
    res_img_questions: np.ndarray,
    ind_ob: int,
    output_dict: Dict,
    pil_image: Image,
    color: Tuple[int]
) -> Tuple[List, List, np.ndarray]:
    """Find target known object bounding box and print it in the image."""
    appended.append(ind_ob)
    pos = output_dict['detection_boxes'][ind_ob]
    X_im, Y_im = pil_image.size
    x0, x1, y0, y1 = pos[1]*X_im, pos[3]*X_im, pos[0]*Y_im, pos[2]*Y_im
    candidate_centers.append([(x0 + x1)/2, (y0 + y1)/2])
    res_img_questions = cv2.rectangle(
        res_img_questions, (int(x0), int(y0)), (int(x1), int(y1)), color, 5)

    return candidate_centers, appended, res_img_questions


def handle_unknown_object(
    candidate_centers: List[List[int]],
    res_img_questions: np.ndarray,
    bounding_box: List[int],
    output_dict: Dict,
    pil_image: Image,
    color: Tuple[int]
) -> Tuple[List, np.ndarray]:
    """Find target unknown object bounding box and print it in the image."""
    x0, x1, y0, y1 = bounding_box
    candidate_centers.append([(x0 + x1)/2, (y0 + y1)/2])
    X_im, Y_im = pil_image.size
    X0_obj = x0/X_im
    Y0_obj = y0/Y_im
    X1_obj = x1/X_im
    Y1_obj = y1/Y_im
    output_dict['detection_boxes'][output_dict['num_detections']] =\
        [Y0_obj, X0_obj, Y1_obj, X1_obj]
    output_dict['detection_classes'][output_dict['num_detections']] = 91
    output_dict['detection_scores'][output_dict['num_detections']] = 1.0
    output_dict['num_detections'] += 1
    res_img_questions = cv2.rectangle(
        res_img_questions, (int(x0), int(y0)), (int(x1), int(y1)), color, 5)

    return candidate_centers, res_img_questions
