"""This script is used to generate the referring expressions."""

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
from typing import Dict, List, Tuple

from disambiguate import label_map_util
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


class REG:
    """
    Model and auxiliary functions for Referring Expressions Generation.

    This module uses two networks:
    - RPN (Relation Presence Network): predicts the spatial relations between objects.
        Structure: 2 hidden layers with 32 and 16 hidden neurons, ReLU non-linearities.
        The output layer uses a softmax function to map activations to probabilities.
        Multi-label cross-entropy loss is used to train the network.
        Moreover, dropout [26] is added between each layer including input and output layers
        with 0.2 probability to prevent overfitting. Early stopping is used to stop training
        whenever validation accuracy stop increasing.
    - RIN (Relation Informativeness Network): predicts the most informative spatial relation
      describing the target object.
        Structure: 3 hidden layers with 64, 16, and 8 hidden neurons.
        The hidden and output layers have ReLU and sigmoid nonlinearities, respectively.
        In RIN, dropout with 0.2 probability is added to each layer.
        Early stopping is used to stop training when validation accuracy stops increasing.

    """

    def __init__(self, working_path: str, source_img: str):
        """Initialize paths and evaluate recognition models."""
        # List of the strings that is used to add correct label for each box.
        self.path_to_labels = os.path.join(
            working_path, 'object_detection/data/mscoco_label_map.pbtxt')
        self.img_path = os.path.join(working_path, 'data/' + source_img)

        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # train models
        self.RPN = load_model(os.path.join(
            working_path, 'object_detection/ref_model/RPN.hdf5'))
        self.RIN = load_model(os.path.join(
            working_path, 'object_detection/ref_model/RIN.hdf5'))

    def get_category_index(self) -> Dict[int, str]:
        """
        Return the category index dictionary.

        The dictionary has the following structure:
            - 'id' : int | the id of the class,
            - 'name' : str | the class name.

        """
        return self.category_index

    def generate_referring_expression(
        self,
        target_name: str,
        x_coord: int,
        y_coord: int,
        class_label: str,
        detection_dict: Dict
    ) -> str:
        """Generate the referring epxression for the followup question."""
        relation_dict = self.__build_relation_dictionary(
            target_name, x_coord, y_coord, class_label, detection_dict)
        spatial_relations = []
        highest = 0
        for key in relation_dict.keys():
            if len(relation_dict[key]) > 0:
                for item in relation_dict[key]:
                    if highest < item[2]:
                        highest = item[2]
                        spatial_relations = [item, key]
        if spatial_relations == [] or highest < 0.3:
            return self.__backup_expression(x_coord, y_coord)
        if spatial_relations[1] == 0:
            return ' to the left of the ' + spatial_relations[0][1] + ' from my viewpoint'
        if spatial_relations[1] == 1:
            return ' to the right of the ' + spatial_relations[0][1] + ' from my viewpoint'
        if spatial_relations[1] == 2:
            return ' close to the ' + spatial_relations[0][1] + ' from my viewpoint'
        if spatial_relations[1] == 3:
            return ' close to the ' + spatial_relations[0][1] + ' from my viewpoint'
        if spatial_relations[1] == 4:
            return ' behind the ' + spatial_relations[0][1] + ' from my viewpoint'
        if spatial_relations[1] == 5:
            return ' in front of the ' + spatial_relations[0][1] + ' from my viewpoint'

    # --------------------------------------------------------------------------------------------
    #                                     PRIVATE FUNCTIONS
    # --------------------------------------------------------------------------------------------
    def __build_relation_dictionary(
        self,
        target_name: str,
        x_coord: int,
        y_coord: int,
        class_label: str,
        detection_dict: Dict
    ) -> Dict[int, list]:
        """
        Build a dictionary of spatial relations using both RPN and RIN models.

        Returns
        -------
            The dictionary of spatial relations for the target object.

        """
        target_rels, distractor_rels = self.__predict_spatial_relations(
            x_coord, y_coord, class_label, detection_dict)

        # Build the dictionary for the target object.
        dic_target_rel = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for rel in target_rels:
            for relation_type in rel[2]:
                # Concatenate bounding boxes and spatial relaitons.
                rin_input = (np.concatenate(
                        (
                            np.asarray([rel[0][1][0] + rel[1][1][0]]),
                            [to_categorical(relation_type, num_classes=6)]
                        ),
                        axis=1))
                pred = self.RIN.predict(rin_input)[0][0]
                dic_target_rel[relation_type] += [[rel[0][0], rel[1][0], pred]]

        # Remove empty relations from the distractor ones.
        distractor_rels_clean = []
        for rel in distractor_rels:
            if rel[2] != []:
                distractor_rels_clean += [rel]

        # Build the dictionary for the distractor object.
        dic_distractor_rel = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for rel in distractor_rels_clean:
            for relation_type in rel[2]:
                # Concatenate bounding boxes and spatial relaitons.
                rin_input = (np.concatenate(
                    (
                        np.asarray([rel[0][1][0] + rel[1][1][0]]),
                        [to_categorical(relation_type, num_classes=6)]
                    ),
                    axis=1))
                pred = self.RIN.predict(rin_input)[0][0]
                dic_distractor_rel[relation_type] += [[rel[0], rel[1], pred]]

        # Find the most informative spatial relations for the distractor object
        most_informative_distractor = []
        for key in dic_distractor_rel.keys():
            distinct_relations = []
            reference_obj = []
            for item in dic_distractor_rel[key]:
                if item[0] not in distinct_relations:
                    distinct_relations.append(item[0])
                    reference_obj.append(item[0])
            if len(distinct_relations) > 0:
                highest = np.zeros(len(distinct_relations))
                for item in dic_distractor_rel[key]:
                    i = distinct_relations.index(item[0])
                    if highest[i] < item[2]:
                        highest[i] = item[2]
                        reference_obj[i] = item[1]
            for i in range(len(distinct_relations)):
                most_informative_distractor.append([reference_obj[i][0], key])

        # find the target relations that are ambiguous
        ambiguous_rels = []
        for key in dic_target_rel.keys():
            for item in dic_target_rel[key]:
                if item[1] == class_label or item[1] == target_name:
                    ambiguous_rels += [[key, item]]
                for i in most_informative_distractor:
                    if i[0] == item[1] and i[1] == key:
                        ambiguous_rels += [[key, item]]

        # remove the relations that are ambiguus from the target relations dictionary
        for pair in ambiguous_rels:
            if pair[0] in dic_target_rel and pair[1] in dic_target_rel[pair[0]]:
                dic_target_rel[pair[0]].remove(pair[1])
        for key in dic_target_rel.keys():
            if len(dic_target_rel[key]) > 1:
                highest = 0
                # this relations pertain the target object
                most_informative_rels = []
                for item in dic_target_rel[key]:
                    if highest < item[2]:
                        highest = item[2]
                        most_informative_rels = item
                dic_target_rel[key] = [most_informative_rels]

        return dic_target_rel

    def __backup_expression(self, x_coord: int, y_coord: int) -> str:
        """Generate a backup referring expression."""
        image = Image.open(self.img_path)
        im_width, im_height = image.size
        if x_coord < im_width/3:
            return ' on your right side of the table'
        if x_coord >= im_width/3 and x_coord < 2*im_width/3:
            if y_coord < im_height/5:
                return ' closer to me'
            if y_coord >= im_height/5 and y_coord < 4*im_height/5:
                return ' close to the middle of the table'
            if y_coord >= 4*im_height/5:
                return ' closer to you'
        if x_coord >= 2*im_width/3:
            return ' on your left side of the table'

    def __predict_spatial_relations(
        self,
        x_coord: int,
        y_coord: int,
        class_label: str,
        detection_dict: Dict
    ) -> Tuple[list, list]:
        """
        Compute spatial relations for target object and distractor objects woth RPN model.

        The target object is the object we want to find spatial relations about.
        The distractor objects are objects of the same class of the target.
        We need distractor objects to obtain non ambiguous spatial relations.

        Returns
        -------
            The target_obj_relations list has structure:
            - The target name (str) and its bounding boxes (List[int]).
            - The name of the reference object (str) anf its bounding boxes (List[int]).
            - A probability distribution for the spatial relation
              between the objects (List[float]).

            The distractor_obj_relations list has structure:
            - The distractor name (str) and its bounding boxes (List[int]).
            - The name of the reference object (str) anf its bounding boxes (List[int]).
            - A probability distribution for the spatial relation
              between the objects (List[float]).

        """
        normalized_bounding_boxes, obj_bounding_boxes =\
            self.__compute_bounding_boxes(detection_dict)
        target_index = self.__find_target_index(x_coord, y_coord, detection_dict, class_label)
        # Extract classes from the detected objects.
        obj_classes = detection_dict['detection_classes']
        target_obj_relations = []
        for i in range(len(obj_bounding_boxes)):
            if i != target_index:
                # Concatenate the bounding boxes of two objects.
                rpn_input = np.asarray(
                    [normalized_bounding_boxes[target_index] + normalized_bounding_boxes[i]])
                prediction = self.RPN.predict(rpn_input)[0]
                target_obj_relations +=\
                    [[
                        [
                            self.category_index[obj_classes[target_index]]['name'],
                            [normalized_bounding_boxes[target_index]]
                        ],
                        [
                            self.category_index[obj_classes[i]]['name'],
                            [normalized_bounding_boxes[i]]
                        ],
                        list(np.nonzero(prediction > 0.50)[0])
                    ]]
        distractor_obj_relations = []
        for i in range(len(obj_bounding_boxes)):
            if i != target_index and\
               self.category_index[obj_classes[i]]['name'] ==\
                    self.category_index[obj_classes[target_index]]['name']:
                for j in range(len(obj_bounding_boxes)):
                    if i != j:
                        rpn_input = np.asarray(
                            [normalized_bounding_boxes[i] + normalized_bounding_boxes[j]])
                        prediction = self.RPN.predict(rpn_input)[0]
                        distractor_obj_relations +=\
                            [[
                                [
                                    self.category_index[obj_classes[i]]['name'],
                                    [normalized_bounding_boxes[i]]
                                ],
                                [
                                    self.category_index[obj_classes[j]]['name'],
                                    [normalized_bounding_boxes[j]]
                                ],
                                list(np.nonzero(prediction > 0.50)[0])
                            ]]
        return target_obj_relations, distractor_obj_relations

    def __compute_bounding_boxes(
        self,
        detection_dict: Dict
    ) -> Tuple[List[int], List[int]]:
        """Return the bounding boxes for the detected objects."""
        # TODO: don't use a self object but pass it as param instead.
        obj_bounding_boxes = []
        normalized_bounding_boxes = []
        image = Image.open(self.img_path)
        im_width, im_height = image.size
        for i in range(len(detection_dict['detection_scores'])):
            if detection_dict['detection_classes'][i] in self.category_index.keys() and\
               detection_dict['detection_scores'][i] > 0.30:
                box = detection_dict['detection_boxes'][i]
                obj_bounding_boxes = obj_bounding_boxes +\
                    [(
                        box[1]*im_width,
                        box[0]*im_height,
                        box[3]*im_width - box[1]*im_width,
                        box[2]*im_height - box[0]*im_height
                    )]
                normalized_bounding_boxes = normalized_bounding_boxes +\
                    [(box[1], box[0], box[3] - box[1], box[2] - box[0])]

        return normalized_bounding_boxes, obj_bounding_boxes

    def __find_target_index(
        self,
        x_coord: int,
        y_coord: int,
        detection_dict: Dict,
        class_label: str
    ) -> int:
        """Return the index of the object with the input class label in the image."""
        # TODO: don't use a self object but pass it as param instead.
        # TODO: in such a case it is possible to test it and move it to a utils script.
        image = Image.open(self.img_path)
        im_width, im_height = image.size
        object_bounding_boxes = []
        boxes_values = detection_dict['detection_boxes']
        classes_values = detection_dict['detection_classes']
        for box in boxes_values:
            object_bounding_boxes = object_bounding_boxes +\
                [(box[1]*im_width, box[3]*im_width, box[0]*im_height, box[2]*im_height)]
        index = 0
        for i, box in enumerate(object_bounding_boxes):
            # Check that the input point is in the bounding box of the object
            # and that we are taking the object with the right label.
            if box[0] < x_coord and box[1] > x_coord and\
               box[2] < y_coord and box[3] > y_coord and\
               self.category_index[classes_values[index]]['name'] == class_label:
                # index = i
                break
            # TODO: remove the 3 lines below if we use index = 1
            # TODO: test it before!!
            index += 1
        if index == len(object_bounding_boxes):
            index = -1

        return index
