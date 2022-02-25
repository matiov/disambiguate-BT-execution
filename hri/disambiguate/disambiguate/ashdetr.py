"""
Class with model and auxiliary functions for object detection.

It uses a detection model adapted from Facebook (from Ashkamath).
"""

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

from collections import defaultdict
import cv2
from disambiguate import utils
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import distance
from skimage.measure import find_contours
import torch
import torchvision.transforms as T


torch.set_grad_enabled(False)


class ASHDetr:
    """
    Model and auxiliary functions for object detection.

    It uses a detection model adapted from Facebook (from Ashkamath).

    """

    def __init__(
        self,
        working_path: str,
        source_img: str,
        colors: List[List[int]],
        output_imgs: bool
    ):
        self.colors = colors
        # path to the results directory
        self.target_dir = os.path.join(working_path, 'data/results')
        self.disambiguated_img = os.path.join(self.target_dir, 'disambiguated.jpg')
        # path to the source image
        self.image_path = os.path.join(working_path, 'data/' + source_img)
        # Load detection model
        self.detection_model, self.postprocessor = torch.hub.load(
            'ashkamath/mdetr:main',
            'mdetr_efficientnetB5',
            pretrained=True,
            return_postprocessor=True
        )
        self.detection_model.eval()

        self.output_imgs = output_imgs

    def target_bounding_box(
        self,
        target_image: str,
        expression: str,
        nouns: List[str],
        item_idx: int,
        items_in_expression: List[str],
        inference_dict: Dict,
        category_index: Dict,
        region_boxes: List[List[int]] = [],
        appended: List[List[int]] = []
    ) -> List[int]:
        """
        Compute the bounding box of the target object.

        Args
        ----
            target_image: path to the target image for detection.
            expression: string associated to the image to query the detection.
            nouns: list of object of interest.
            item_idx: index of the item to compute the bounding box of.
            items_in_expression: list of possible objects.
            inference_dict: dictionary with the detection results.
            category_index: dictionary with class index and label.
            region_boxes: region of interest in the image.
            appended: list with the idex of the detetctable target objects to disambiguate.

        Results
        -------
            The bounding box of the target object.

        Note
        ----
            The bounding boxes are in the convention [xmin, ymin, XMAX, YMAX]

        """
        target_item = utils.get_target_item_name(
            nouns, items_in_expression, inference_dict, category_index)
        if target_item:
            if item_idx < len(appended):
                # the object is detectable so we can return the bounding box
                bounding_box = self.__find_target_bounding_box(
                    appended[item_idx],
                    inference_dict,
                    target_item,
                    category_index
                )
            else:
                # the object is not detectable so we return the region instead
                bounding_box = self.__find_target_bounding_box(
                    region_boxes[item_idx],
                    inference_dict,
                    target_item,
                    category_index,
                    detectable_object=False
                )
        else:
            pil_image = Image.open(target_image)
            try:
                probas, bboxes, res_labels = self.__detect_objects(pil_image, expression)
            except Exception:
                print("Something went wrong with transformers network.")
                probas, bboxes, res_labels = [], [], []
            finally:
                x0, y0, x1, y1 = float('inf'), float('inf'), 0, 0
                for i in range(len(probas)):
                    if nouns[0].lower().replace(' ', '') in\
                       res_labels[i].lower().replace(' ', ''):
                        (xmin, ymin, xmax, ymax) = bboxes[i].tolist()
                        if xmin < x0:
                            x0 = xmin
                        if ymin < y0:
                            y0 = ymin
                        if xmax > x1:
                            x1 = xmax
                        if ymax > y1:
                            y1 = ymax
                image = cv2.imread(self.image_path)
                color = (0, 0, 255)
                linewidth = 5
                if x0 != float('inf') and y0 != float('inf') and x1 != 0 and y1 != 0:
                    rectA = int(x0 + region_boxes[item_idx][0])
                    rectB = int(y0 + region_boxes[item_idx][2])
                    rectC = int(x1 + region_boxes[item_idx][0])
                    rectD = int(y1 + region_boxes[item_idx][2])
                else:
                    rectA = int(region_boxes[item_idx][0])
                    rectB = int(region_boxes[item_idx][2])
                    rectC = int(region_boxes[item_idx][1])
                    rectD = int(region_boxes[item_idx][3])

                res_img = cv2.rectangle(image, (rectA, rectB), (rectC, rectD), color, linewidth)
                bounding_box = [rectA, rectB, rectC, rectD]
                cv2.imwrite(self.disambiguated_img, res_img)

        if self.output_imgs:
            res_img = plt.imread(self.disambiguated_img)
            plt.imshow(res_img)
            plt.axis('off')
            plt.show()

        return bounding_box

    # --------------------------------------------------------------------------------------------
    #                                     PRIVATE FUNCTIONS
    # --------------------------------------------------------------------------------------------
    def __find_target_bounding_box(
        self,
        region: int or List[int],
        inference_dict: Dict,
        target_name: str,
        category_index: Dict,
        detectable_object: bool = True
    ) -> List[int]:
        """
        Find the bounding box of the target object and print it in the image.

        If the object is detectable then it is found in the list of appended ones.
        In that case, region is the index of the item, otherwise is a region box.

        """
        im = Image.open(self.image_path)
        im_width, im_height = im.width, im.height
        if detectable_object:
            show_ind = region
        else:
            object_ind = []
            for i in range(inference_dict['num_detections']):
                if category_index[inference_dict['detection_classes'][i]]['name'] == target_name:
                    object_ind.append(i)
            dis = float("inf")
            show_ind = -1
            for ind in object_ind:
                pos = inference_dict['detection_boxes'][ind]
                object_pos = [pos[1]*im_width, pos[3]*im_width, pos[0]*im_height, pos[2]*im_height]
                x_bound_center = (object_pos[0] + object_pos[1])/2
                y_bound_center = (object_pos[2] + object_pos[3])/2
                x_center = (region[0] + region[1])/2
                y_center = (region[2] + region[3])/2
                temp_d = distance.euclidean(
                    [x_bound_center, y_bound_center], [x_center, y_center])
                if dis > temp_d:
                    show_ind = ind
                    dis = temp_d

        pos = inference_dict['detection_boxes'][show_ind]
        object_pos = [pos[1]*im_width, pos[3]*im_width, pos[0]*im_height, pos[2]*im_height]
        original_image = cv2.imread(self.image_path)
        color = (0, 0, 255)
        linewidth = 5
        result_img = cv2.rectangle(
            original_image,
            (int(object_pos[0]), int(object_pos[2])),
            (int(object_pos[1]), int(object_pos[3])),
            color,
            linewidth
        )
        cv2.imwrite(self.disambiguated_img, result_img)

        return [int(object_pos[0]), int(object_pos[2]), int(object_pos[1]), int(object_pos[3])]

    def __detect_objects(
        self,
        image: Image,
        caption: str
    ) -> Tuple[List[float], List[List[int]], List[str]]:
        """
        Detect the objects in the given image.

        Returns detection results and print the image with detection results.

        """
        transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # mean-std normalize the input image (batch-size: 1)
        transformed_img = transform(image).unsqueeze(0)
        # propagate through the model
        memory_cache = self.detection_model(transformed_img, [caption], encode_and_save=True)
        detection_dict = self.detection_model(
            transformed_img, [caption], encode_and_save=False, memory_cache=memory_cache)
        # keep only predictions with 0.7+ confidence
        probas = 1 - detection_dict['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > 0.70).cpu()
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.__rescale_bboxes(
            detection_dict['pred_boxes'].cpu()[0, keep], image.size)
        # Extract the text spans predicted by each box
        positive_tokens =\
            (detection_dict['pred_logits'].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
        predicted_spans = defaultdict(str)
        for token in positive_tokens:
            item, pos = token
            if pos < 255:
                span = memory_cache['tokenized'].token_to_chars(0, pos)
                predicted_spans[item] += ' ' + caption[span.start:span.end]

        labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]
        self.__plot_results(image, probas[keep], bboxes_scaled, labels)

        return (probas[keep], bboxes_scaled, labels)

    def __plot_results(
        self,
        pil_img: Image,
        scores: List[float],
        boxes: List[List[int]],
        labels: List[str],
        masks: list = None
    ):
        """Plot the detection results in the image."""
        plt.figure(figsize=(16, 10))
        np_image = np.array(pil_img)
        ax = plt.gca()
        colors = self.colors*100
        if masks is None:
            masks = [None for _ in range(len(scores))]
        assert len(scores) == len(boxes) == len(labels) == len(masks)
        for score, (xmin, ymin, xmax, ymax), label, mask, color in\
                zip(scores, boxes.tolist(), labels, masks, colors):
            ax.add_patch(plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=color,
                linewidth=1
            ))
            text = f'{label}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

            if mask is None:
                continue
            np_image = self.__apply_mask(np_image, mask, color)

            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor='none', edgecolor=color)
                ax.add_patch(p)
        plt.imshow(np_image)
        plt.axis('off')
        plt.imsave(os.path.join(self.target_dir, 'detection_show.jpg'), np_image)
        plt.close()

    def __rescale_bboxes(
        self,
        bounding_box: Any,
        size: Tuple[int, int]
    ) -> Any:
        """
        Resize the bounding boxes depending on the image size.

        Note:
        ----
            The bounding boxes are of type torch.tensor.

        """
        img_w, img_h = size
        x_c, y_c, w, h = bounding_box.unbind(1)
        box = [(x_c - 0.5*w), (y_c - 0.5*h), (x_c + 0.5*w), (y_c + 0.5*h)]
        box = torch.stack(box, dim=1)
        box = box*torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        return box

    def __apply_mask(
        self,
        image: np.ndarray,
        mask: list,
        color: List[int],
        alpha: float = 0.5
    ) -> np.ndarray:
        """Apply the object mask in the given image."""
        for c in range(3):
            image[:, :, c] = np.where(
                mask == 1, image[:, :, c]*(1 - alpha) + alpha*color[c]*255, image[:, :, c])

        return image
