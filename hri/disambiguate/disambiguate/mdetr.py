"""Modulated Detection for End-to-End Multi-Modal Understanding."""

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
from pathlib import Path
import subprocess
# import time
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans


class MDetr:
    """
    Modulated Detection for End-to-End Multi-Modal Understanding.

    This method uses
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

    """

    def __init__(self, working_path: str, image_name: str):
        self.working_path = working_path
        self.image_name = image_name
        data_path = os.path.join(working_path, 'data')
        self.image_path = os.path.join(data_path, self.image_name)
        self.results_path = os.path.join(data_path, 'results')

    def detection(
        self,
        expression: str,
        areas_of_interest: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Apply Grad-CAM to compute the bounding boxes corresponding to the expression.

        Args
        ----
            expression: the expression to use to identify the objects in the image.
            areas_of_interest: bounding boxes of the objects described by the expression.

        Returns
        -------
            bounding_boxes: bounding box resulting from applying the method.
            areas_of_interest: areas of interest in the image found by the method.

        Notes
        -----
            The bounding boxes have convention [xmin, XMAX, ymin, YMAX].

        """
        # start = time.time()
        parent_path = Path(self.working_path).parent
        correct_path = Path(parent_path).parent.absolute()
        grad_cam_path = os.path.join(correct_path, 'grad-cam')
        # call the captioning method in the Grad-Cam pakage
        bash_file = os.path.join(self.working_path, 'captioning.sh')
        subprocess.call([
            'bash',
            bash_file,
            '%s' % grad_cam_path,
            '%s' % self.image_path,
            '%s' % (self.results_path + '/'),
            '%s' % expression
        ])
        original_image = Image.open(self.image_path)
        gcam_img_path = self.results_path + '/caption_gcam_hm_' + expression + '.png'
        gradcam_image = Image.open(gcam_img_path)
        n_clusters = self.__count_clusters(gcam_img_path)
        n_outputs = 2
        # t1 = time.time()
        # print(f'Elapsed time 1: {start - t2}.')
        bounding_boxes = self.__extract_boxes_from_clusters(gcam_img_path, n_clusters, n_outputs)
        oimX, oimY = original_image.size[0], original_image.size[1]
        gcX, gcY = gradcam_image.size[0], gradcam_image.size[1]
        # t2 = time.time()
        # print(f'Elapsed time 2: {start - t2}.')
        resized_boxes, areas_of_interest = self.__compute_bounding_boxes(
            areas_of_interest, n_outputs, bounding_boxes, oimX, oimY, gcX, gcY)

        return resized_boxes, areas_of_interest

    # --------------------------------------------------------------------------------------------
    #                                     PRIVATE FUNCTIONS
    # --------------------------------------------------------------------------------------------
    def __count_clusters(self, im_name: str) -> int:
        """Count the number of clusters in the input image."""
        pic = plt.imread(im_name)
        newpic = np.zeros((pic.shape[0], pic.shape[1]))
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                if pic[i][j][0] > 0.9 or pic[i][j][1] > 0.9:
                    newpic[i][j] = 1
        plt.imshow(newpic)
        plt.imsave(self.results_path + "/cluster2.png", newpic)
        plt.close()
        label_img = label(newpic)
        regions = regionprops(label_img)
        cluster = 0
        for i, region in enumerate(regions):
            if region.area > 150:
                cluster += 1
        if cluster == 1:
            return 3
        else:
            return cluster + 1

    def __ordered_activations(
        self,
        original_image: np.ndarray,
        all_regions: list
    ) -> np.ndarray:
        """Compute and sort the activation regions."""
        activations = np.zeros((len(all_regions), 2))
        for i, region in enumerate(all_regions):
            for ind in region:
                activations[i][0] += original_image[ind[0]][ind[1]][0]*0.7 +\
                    original_image[ind[0]][ind[1]][1]*0.3
                activations[i][1] += 1

        resulting_activations = np.zeros((len(all_regions)))
        for i, activation in enumerate(activations):
            resulting_activations[i] = activation[0]/activation[1]

        return np.argsort(resulting_activations)[::-1]

    def __extract_boxes_from_clusters(
        self,
        im_name: str,
        n_clusters: int,
        n_outputs: int
    ) -> List[int]:
        """Extract the bounding boxes from the input image."""
        original_image = plt.imread(im_name)
        kernel_size = (11, 11)
        std = 0
        blurred_image = cv2.GaussianBlur(original_image, kernel_size, std)
        newpic = np.zeros(
            (blurred_image.shape[0], blurred_image.shape[1], blurred_image.shape[2] + 2))
        for i in range(blurred_image.shape[0]):
            for j in range(blurred_image.shape[1]):
                if blurred_image[i][j][0] > 0.5 or blurred_image[i][j][1] > 0.5:
                    newpic[i][j] = np.append(
                        [i*1.0/blurred_image.shape[0], j*1.0/blurred_image.shape[1]],
                        blurred_image[i][j]
                    )
        reshaped_image = newpic.reshape(newpic.shape[0]*newpic.shape[1], newpic.shape[2])
        # find clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped_image)
        labels = kmeans.labels_.reshape(blurred_image.shape[0], blurred_image.shape[1])
        blobs = []
        for i in range(0, n_clusters):
            blobs.append([])
        for i in range(blurred_image.shape[0]):
            for j in range(blurred_image.shape[1]):
                blobs[labels[i][j]].append([i, j])
        all_regions = []
        for blob in blobs:
            finalpic = np.zeros((blurred_image.shape[0], blurred_image.shape[1]))
            for co in blob:
                finalpic[co[0]][co[1]] = 1
            label_img = label(finalpic)
            regions = regionprops(label_img)
            for i, region in enumerate(regions):
                if region.area > 150:
                    all_regions.append(region.coords)
                else:
                    continue
        # find activation regions
        ordered_activations = self.__ordered_activations(original_image, all_regions)
        finalpic = np.zeros(
            (blurred_image.shape[0], blurred_image.shape[1], blurred_image.shape[2]))
        labelcolor = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1],
            [0, 1, 1], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]
        ]
        for i in range(n_outputs):
            for cord in all_regions[ordered_activations[i]]:
                finalpic[cord[0]][cord[1]] = labelcolor[i]
        plt.imshow(finalpic)
        plt.imsave(self.results_path + "/cluster1.png", finalpic)
        # find bounding boxes
        activation_bounds = []
        for i in range(len(ordered_activations)):
            currlist = all_regions[ordered_activations[i]]
            y0 = min(currlist, key=itemgetter(0))[0]
            y1 = max(currlist, key=itemgetter(0))[0]
            x0 = min(currlist, key=itemgetter(1))[1]
            x1 = max(currlist, key=itemgetter(1))[1]
            activation_bounds.append(([x0, x1, y0, y1]))
        output_boxes = []
        for bound in activation_bounds:
            accept_bound = True
            for box in output_boxes:
                if box[0] >= bound[0] and box[1] <= bound[1] and\
                        box[2] >= bound[2] and box[3] <= bound[3]:
                    accept_bound = False
                elif box[0] <= bound[0] and box[1] >= bound[1] and\
                        box[2] <= bound[2] and box[3] >= bound[3]:
                    accept_bound = False
            if accept_bound:
                output_boxes.append(bound)

        return output_boxes

    def __resize_boundingbox(
        self,
        box: List[int],
        width: int,
        height: int
    ) -> List[int]:
        """
        Resize the input bounding box.

        The resiezed box is a zoomed one that is printed in the cropped images.

        """
        x0_new = box[0] - (box[1] - box[0])/2
        x1_new = box[1] + (box[1] - box[0])/2
        y0_new = box[2] - (box[3] - box[2])/2
        y1_new = box[3] + (box[3] - box[2])/2
        if x0_new < 0:
            x0_new = 0
        if y0_new < 0:
            y0_new = 0
        if x1_new >= width:
            x1_new = width - 1
        if y1_new >= height:
            y1_new = height - 1

        return [x0_new, x1_new, y0_new, y1_new]

    def __compute_bounding_boxes(
        self,
        areas_of_interest: List[int],
        n_outputs: int,
        bounding_boxes: List[int],
        image_width: int,
        image_height: int,
        gradcam_width: int,
        gradcam_height: int
    ) -> Tuple[List[int], List[int]]:
        """
        Compute the bounding boxes of the object of interests matched by the method.

        This function prints the bounding boxes in output images, that are cropped images
        showing the output object from the detection algorithm.

        """
        x0, y0 = float('inf'), float('inf')
        x1, y1 = -1, -1
        pure_im = cv2.imread(self.image_path)
        Ry = image_height/gradcam_height
        Rx = image_width/gradcam_width
        if len(bounding_boxes) < n_outputs:
            n_outputs = len(bounding_boxes)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
        resized_boxes = []
        for i in range(n_outputs):
            x0 = bounding_boxes[i][0]*Rx
            x1 = bounding_boxes[i][1]*Rx
            y0 = bounding_boxes[i][2]*Ry
            y1 = bounding_boxes[i][3]*Ry
            resized_boxes.append([x0, x1, y0, y1])
            resimage = cv2.rectangle(
                pure_im, (int(x0), int(y0)), (int(x1), int(y1)), colors[2], thickness=5)
            crop_bounds = self.__resize_boundingbox(
                [x0, x1, y0, y1], image_width, image_height)
            crop_img = pure_im[
                int(crop_bounds[2]):int(crop_bounds[3]),
                int(crop_bounds[0]):int(crop_bounds[1])
            ]
            areas_of_interest.append(crop_bounds)
            cv2.imwrite(self.results_path + "/cropped" + str(i) + ".jpg", crop_img)
        cv2.imwrite(os.path.join(self.results_path, self.image_name), resimage)

        return resized_boxes, areas_of_interest
