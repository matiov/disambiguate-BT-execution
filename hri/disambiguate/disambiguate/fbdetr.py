"""Uses DETR RESNET50 from facebook to perform object detection."""

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
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T


torch.set_grad_enabled(False)


class FBDetr:
    """
    Model and auxiliary functions for object detection.

    It uses a detection model from Facebook.

    """

    def __init__(self, classes: List[str], colors: List[List[int]]):
        # target riectory is data/results
        self.classes = classes
        self.colors = colors
        self.fb_detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.fb_detr.eval()

    def detection(
        self,
        source_path: str,
        source_img_name: str
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Detect objects in the input image.

        Args
        ----
            source_path: path to the input image.
            source_img_name: name of the input image.

        Returns
        -------
            probas: 'pred_logits' field of output tensor from the detection algorithm.
            bboxes_scaled: scaled bounding boxes of the detected objects.

        """
        image = Image.open(os.path.join(source_path + '/data', source_img_name))
        # standard PyTorch mean-std input image normalization
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # mean-std normalize the input image (batch-size: 1)
        img = transform(image).unsqueeze(0)
        # propagate through the model
        outputs = self.fb_detr(img)
        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.__rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
        self.__plot_results(
            os.path.join(source_path, 'data/results'),
            source_img_name,
            image,
            probas[keep],
            bboxes_scaled
        )

        return (probas[keep], bboxes_scaled)

    # --------------------------------------------------------------------------------------------
    #                                     PRIVATE FUNCTIONS
    # --------------------------------------------------------------------------------------------
    def __rescale_bboxes(
        self,
        out_bbox: torch.tensor,
        size: Tuple[int, int]
    ) -> torch.tensor:
        """Compute and rescale the bounding boxes from the detection."""
        img_w, img_h = size
        x_c, y_c, w, h = out_bbox.unbind(1)
        box = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        box = torch.stack(box, dim=1)
        box = box * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return box

    def __plot_results(
        self,
        save_path: str,
        source_name: str,
        image: Image,
        prob: torch.tensor,
        boxes: torch.tensor
    ):
        """Plot the results of the object detection with the bounding boxes and save the figure."""
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()
        colors = self.colors * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=2))
            cl = p.argmax()
            text = f'{self.classes[cl]}'
            ax.text(xmin, ymin, text, fontsize=18, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.savefig(save_path + '/detection_' + source_name, bbox_inches='tight')
        plt.close()
