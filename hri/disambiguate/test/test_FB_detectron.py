"""Test script for the Facebook Detectron."""

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
from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T


torch.set_grad_enabled(False)


CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def __rescale_bboxes(
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
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=2))
        cl = p.argmax()
        text = f'{CLASSES[cl]}'
        ax.text(xmin, ymin, text, fontsize=18, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(save_path + '/current_scene_fbdet.jpg', bbox_inches='tight')
    plt.close()


def test_detectron():
    # get this file folder 'test'
    working_path = os.path.dirname(os.path.realpath(__file__))
    # get diambiguate folder
    disambiguate_path = Path(working_path).parent.absolute()
    data_directory = os.path.join(disambiguate_path, 'disambiguate/data')
    source_img_name = 'current_scene.jpg'

    fb_detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    fb_detr.eval()

    image = Image.open(os.path.join(data_directory, source_img_name))
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # mean-std normalize the input image (batch-size: 1)
    img = transform(image).unsqueeze(0)
    # propagate through the model
    outputs = fb_detr(img)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = __rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
    __plot_results(
        data_directory,
        source_img_name,
        image,
        probas[keep],
        bboxes_scaled
    )


if __name__ == '__main__':
    test_detectron()
