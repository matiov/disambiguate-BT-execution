# Overview

This package uses [MaskRCNN](https://github.com/matterport/Mask_RCNN/tree/master), an object detection and segmentation algorithm to obtain objects masks.


# Installation

Follow the following instructions to download the MaskRCNN detection algorithm:

1. Instal the Mask-RCNN repository as described in [here](https://github.com/matterport/Mask_RCNN#installation). When running the [test](./object_detection/test/test_maskRCNN.py) there will be many subsequent errors due to vesion mismatch in `tensorflow` and `numpy`. This seems to be a very well known and used method to generate masks, so it is easy to find answers to the errors in internet. It is mainly about modifying some lines in the [model](https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py) script.



# Usage


### With ROS (temporary)


### Without ROS


(**TODO**: make a better README.md with tutorials and explaination and link to paeprs).
