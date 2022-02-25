"""Util functions for the Pinhole Camera."""

# Copyright (c) 2021 Leonard Bruns
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

from typing import Tuple

import cv_bridge
import numpy as np
import open3d as o3d
from sensor_msgs.msg import Image

_cv_bridge = cv_bridge.CvBridge()


class Camera:
    """
    Pinhole camera parameters.

    This class allows conversion between different pixel conventions, i.e., pixel
    center at (0.5, 0.5) (as common in computer graphics), and (0, 0) as common in
    computer vision.

    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        s: float = 0.0,
        pixel_center: float = 0.0,
    ):
        """
        Initialize camera parameters.

        Note that the principal point is only fully defined in combination with
        pixel_center.

        The pixel_center defines the relation between continuous image plane
        coordinates and discrete pixel coordinates.

        A discrete image coordinate (x, y) will correspond to the continuous
        image coordinate (x + pixel_center, y + pixel_center). Normally pixel_center
        will be either 0 or 0.5. During calibration it depends on the convention
        the point features used to compute the calibration matrix.

        Note that if pixel_center == 0, the corresponding continuous coordinate
        interval for a pixel are [x-0.5, x+0.5). I.e., proper rounding has to be done
        to convert from continuous coordinate to the corresponding discrete coordinate.

        For pixel_center == 0.5, the corresponding continuous coordinate interval for a
        pixel are [x, x+1). I.e., floor is sufficient to convert from continuous
        coordinate to the corresponding discrete coordinate.

        Args:
        ----
            width: Number of pixels in horizontal direction.
            height: Number of pixels in vertical direction.
            fx: Horizontal focal length.
            fy: Vertical focal length.
            cx: Principal point x-coordinate.
            cy: Principal point y-coordinate.
            s: Skew.
            pixel_center: The center offset for the provided principal point.

        """
        # focal length
        self.fx = fx
        self.fy = fy

        # principal point
        self.cx = cx
        self.cy = cy

        self.pixel_center = pixel_center

        # skew
        self.s = s

        # image dimensions
        self.width = width
        self.height = height

    def get_o3d_pinhole_camera_parameters(self) -> o3d.camera.PinholeCameraParameters():
        """
        Convert camera to Open3D pinhole camera parameters.

        Open3D camera is at (0,0,0) looking along positive z axis (i.e., positive z
        values are in front of camera). Open3D expects camera with pixel_center = 0
        and does not support skew.

        Returns
        -------
            The pinhole camera parameters.

        """
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(0)
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)
        params.extrinsic = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        return params

    def get_pinhole_camera_parameters(self, pixel_center: float) -> Tuple:
        """
        Convert camera to general camera parameters.

        Args
        ----
            pixel_center:
                At which ratio of a square the pixel center should be for the resulting
                parameters. Typically 0 or 0.5. See class documentation for more info.

        Returns
        -------
            - fx, fy: The horizontal and vertical focal length
            - cx, cy:
                The position of the principal point in continuous image plane
                coordinates considering the provided pixel center and the pixel center
                specified during the construction.
            - s: The skew.

        """
        cx_corrected = self.cx - self.pixel_center + pixel_center
        cy_corrected = self.cy - self.pixel_center + pixel_center
        return self.fx, self.fy, cx_corrected, cy_corrected, self.s


def ros_img_to_np_img(
    image: Image,
    desired_encoding: str = 'passthrough'
) -> np.ndarray:
    """
    Convert ROS image to numpy array.

    Args
    ----
        image: The ROS image message to be converted to a numpy array.
        desired_encoding: Output encoding string, see http://wiki.ros.org/cv_bridge/.

    Returns
    -------
        The image as a numpy array

    """
    if image.encoding.lower() == '16uc1':
        image.encoding = 'mono16'
    np_img = _cv_bridge.imgmsg_to_cv2(image, desired_encoding)

    return np_img


def pixel_to_point_transform(
    pixel: np.ndarray,
    depth_image: np.ndarray,
    camera: Camera
) -> np.ndarray:
    """
    Convert 2D pixel into a 3D point.

    Args
    ----
        pixel: the 2D pixel point to convert.
        depth_image: The depth image to convert to pointcloud, shape (H,W).
        camera: The camera used to lift the points.

    Returns
    -------
        The 3D point in the camera depth frame.

    """
    fx, fy, cx, cy, _ = camera.get_pinhole_camera_parameters(0.0)
    # Parameters: 504.039794921875 504.1357116699219 324.18121337890625 331.11407470703125

    px = pixel[0]
    py = pixel[1]

    depth_queue = []
    # Take depth average in the surrounding area of 10x10 pixels
    for x_ in range(5):
        for y_ in range(5):
            try:
                depth_queue.append(depth_image[py + (y_ + 1)][px + (x_ + 1)])
            except IndexError:
                continue
            try:
                depth_queue.append(depth_image[py + (y_ + 1)][px - (x_ + 1)])
            except IndexError:
                continue
            try:
                depth_queue.append(depth_image[py - (y_ + 1)][px + (x_ + 1)])
            except IndexError:
                continue
            try:
                depth_queue.append(depth_image[py - (y_ + 1)][px - (x_ + 1)])
            except IndexError:
                continue
            # Remove 0 values
            if 0 in depth_queue:
                depth_queue.remove(0)

    depth = sum(depth_queue)/len(depth_queue)
    # depth = depth_image[py, px]
    # Depth is in [mm] so we convert in [m]
    z = depth/1e03
    x = (px - cx)*z/fx
    y = (py - cy)*z/fy

    point_3D = np.array([x, y, z])

    return point_3D
