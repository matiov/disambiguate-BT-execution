"""Service that calls the HRI loop to disambiguate the scene."""

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

import cv2
import cv_bridge
from disambiguate.disambiguate import disambiguate_scene
from hri_interfaces.srv import Disambiguate
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


_cv_bridge = cv_bridge.CvBridge()


# TODO:
# 1. check how to get input (voice / text) from ROS execution

class DisambiguateService(Node):
    """Read the current image and perform disambiguation upon request."""

    def __init__(self):
        super().__init__('disambiguate_service')

        # Declare and read parameters
        self.declare_parameter('image_name', 'ambiguous_scene.jpg')
        self.declare_parameter('verbal_interaction', False)

        self.image_name = self.get_parameter('image_name').get_parameter_value().string_value
        self.verbal_interaction =\
            self.get_parameter('verbal_interaction').get_parameter_value().bool_value

        self.namespace = '/abb/sensors/camera'

        self.latest_color_ros_image = None
        self._color_subscriber = self.create_subscription(
            Image,
            self.namespace + '/rgb_to_depth/image_rect',
            self.color_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
        )

        self.task_target = None
        self.disambiguate_srv = self.create_service(
            Disambiguate,
            self.namespace + '/disambiguate_srv',
            self.disambiguate_callback
        )

    def color_callback(self, color_ros_image: Image):
        """
        Store the most recent color image.

        Args:
        ----
            color_ros_image: the color image to store.

        """
        self.get_logger().debug('Received color image.')
        self.latest_color_ros_image = color_ros_image

    def disambiguate_callback(self, request, response):
        # call HRI stuff and write code to determine if the thing worked!
        # task_target is list with a lable and a pose [x, y, z]
        # --> the bounding box is a rectangle [starting_point, ending_point] where:
        # --> starting_point and ending_point are X, Y values in pixels | bb = [Xs, Ys, Xe, Ye]
        self.get_logger().warn('Entering the Disambiguation loop.')

        root_path = '/home/wasp/abb/ros2/core_ws/src'
        repo_path = 'behavior-tree-learning/hri/disambiguate/disambiguate/data'
        save_path = os.path.join(root_path, repo_path)
        cv_image = _cv_bridge.imgmsg_to_cv2(self.latest_color_ros_image, 'bgr8')
        cv2.imwrite(os.path.join(save_path, self.image_name), cv_image)

        # Test command line interaction:
        print(f'Do you want to disambiguate the item: {request.category_str}?')
        answer = input('Yes, No? ')
        if answer.lower() == 'yes':
            print(f'Starting the disambiguation framework with query: {request.category_str}.')
            response.result, response.bounding_box, _ = disambiguate_scene(
                self.image_name,
                request.category_str,
                verbal=self.verbal_interaction
            )
        else:
            # Feed forward, case if we have already run it.
            response.result = True
            response.bounding_box = []

        return response


def main(args=None):
    rclpy.init(args=args)

    disambiguate_service = DisambiguateService()

    rclpy.spin(disambiguate_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
