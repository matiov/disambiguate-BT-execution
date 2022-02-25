"""Client to test the services in disambiguate_ros with the maskRCNN detection."""

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

import sys
from typing import List

from hri_interfaces.srv import Disambiguate
from object_recognition_interfaces.srv import GetObjects, UpdateObject
from perception_utils.process_detection import boundingbox_intersection
import rclpy
from rclpy.node import Node


class DetectionHRIClient(Node):
    """Call the disambiguation framework with maskRCNN detection running."""

    def __init__(self):
        super().__init__('disambiguate_client')

        self.namespace = '/abb/sensors/camera'

        self.disambiguate = self.create_client(
            Disambiguate, self.namespace + '/disambiguate_srv')
        while not self.disambiguate.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service Disambiguate not available, waiting again...')
        self.get_objects = self.create_client(
            GetObjects, self.namespace + '/detection_srv')
        while not self.get_objects.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service Get not available, waiting again...')
        self.update_objects = self.create_client(
            UpdateObject, self.namespace + '/update_objects_srv')
        while not self.update_objects.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service Update not available, waiting again...')

    def disambiguation_request(self, target: str):
        request = Disambiguate.Request()
        request.category_str = target
        self.future = self.disambiguate.call_async(request)

    def get_objects_request(self):
        request = GetObjects.Request()
        self.object_future = self.get_objects.call_async(request)

    def update_objects_request(
        self,
        detected_objects: List,
        object_class: str,
        bounding_box: List[int]
    ):
        updated_obj = None
        for obj in detected_objects:
            if obj.category_str != object_class:
                continue
            area, _ = boundingbox_intersection(bounding_box, list(obj.bounding_box))
            if area > 0:
                obj.disambiguated = True
            else:
                continue
            updated_obj = obj
        request = UpdateObject.Request()
        request.object = updated_obj
        self.update_future = self.update_objects.call_async(request)


def main():
    rclpy.init()

    minimal_client = DetectionHRIClient()
    minimal_client.get_objects_request()

    objects = None
    target = sys.argv[1]

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.object_future.done():
            try:
                object_response = minimal_client.object_future.result()
            except Exception as e:
                minimal_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                objects = object_response.objects
                if object_response.success:
                    minimal_client.disambiguation_request(target)
                break

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.future.done():
            try:
                response = minimal_client.future.result()
            except Exception as e:
                minimal_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                minimal_client.get_logger().info(f'Disambiguated? {response.result}')
                minimal_client.update_objects_request(
                    objects, target, list(response.bounding_box))
                break

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.update_future.done():
            try:
                result = minimal_client.update_future.result()
            except Exception as e:
                minimal_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                minimal_client.get_logger().info(f'Update result: {result.success}')
                break

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
