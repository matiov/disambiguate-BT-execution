"""Client to test the services in maskRCNN_ros."""

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

from typing import List

from object_recognition_interfaces.srv import GetObjects, UpdateObject
import rclpy
from rclpy.node import Node


class ObjectClient(Node):
    """Retrieve the detected objecs and modify the disambiguate value of one of them."""

    def __init__(self):
        super().__init__('object_client')

        self.get_objects = self.create_client(
            GetObjects, '/abb/sensors/camera/object_publisher/detection_srv')
        while not self.get_objects.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service Get not available, waiting again...')
        self.update_objects = self.create_client(
            UpdateObject, '/abb/sensors/camera/object_publisher/update_objects_srv')
        while not self.update_objects.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service Update not available, waiting again...')

    def get_objects_request(self):
        request = GetObjects.Request()
        self.object_future = self.get_objects.call_async(request)

    def update_objects_request(self, detected_objects: List, key: str):
        updated_obj = None
        for obj in detected_objects:
            if obj.header.frame_id != key:
                continue
            obj.disambiguated = True
            updated_obj = obj
        request = UpdateObject.Request()
        request.object = updated_obj
        self.update_future = self.update_objects.call_async(request)


def main(args=None):
    rclpy.init(args=args)

    minimal_client = ObjectClient()
    minimal_client.get_objects_request()

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
                minimal_client.get_logger().info(f'Result: {object_response.success}')
                minimal_client.update_objects_request(objects, 'banana_1')
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
    main()
