"""Interface to use behaviors to handle HRI."""

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

import time
from threading import Thread
from typing import List

from hri_interfaces.srv import Disambiguate
from object_recognition_interfaces.msg import DetectedObject
from object_recognition_interfaces.srv import GetObjects, UpdateObject
from perception_utils.process_detection import boundingbox_intersection
from rclpy.client import Client
from rclpy.node import Node
import robot_interface.interface as wi


class HRIInterface(wi.HRIInterface):
    """Interface to the HRI pipeline."""

    def __init__(
        self,
        node: Node,
        base_frame: str,
        namespace: str = ''
    ):
        """
        Initialize the communication with ROS.

        Args:
        ----
            node: ROS node that is spinning.
            base_frame: base frame of the robot.
            namespace: node namespace for creating ROS clients and subscriptions.

        """
        super().__init__(node, base_frame, namespace)
        self.ns = namespace + '/sensors/camera'

        # HRI services
        self.disambiguate_srv = self.node.create_client(
            Disambiguate, self.ns + '/disambiguate_srv')
        self.get_objects_srv = self.node.create_client(
            GetObjects, self.ns + '/detection_srv')
        self.update_objects_srv = self.node.create_client(
            UpdateObject, self.ns + '/update_objects_srv')

        while not self.disambiguate_srv.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('Waiting for ' + self.ns + '/disambiguate_srv')
        while not self.get_objects_srv.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for ' + self.ns + '/detection_srv')
        while not self.update_objects_srv.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for ' + self.ns + '/update_objects_srv')

    def __del__(self):
        """Destroy clients and subscriptions."""
        self.node.destroy_client(self.disambiguate_srv)
        self.node.destroy_client(self.get_objects_srv)
        self.node.destroy_client(self.update_objects_srv)

    def terminate(self):
        self.preempted = True

    def disambiguate(
        self,
        target: str
    ) -> Thread:
        """Call the disambiguation pipeline with the target request."""
        return DisambiguateTask(
            target, self.get_objects_srv, self.disambiguate_srv, self.update_objects_srv)


class HRITask(wi.Task, Thread):
    """Thread interface for the HRI framework."""

    def __init__(
        self,
        get_objects: Client,
        disambiguate: Client,
        update_objects: Client
    ):
        """
        Initialize a HRI task with by linking the ROS clients.

        Args:
        ----
            get_objects: Client that returns the detected objects.
            disambiguate: Client that starts the disambiguation framework.
            update_objects: : Client that updates the detected objects.

        """
        super().__init__()
        self.get_objects_client = get_objects
        self.disambiguate_client = disambiguate
        self.update_objects_client = update_objects

    def get_objects(self) -> List[DetectedObject]:
        """Get the detected objects."""
        request = GetObjects.Request()
        response = self.get_objects_client.call(request)
        if not response.success:
            raise RuntimeError(f'Get objects failed. Error: {response.message}')
        else:
            return response.objects

    def update_objects(
        self,
        detected_objects: List,
        object_class: str,
        bounding_box: List[int]
    ) -> bool:
        """Update the detected objects."""
        updated_obj = None
        for obj in detected_objects:
            if obj.category_str != object_class:
                continue
            print(f'Intersecting BB {bounding_box} with BB {list(obj.bounding_box)}')
            area, _ = boundingbox_intersection(bounding_box, list(obj.bounding_box))
            if area > 0:
                obj.disambiguated = True
            else:
                continue
            updated_obj = obj
        request = UpdateObject.Request()
        if updated_obj is not None:
            request.object = updated_obj
        response = self.update_objects_client.call(request)
        return response.success


class DisambiguateTask(HRITask):

    def __init__(self, target: str, *args):
        super().__init__(*args)

        self.target = target
        self.result = None

    def done(self):
        if self.result is not None:
            return self.result
        else:
            return False

    def run(self):
        """Call the disambiguation pipeline with the target request."""
        try:
            objects = self.get_objects()
        except RuntimeError:
            print('Unable to retrieve objects.')
            self.terminate()
        disambiguate_request = Disambiguate.Request()
        disambiguate_request.category_str = self.target
        response = self.disambiguate_client.call(disambiguate_request)
        target_bb = list(response.bounding_box)
        if not response.result:
            print(f'Not able to disambiguate object {self.target}')
            self.result = False
        elif target_bb != []:
            print(f'Updating info for item {self.target} with contour {target_bb}.')
            self.result = self.update_objects(objects, self.target, target_bb)
            # Add delay to allow disambiguation service to update the objects
            time.sleep(10)
        else:
            # We just assume that we don't want to run disambiguation
            self.result = True

    def terminate(self):
        self.result = False
