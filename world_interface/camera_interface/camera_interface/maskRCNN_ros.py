"""Node that calls the MaskRCNN module for object detection."""

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

from copy import copy
import os
from typing import Dict

import cv2
import cv_bridge
from execution_interfaces.msg import TaskStatus
from geometry_msgs.msg import TransformStamped
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import object_detection.maskRCNN as maskRCNN
import object_detection.mask_centroid_heuristic as heuristic
from object_recognition_interfaces.msg import BoolList, BoolMask, DetectedObject
from object_recognition_interfaces.srv import GetObjects, UpdateObject
from perception_utils.homogeneous_matrix import homogeneous_matrix
import perception_utils.pinhole_camera as camera_utils
import perception_utils.process_detection as detect_utils
from perception_utils.transformations import translation_matrix
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


_cv_bridge = cv_bridge.CvBridge()


class MaskRCNNDetection(Node):
    """
    This node performs object detection and publishes object frames using heuristics.

    Moreover, it exposes two services to get the detected objects and update their informations.
    An example can be to use it together with the disambiguation framework to change the <unique>
    value of an item. Frame names will be updated consequently.
    """

    def __init__(self):
        super().__init__('object_publisher')

        # Parameter description
        objects_ = ParameterDescriptor(
            description='List of objects to detect.')
        freq_ = ParameterDescriptor(
            description='Timer frequence for the object detection.')
        detect_ = ParameterDescriptor(
            description='Perform detection only.',
            additional_constraints='If false, a frame is attached to every object.'
        )
        rotation_ = ParameterDescriptor(
            description='Use the same rotation as the base frame.',
            additional_constraints='If false, rotation is given from the heuristic.'
        )

        # Declare and read parameters
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('objects', ['banana'], objects_)
        self.declare_parameter('detection_freq', 1.0, freq_)
        self.declare_parameter('only_detect', True, detect_)
        self.declare_parameter('simple_rotation', True, rotation_)

        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.objects = self.get_parameter('objects').get_parameter_value().string_array_value
        self.frequence = self.get_parameter('detection_freq').get_parameter_value().double_value
        self.only_detect = self.get_parameter('only_detect').get_parameter_value().bool_value
        self.simple_rot = self.get_parameter('simple_rotation').get_parameter_value().bool_value

        self.camera_proj_matrix = None
        self.camera_model = None
        self.latest_color_ros_image = None
        self.latest_depth_ros_image = None
        self.depth_image_avg = None
        self.image_saved = False

        self.task = None
        self.status = None
        self.trigger_place = False
        self.standby_detection_counter = 0
        self.first_detection = True
        self.detected_objects = {}
        self.objects_list = []

        # Initialize MaskRCNN
        self.get_logger().warn('Initializing detection.')
        self.detectron = maskRCNN.MaskRCNN_detectron()

        # ---------------- SUBSCRIBERS
        self._camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '~/camera_info',
            self.camera_info_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data
        )
        self._color_subscriber = self.create_subscription(
            Image,
            '~/rgb_to_depth/image_rect',
            self.color_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
        )
        self._depth_subscriber = self.create_subscription(
            Image,
            '~/depth/image_rect',
            self.depth_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
        )
        self._task_subscriber = self.create_subscription(
            TaskStatus,
            '~/robot/task_status',
            self.task_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
        )

        # ---------------- PUBLISHER
        self._masked_image_publisher = self.create_publisher(
            Image,
            '~/rgb_to_depth/image_masked',
            10
        )

        # ---------------- TF
        # Initialize the transform broadcaster
        self.br = TransformBroadcaster(self)
        # Initialize the transform listener
        # Make the buffer retain past transforms for 5 seconds
        self.tf_buffer = Buffer(rclpy.duration.Duration(seconds=5))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        timer_period = 1.0/self.frequence  # seconds
        self.timer = self.create_timer(timer_period, self.maskRCNN_detection)

        # ---------------- SERVICES
        self.get_obj_srv = self.create_service(
            GetObjects, '~/detection_srv', self.get_objects_callback)
        self.update_obj_srv = self.create_service(
            UpdateObject, '~/update_objects_srv', self.update_object_callback)

    # --------------------------------------------------------------------------------------------
    # ---------------------------------------- SUBSCRIBERS ---------------------------------------
    # --------------------------------------------------------------------------------------------
    def camera_info_callback(self, camera_info: CameraInfo) -> None:
        """
        Store the projection matrix for the camera.

        Args
        ----
            camera_info:
                the camera info for the color image and depth image;
                depth and color are assumed to have the same camera info.

        """
        self.get_logger().debug('Received camera info.')
        self.camera_proj_matrix = np.array(camera_info.p).reshape((3, 4))
        self.camera_model = camera_utils.Camera(
            width=camera_info.width,
            height=camera_info.height,
            fx=self.camera_proj_matrix[0, 0],
            fy=self.camera_proj_matrix[1, 1],
            cx=self.camera_proj_matrix[0, 2],
            cy=self.camera_proj_matrix[1, 2],
            pixel_center=0.0,  # not sure what this is in ROS
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
        # This is used to save the depth image for later testing.
        if not self.image_saved:
            root_path = '/home/wasp/abb/ros2/core_ws/src'
            repo_path = 'behavior-tree-learning/hri/disambiguate/disambiguate/data'
            save_path = os.path.join(root_path, repo_path)
            cv_image = _cv_bridge.imgmsg_to_cv2(color_ros_image, 'bgr8')
            cv2.imwrite(os.path.join(save_path, 'color_img.jpg'), cv_image)
            self.image_saved = True

    def depth_callback(self, depth_ros_image: Image):
        """
        Store the most recent aligned depth image.

        Args:
        ----
            depth_ros_image: the aligned depth image to store.

        """
        self.get_logger().debug('Received depth image.')
        self.latest_depth_ros_image = depth_ros_image

    def task_callback(self, task_status: TaskStatus):
        """Retrieve the task status and save it locally."""
        if task_status.current_task != 'IDLE':
            self.task = task_status.current_task
        if task_status.execution_status != 'INVALID':
            self.status = task_status.execution_status

    # --------------------------------------------------------------------------------------------
    # ---------------------------------------- PUBLISHERS ----------------------------------------
    # --------------------------------------------------------------------------------------------
    def maskRCNN_detection(self):
        """Detect and process image with MaskRCNN detectron."""
        # convert ROS image to numpy array
        color_img_np = camera_utils.ros_img_to_np_img(self.latest_color_ros_image, 'rgb8')

        # predict instances
        detection_result = self.detectron.detect(color_img_np)
        # Detect frisbee as bowl
        if 'frisbee' in detection_result['names']:
            idx = detection_result['names'].index('frisbee')
            detection_result['names'][idx] = 'bowl'
            detection_result['class_ids'][idx] = 46

        results = copy(detection_result)
        np_image = copy(color_img_np)
        self.__publish_masks(results, np_image[:, :, ::-1])

        if not self.only_detect:
            self.get_logger().warn(f'Executing {self.task}, with status {self.status}.')
            if (self.task == 'Place' and self.status == 'RUNNING') or\
               ((self.task == 'Pick' and self.status != 'FAILURE') and
               self.standby_detection_counter < 5):
                self.get_logger().warn('Pausing object detection while Manipulating.')
                self.trigger_place = True
                self.standby_detection_counter += 1
            else:
                self.detected_objects, self.objects_list, self.first_detection =\
                    detect_utils.process_maskRCNN_results(
                        detection_result,
                        self.detected_objects,
                        self.objects_list,
                        self.first_detection,
                        only_update=self.trigger_place
                    )
                self.get_logger().warn(f'Detected: {self.objects_list}.')
                self.get_logger().warn(f'Keys: {self.detected_objects.keys()}.')

                self.get_logger().debug('Publishing transforms.')
                self.__publish_transformations(self.detected_objects)
                self.trigger_place = False
                self.standby_detection_counter = 0

    # --------------------------------------------------------------------------------------------
    # ----------------------------------------- SERVICES -----------------------------------------
    # --------------------------------------------------------------------------------------------
    def get_objects_callback(self, request, response):
        """Get the current detected objects."""
        if not self.detected_objects:
            # The dictionary is empty!
            response.success = False
            response.message = 'No object detected!'
            response.objects = []
        else:
            response.success = True
            response.message = ''
            for i, key in enumerate(self.detected_objects.keys()):
                object_item = DetectedObject()
                object_item.header.stamp = self.get_clock().now().to_msg()
                object_item.header.frame_id = key
                object_item.category_str = self.detected_objects[key]['category']
                object_item.object_id = self.detected_objects[key]['id']
                object_item.bounding_box = self.detected_objects[key]['bounding_box'].tolist()
                mask_list = self.detected_objects[key]['mask'].tolist()
                mask_msg = BoolMask()
                row_msg = BoolList()
                for i, row in enumerate(mask_list):
                    row_msg.list = row
                    mask_msg.bool_list.append(row_msg)
                object_item.mask = mask_msg
                object_item.disambiguated = self.detected_objects[key]['unique']
                response.objects.append(object_item)
            self.get_logger().warn('Finished creating response message.')

        return response

    def update_object_callback(self, request, response):
        """Update the dictionary of detected objects with the incoming data."""
        object_name = request.object.header.frame_id
        try:
            object_data = self.detected_objects[object_name]
        except KeyError:
            response.message = 'Something is wrong, trying to update the wrong object!'
            response.success = False
            return response
        # Now we can create a new entry on the dictionary of detected objects.
        area, _ = detect_utils.boundingbox_intersection(
            request.object.bounding_box, object_data.__getitem__('bounding_box'))
        if not area > 0:
            response.message = 'Something is wrong, trying to update the wrong object!'
            response.success = False
            return response

        # Sanity check, is the object already disambiguated?
        if object_data.__getitem__('unique') is False:
            new_object = detect_utils.ObjectData(copy(detect_utils.TEMPLATE_OBJ_DICT))
            new_object.__setitem__('category', request.object.category_str)
            new_object.__setitem__('bounding_box', copy(object_data.__getitem__('bounding_box')))
            new_object.__setitem__('mask', copy(object_data.__getitem__('mask')))
            # The object is unique!
            new_object.__setitem__('id', 0)
            new_object.__setitem__('unique', request.object.disambiguated)
            new_object_name = request.object.category_str

            self.detected_objects[new_object_name] = new_object
            # Since this object is not ambiguous anymore, we can delete from the dictionary
            # the element corresponding to 'class_id'.
            del self.detected_objects[object_name]

        response.success = True
        response.message = f'Updated infrormation for object {request.object.category_str}.'

        return response

    # --------------------------------------------------------------------------------------------
    # ----------------------------------------- AUXILIARY ----------------------------------------
    # --------------------------------------------------------------------------------------------
    def __publish_transformations(self, detected_objects: Dict):
        """Publish the a TF frame for every detected object."""
        now = rclpy.time.Time()
        heuristic_params = heuristic.Parameters()
        heuristic_params.use_max = True

        # Echo the TF between camera and base frame
        constant_tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, now)
        base_T_camera = homogeneous_matrix(
            constant_tf.transform.translation, constant_tf.transform.rotation)

        # Iterate the detected objects and compute the transform
        for key in detected_objects.keys():
            if self.detected_objects[key]['category'] not in self.objects:
                self.get_logger().debug(f'Key {key} not relevant, ignoring.')
                continue
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.base_frame
            t.child_frame_id = str(key)
            # Compute centroid of the bounding box
            try:
                self.get_logger().debug(
                    f'Mask: {self.detected_objects[key]["mask"].shape}')
                self.get_logger().debug(
                    f'Bounding box: {self.detected_objects[key]["bounding_box"]}')
                point, orientation = heuristic.compute_mask_frame(
                    self.objects,
                    self.detected_objects[key]['category'],
                    self.detected_objects[key]['bounding_box'],
                    self.detected_objects[key]['mask'],
                    heuristic_params
                )
            except ValueError:
                continue
            # Transform the centroid in a 3D point
            self.get_logger().debug(f'{key} 2D point: {point}.')
            np_depth_image = camera_utils.ros_img_to_np_img(self.latest_depth_ros_image, 'mono16')
            if self.get_logger().get_effective_level() == 'DEBUG':
                # For debugging:
                plt.imshow(np_depth_image)
                plt.show()

            point_3D = camera_utils.pixel_to_point_transform(
                point, np_depth_image, self.camera_model)
            self.get_logger().debug(f'{key} 3D point: {point_3D}.')

            # Centroid transform matrix
            camera_T_obj = translation_matrix(point_3D)
            self.get_logger().debug(f'Translation:\n {camera_T_obj}.')
            base_T_obj = base_T_camera@camera_T_obj
            self.get_logger().debug(f'Transform:\n {base_T_obj}.')

            # Build the TF message
            t.transform.translation.x = base_T_obj[0, 3]
            t.transform.translation.y = base_T_obj[1, 3]
            t.transform.translation.z = base_T_obj[2, 3]
            # Use same orientation as the base frame
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            if not self.simple_rot:
                # Get rotation from heuristic
                try:
                    t.transform.rotation.x = orientation[0]
                    t.transform.rotation.y = orientation[1]
                    t.transform.rotation.z = orientation[2]
                    t.transform.rotation.w = orientation[3]
                except AssertionError:
                    self.get_logger().warn(f'Orientation error for item {key}.')
                    self.get_logger().warn(f'Got value: {orientation}.')

            # Send the transformation
            self.br.sendTransform(t)

    def __publish_masks(self, result: Dict, np_image: np.array):
        """Publish an Image with the detected masks."""
        masked_image = self.__visualize(result, np_image)
        cv_result = np.zeros(shape=masked_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(masked_image, cv_result)
        image_msg = _cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        image_msg.header = self.latest_color_ros_image.header
        self._masked_image_publisher.publish(image_msg)

    def __visualize(self, result: Dict, image: np.ndarray):
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        self.detectron.print_results_on_image(image, result, axes=axes)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result


def main(args=None):
    rclpy.init(args=args)

    maskRCNN_detection = MaskRCNNDetection()

    rclpy.spin(maskRCNN_detection)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
