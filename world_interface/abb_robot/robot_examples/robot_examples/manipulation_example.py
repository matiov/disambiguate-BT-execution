"""
Example of a manipulation task.

The robot skills are run using the Behavior Tree policy representation.
"""

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

import threading
import time
from typing import Any, List
import numpy as np
import py_trees as pt
import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from robot_interface.online_yumi_interface import OnlineYuMiInterface
from robot_behaviors.yumi_behaviors.simple_behaviors import PickBehavior, PlaceBehavior


class TestNode(Node):

    def __init__(self):
        super().__init__('test_manipulation')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


def move_object(
    node: Node,
    target_object: str,
    positions: List[np.ndarray],
    orientations: List[np.ndarray] = None,
    world_interface: Any = None,
    force_arm: str = None):
    """
    Move the given object.

    Args:
    ----
        node: the ROS node giving access to the TF buffer.
        target_object: frame_id attached to the object to move.
        position: target positions for pick and place.
        orientation: target orientations for pick and place..
        world_interface: interface to the robot.
                         if None, OnlineYuMiInterface is used.
        force_arm: target arm to perform the motion.
                   if None, a simple heuristic decides it.
                   if given, must be either T_ROB_L or T_ROB_R.

    """
    if world_interface is None:
        world_interface = OnlineYuMiInterface(node)

    pick_position = positions[0]
    pick_orientation = orientations[0]
    place_position = positions[1]
    place_orientation = orientations[1]

    if force_arm is None:
        rob_task = 'T_ROB_R'
        if place_position[1] > 0.0:
            rob_task = 'T_ROB_L'
    else:
        rob_task = force_arm


    behavior = PickBehavior(
        'pick',
        world_interface, 
        target_object,
        pick_position,
        pick_orientation,
        tool=None,
        rob_task=rob_task
    )
    print('Pick behavior initialized in the interface.')

    behavior.initialise()
    while behavior.update() is not pt.common.Status.SUCCESS:
        time.sleep(0.1)
        status = behavior.update()
        behavior.terminate(status)

    behavior = PlaceBehavior(
        'place',
        world_interface,
        place_position,
        place_orientation,
        frame=world_interface.base_frame,
        rob_task=rob_task)
    print('Place behavior initialized in the interface.')

    behavior.initialise()
    while behavior.update() is not pt.common.Status.SUCCESS:
        time.sleep(0.1)
        status = behavior.update()
        behavior.terminate(status)


def main():
    rclpy.init()

    test_node = TestNode()

    world_interface = OnlineYuMiInterface(
        test_node, base_frame='abb_yumi_base_link', namespace='/abb')
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(test_node,))
    spin_thread.start()

    # If pick position and orientation are None, then the pick pose is computed from the frame.
    positions = [None, np.array([0.4, 0.2, 0.06])]
    orientations = [None, np.array([0.0, 0.707107, 0.707107, 0.0])]

    # Make sure that a frame exists with the name of the parameter target object.
    move_object(
        test_node,
        'knife',
        positions,
        orientations,
        world_interface=world_interface,
        force_arm='T_ROB_R'
    )

    # Shut down execution.
    world_interface.__del__()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
