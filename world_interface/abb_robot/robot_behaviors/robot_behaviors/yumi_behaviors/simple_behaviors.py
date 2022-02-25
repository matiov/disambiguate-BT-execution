"""Simple YuMi behaviors."""

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

from typing import Any
import py_trees as pt

import numpy as np

# Note: The tasks are threads, so the function 'start()' calls the function 'run()'.
#       The thread function 'run()' is overwritten in rapid_interface.py


class PickBehavior(pt.behaviour.Behaviour):

    def __init__(
        self,
        name: str,
        world_interface: Any,
        target_object: str,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        tool: str = 'gripper',
        rob_task: str = 'T_ROB_R'
    ):
        """
        Initialize the Pick behavior.

        Args:
        ----
            name: name of the Action behavior.
            world_interface: interface to the robot.
            target_object: frame_id attached to the object to move.
            position: target pick position.
            orientation: target pick orientation.
            tool: the tool used to perform the picking.
            rob_task: either the right or left arm of the robot.

        """
        self.world_interface = world_interface
        self.object = target_object
        self.tool = tool
        self.position = position
        self.orientation = orientation
        self.rob_task = rob_task
        self.picking_task = None
        self.pick_pose = None

        super().__init__(name)

    def initialise(self):
        self.picking_task = self.world_interface.pick(
            self.position, self.orientation, self.object, self.tool, self.rob_task)
        print('Starting the picking task.')
        self.picking_task.start()

    def update(self):
        if not self.picking_task.is_alive():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.RUNNING

    def terminate(self, new_status):
        # Clear locks
        if hasattr(self.world_interface, 'state'):
            self.world_interface.state.blocking = False

        if new_status == pt.common.Status.INVALID and self.picking_task is not None:
            self.picking_task.terminate()

    def get_display_name(self):
        return f'Pick {self.object.category_str}.'


class PlaceBehavior(pt.behaviour.Behaviour):

    def __init__(
        self,
        name: str,
        world_interface: Any,
        position: np.ndarray,
        orientation: np.ndarray,
        frame: str,
        tool: str = 'gripper',
        rob_task: str = 'T_ROB_R'
    ):
        """
        Initialize the Place behavior.

        Args:
        ----
            name: name of the Action behavior.
            world_interface: interface to the robot.
            position: target place position.
            orientation: target place orientation.
            frame: reference frame for the action.
            tool: the tool used to perform the picking.
            rob_task: either the right or left arm of the robot.

        """
        self.world_interface = world_interface
        self.tool = tool
        self.position = position
        self.orientation = orientation
        self.frame = frame
        self.rob_task = rob_task
        self.placing_task = None

        super().__init__(name)

    def initialise(self):
        self.placing_task = self.world_interface.place(
            self.position, self.orientation, self.frame, tool=self.tool, rob_task=self.rob_task)
        print('Starting the placing task.')
        self.placing_task.start()

    def update(self):
        if not self.placing_task.is_alive():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.RUNNING

    def terminate(self, new_status):
        # Clear locks
        if hasattr(self.world_interface, 'state'):
            self.world_interface.state.blocking = False

        if new_status == pt.common.Status.INVALID and self.placing_task is not None:
            self.placing_task.terminate()

    def get_display_name(self):
        return f'Placing.'


class HomeBehavior(pt.behaviour.Behaviour):

    def __init__(
        self,
        name: str,
        world_interface: Any,
        rob_task: str = 'T_ROB_R'
    ):
        """
        Send Arm to Home configuration.

        Args:
        ----
            name: name of the Action behavior.
            world_interface: interface to the robot.
            rob_task: either the right or left arm of the robot.

        """
        self.world_interface = world_interface
        self.rob_task = rob_task

        super().__init__(name)

    def initialise(self):
        self.home_task = self.world_interface.home(self.rob_task)
        print('Bringing arm to home configuration')
        self.home_task.start()

    def update(self):
        if not self.home_task.is_alive():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.RUNNING

    def terminate(self, new_status):
        # Clear locks
        if hasattr(self.world_interface, 'state'):
            self.world_interface.state.blocking = False

        if new_status == pt.common.Status.INVALID and self.home_task is not None:
            self.home_task.terminate()

    def get_display_name(self):
        return 'Arm to home configuration.'
