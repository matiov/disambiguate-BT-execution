"""Definition of the HRI behaviors."""

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

from typing import Any, List, Tuple

from behaviors.behavior_lists import BehaviorLists
from behaviors.common_behaviors import Behaviour, RandomSelector, RSequence
import bt_learning.learning_from_demo.lfd_behaviors as lfd_bt
import py_trees as pt
from py_trees.composites import Selector, Sequence
from robot_interface.hri_interface import HRIInterface
import yaml


NUMBER_REGEX = r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'


class HRIBehaviors(lfd_bt.Behaviors):
    """Defines all executable behaviors for the HRI pipeline."""

    def __init__(self, directory_path: str):
        """Directory_path is the path to the directory where planner settings are saved."""
        self.directory_path = directory_path
        self.behavior_list = None

    def get_behavior_list(self) -> BehaviorLists:
        """Parse the yaml file and returns the behavior list."""
        # initialize the dictionary an populate it while parsing the yaml
        condition_nodes = {}
        action_nodes = {}

        file = self.directory_path + '/BT_SETTINGS.yaml'
        if file is not None:
            with open(file) as f:
                bt_settings = yaml.load(f, Loader=yaml.FullLoader)
            try:
                node_name = bt_settings['condition_nodes']
                for _ in range(len(node_name)):
                    condition_nodes[node_name[_]] = []
            except KeyError:
                pass
            try:
                node_name = bt_settings['action_nodes']
                for _ in range(len(node_name)):
                    action_nodes[node_name[_]] = []
            except KeyError:
                pass

        self.behavior_list = BehaviorLists(
            condition_nodes=condition_nodes, action_nodes=action_nodes)

        return self.behavior_list

    def get_node_from_string(
        self,
        string: str,
        world_interface: HRIInterface,
        condition_parameters: Any
    ) -> Tuple[Behaviour or RSequence or RandomSelector or Selector or Sequence, bool]:
        """
        Return the Behavior Tree node given its string representation.

         Args
         ----
            string: name of the robot skill as string.
            world_interface: interface to the robot hardware.
            condition_parameters: pre- and post-conditions of the skill.

        Returns
        -------
            node: behavior tree node, eventually inherits from py_trees
            has_children: bool to determine if the node is a control node or a behavior.

        """
        has_children = False

        # Actions
        if string.startswith('disambiguate'):
            node = Disambiguate(string, world_interface)

        # Conditions
        elif string.startswith('clear'):
            node = SceneClear(string, world_interface)

        else:
            # get control node from the super class
            node, has_children = super().get_node_from_string(
                string, world_interface, condition_parameters)

        return node, has_children

    def get_actions(self, demonstration: Any) -> List[str]:
        """Get the HRI actions."""
        actions = ['disambiguate']

        return actions

    def get_conditions(self, demonstration: Any) -> List[str]:
        """Get the HRI conditions."""
        conditions = ['clear']

        return conditions


class Disambiguate(pt.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        world_interface: HRIInterface
    ):
        """
        Initialize the disambiguation.

        Args:
        ----
            name: name of the behavior.
            world_interface: interface to the robot.

        """
        super().__init__(name)
        self.world_interface = world_interface
        self.target = ''

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='scene_clear', access=pt.common.Access.WRITE)
        self.blackboard.register_key(key='target', access=pt.common.Access.READ)

        self.disambiguate = None

    def initialise(self):
        """Initialize the task as a thread."""
        # read the blackboard to retrieve the object to disambiguate
        print(f'Initializing Disambiguation with TARGET: {self.blackboard.target}')
        self.disambiguate = self.world_interface.disambiguate(self.blackboard.target)
        self.disambiguate.start()

    def update(self) -> pt.common.Status:
        """Return the status of the behavior."""
        if self.disambiguate.done():
            if self.disambiguate.result:
                print('Disambiguation successful.')
                self.blackboard.scene_clear = True
                return pt.common.Status.SUCCESS
            else:
                self.blackboard.scene_clear = False
                return pt.common.Status.FAILURE
        else:
            return pt.common.Status.RUNNING

    def terminate(self, new_status: pt.common.Status):
        """Terminate the task thread."""
        print(f'Terminating Disambiguation with status: {new_status}.')
        if new_status == pt.common.Status.INVALID and self.disambiguate is not None:
            self.disambiguate.terminate()

    def get_display_name(self) -> str:
        """Returnt the action name."""
        return 'Disambiguate Behavior'


class SceneClear(pt.behaviour.Behaviour):
    """Returns SUCCESS if the scene is not ambiguous."""

    def __init__(
        self,
        name: str,
        world_interface: HRIInterface
    ):
        """Initialize the condition."""
        super().__init__(name)

        self.world_interface = world_interface
        self.last_state = pt.common.Status.INVALID

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key='scene_clear', access=pt.common.Access.WRITE)
        # The initialization can also be done at BT creation.
        try:
            self.blackboard.scene_clear = True
        except KeyError as e:
            raise RuntimeError(f'Blackboard variable "scene_clear" not found [{e}].')

    def update(self) -> pt.common.Status:
        """Return the status of the behavior."""
        # read blackboard

        if self.blackboard.scene_clear:
            self.last_state = pt.common.Status.SUCCESS
            return pt.common.Status.SUCCESS
        else:
            self.last_state = pt.common.Status.FAILURE
            return pt.common.Status.FAILURE

    def get_display_name(self) -> str:
        """Returnt the condition name."""
        return 'Scene Clear?'
