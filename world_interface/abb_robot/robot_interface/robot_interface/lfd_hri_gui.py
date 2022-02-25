"""
Main node launching the GUI for the LfD framework.

In this node:
 - the Mobile Base is NOT enabled.
 - the perception uses MaskRCNN for object detection.
 - it is possible to enable HRI to disambiguate the scene.
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

from distutils.dir_util import copy_tree
import glob
import logging
import os
import subprocess
from tempfile import TemporaryDirectory
import threading

from bt_learning.learning_from_demo.clustering import find_equivalent_actions
from bt_learning.learning_from_demo.debug import BTVisualizer
from bt_learning.learning_from_demo.learning import learn_tree
import bt_learning.learning_from_demo.learning_postprocessing as post_prox
import bt_learning.learning_from_demo.render_tree as tree_render
import numpy as np
from py_trees.blackboard import Blackboard
from py_trees.display import render_dot_tree
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
import robot_behaviors.mobile_base_behaviors.lfd_actions as base_actions
from robot_behaviors.lfd_behaviors import LfDBehaviors
from robot_behaviors.mobile_yumi_behaviors.lfd_behaviors import MobileYuMiBehaviors
import robot_behaviors.yumi_behaviors.lfd_actions as yumi_actions
from robot_interface.demonstration import RobotDemonstration
from robot_interface.hri_interface import HRIInterface
from robot_interface.offline_yumi_itnerface import OfflineInterface
from robot_interface.online_yumi_interface import OnlineYuMiInterface
from simulation.py_trees_interface import PyTree
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import yaml

# OS dependent import
if os.name == 'nt':  # Windows
    import PySimpleGUIQt as gui
elif os.name == 'posix':  # Linux Ubuntu
    import PySimpleGUI as gui

logging.basicConfig(level=logging.NOTSET)


class TaskInterface(OnlineYuMiInterface, HRIInterface):

    def __init__(self, node: Node, base_frame: str, namespace: str = ''):
        super().__init__(node, base_frame, namespace)


class LfDGUI(Node):

    def __init__(self):
        super().__init__('lfd_hri_gui')

        # Parameter description
        frames_ = ParameterDescriptor(
            description='List of object frames to consider in the learning of a Behavior Tree.')
        tick_ = ParameterDescriptor(
            description='Tick signal frequence for the Behavior Tree.')
        robot_ = ParameterDescriptor(
            description='Use the GUI with or without the robot (online/offline).',
            additional_constraints='If false, it is only possible to display the Behavior Tree.'
        )
        hri_ = ParameterDescriptor(
            description='Enable Human Robot Interaction during the execution of the task.',
            additional_constraints='If true, a subtree calling the disambiguation srv is added.'
        )

        # Declare parameters
        self.declare_parameter('namespace', '')
        self.declare_parameter('default_frame', 'yumi_base_link')
        self.declare_parameter('frames', ['yumi_base_link', 'map'], frames_)
        self.declare_parameter('ee_frame', 'right_gripper_fingers')
        self.declare_parameter('base_frame', 'yumi_base_link')
        self.declare_parameter('bt_tick_freq', 0.4, tick_)
        self.declare_parameter('has_robot', False, robot_)
        self.declare_parameter('use_hri', True, hri_)

        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        # TODO: get those from the detection algorithm instead of hard coding them.
        self.default_frame =\
            self.get_parameter('default_frame').get_parameter_value().string_value
        self.all_frames = self.get_parameter('frames').get_parameter_value().string_array_value
        self.ee_frame = self.get_parameter('ee_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tick_freq = self.get_parameter('bt_tick_freq').get_parameter_value().double_value
        self.has_robot = self.get_parameter('has_robot').get_parameter_value().bool_value
        self.use_hri = self.get_parameter('use_hri').get_parameter_value().bool_value

        # Add base frame to the list of available frames
        self.all_frames.append(self.base_frame)

        # Make the buffer retain past transforms for 5 seconds
        self.tf_buffer = Buffer(rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.has_robot:
            # TODO: set up the interface to include also Mobile Base stuff
            # maybe we can have 2 separate interfaces: yumi interface and mobile base interface
            # since one communicates with RAPID while the other is only ROS based
            self.world_interface = OnlineYuMiInterface(
                node=self, base_frame=self.base_frame, namespace=self.namespace)
            if self.use_hri:
                self.world_interface = TaskInterface(
                    node=self, base_frame=self.base_frame, namespace=self.namespace)

            print('Initializing INTERFACES')

        self.tree_dir = None
        self.demo_dir = None
        self.tree_timer = None
        self.bt = None
        self.viz = None
        self.demonstrations = None
        self.planning_thread = None

    def show(self):
        """Define the GUI functionalities."""
        size_parameter = 0
        if os.name == 'nt':  # Windows
            size_parameter = 500
            button_length = None
        elif os.name == 'posix':  # Linux Ubuntu
            size_parameter = 60
            button_length = 36
        layout = [
            [
                gui.Input(
                    'No folder selected',
                    key='__folder_display__',
                    enable_events=True,
                    disabled=True,
                    size=(size_parameter, None)
                ),
                gui.FolderBrowse(
                    'Select demo folder',
                    target='__folder_display__',
                    key='__demo_folder__'
                )
            ],
            [gui.Text(
                'Number of demonstrations:', key='__n_demos__', size=(size_parameter, None))],
            [gui.Text('Number of nodes:', key='__n_nodes__', size=(size_parameter, None))],
            [gui.ProgressBar(20, size=(52, 3), key='__progress__', visible=False)],
            [
                gui.Button(
                    'Add demonstration',
                    tooltip='Teach the robot a new task buy guiding its arm.',
                    size=(button_length, None),
                    key='__new_demo__'),
                gui.Button(
                    'Run',
                    tooltip='Run the loaded Behavior Tree and execute the learnt task.',
                    disabled=True,
                    size=(button_length, None),
                    key='__run__')
            ],
            [
                gui.Button(
                    'Show tree',
                    tooltip='Display the Behavior Tree.',
                    disabled=True,
                    size=(button_length, None),
                    key='__show__'),
                gui.Button(
                    'Save tree',
                    tooltip='Save the Behavior Tree for future reference.',
                    disabled=True,
                    size=(button_length, None),
                    key='__save__')
            ]
        ]
        title = 'Behavior Tree LfD'
        window = gui.Window(title, layout)

        if not self.has_robot:
            # Hide unsupported buttons when not connected to the robot
            window.read(timeout=0)
            window.find_element('__new_demo__').update(visible=False)
            window.find_element('__run__').update(visible=False)

        ret = 0
        while ret is not None:
            ret, values = window.read()

            if ret == '__folder_display__' and os.path.isdir(values['__folder_display__']):
                self.demo_dir = values['__folder_display__']
                self.__reload_demos(window)
            elif ret == '__new_demo__':
                window.disappear()
                self.__new_demo()
                self.__reload_demos(window)
                window.reappear()
            elif ret == '__run__':
                if self.tree_timer is None:
                    window.find_element('__new_demo__').update(disabled=True)
                    window.find_element('__demo_folder__').update(disabled=True)

                    self.viz = BTVisualizer(self.bt)

                    self.world_interface.set_leadthrough('off')
                    task = self.world_interface.home()
                    task.start()
                    task.join()

                    window.find_element('__run__').update('Stop')
                    print('Launching tree...')
                    self.tree_timer = self.create_timer(1.0/self.tick_freq, self.viz.tick)
                else:
                    self.destroy_timer(self.tree_timer)
                    self.tree_timer = None
                    self.bt.root.stop()
                    self.world_interface.restart_rapid()
                    window.find_element('__new_demo__').update(disabled=False)
                    window.find_element('__demo_folder__').update(disabled=False)
                    window.find_element('__run__').update('Run')
            elif ret == '__show__':
                if os.name == 'nt':  # Windows
                    os.startfile(os.path.join(self.tree_dir.name, 'full.svg'))
                else:
                    opener = 'xdg-open'
                    subprocess.call([opener, os.path.join(self.tree_dir.name, 'full.svg')])
            elif ret == '__save__':
                folder = gui.popup_get_folder('Select a folder')
                if folder is not None and folder != '':
                    if os.path.isdir(folder):
                        number = 1
                        while os.path.isdir(os.path.join(folder, f'bt{number}')):
                            number += 1

                        folder = os.path.join(folder, f'bt{number}')

                os.makedirs(folder)
                copy_tree(self.tree_dir.name, folder)

    def __new_demo(self):
        """Launch a new window to start a new demonstration."""
        task = self.world_interface.set_gripper('open')
        task.start()
        task.join()

        if os.name == 'nt':  # Windows
            button_length = None
        elif os.name == 'posix':  # Linux Ubuntu
            button_length = 15

        layout = [
            [gui.Button(
                'Pick',
                tooltip='Close the grippers around the target object.',
                size=(button_length, None),
                key='__pick__'
            )],
            [gui.Button(
                'Fine place',
                tooltip='Open the grippers. Use when the target position has to be precise.',
                size=(button_length, None),
                key='__place__'
            )],
            [gui.Button(
                'Drop',
                tooltip='Open the grippers. Use when the target position does not matter much.',
                size=(button_length, None),
                key='__drop__'
            )]
        ]
        # TODO: add MoveBase buttons
        title = 'New demo'
        window = gui.Window(title, layout)

        if self.demonstrations is None:
            n_demos = 0
        else:
            n_demos = self.demonstrations.n_demonstrations()

        os.mkdir(os.path.join(self.demo_dir, 'demo%d' % (n_demos + 1)))

        ret = 0
        while ret is not None:
            self.world_interface.set_leadthrough('on')
            ret, values = window.read()

            if ret == '__pick__':
                task = self.world_interface.set_gripper('closed')
                task.start()
                task.join()
                self.__write_action('pick', n_demos + 1)
            if ret == '__place__':
                task = self.world_interface.set_gripper('open')
                task.start()
                task.join()
                self.__write_action('place', n_demos + 1)
            if ret == '__drop__':
                task = self.world_interface.set_gripper('open')
                task.start()
                task.join()
                self.__write_action('drop', n_demos + 1)
            # TODO: add MoveBase buttons

    def __write_action(
        self,
        action_type: str,
        current_demo: str
    ):
        """
        Write the pose of END_EFFECTOR_FRAME in all available frames as a new action.

        Args:
        ----
            action_type: name of the type of the action.
            current_demo: path to the directory where demo data is stored.

        """
        action_data = {'type': action_type, 'vec_pos': {}, 'vec_quat': {}}
        current_demo_dir = os.path.join(self.demo_dir, 'demo%d' % current_demo)
        # TODO: check if this is actually working
        if self.demonstrations is None:
            frames = self.all_frames
        else:
            frames = self.demonstrations.frames

        now = rclpy.time.Time()
        for frame in frames:
            tf_msg = self.tf_buffer.lookup_transform(frame, self.ee_frame, now)
            position_msg = tf_msg.transform.translation
            position = np.array([position_msg.x, position_msg.y, position_msg.z])
            # Round position to 3 decimal places
            position = np.round(position, 3)
            orientation_msg = tf_msg.transform.rotation
            orientation = np.array([
                orientation_msg.x,
                orientation_msg.y,
                orientation_msg.z,
                orientation_msg.w
            ])
            action_data['vec_pos'][frame] = position.tolist()
            action_data['vec_quat'][frame] = orientation.tolist()

        action_number = len(glob.glob(os.path.join(current_demo_dir, 'data_*.yaml'))) + 1
        with open(os.path.join(current_demo_dir, f'data_{action_number}.yaml'), 'w') as f:
            yaml.dump(action_data, f, default_flow_style=None)

    def __reload_demos(self, window: gui.Window):
        n_demos = self.__load_demo()
        window.find_element('__n_demos__').update('Number of demonstrations: %d' % n_demos)
        window.find_element('__new_demo__').update(disabled=False)

        if self.demonstrations is None:
            return

        self.__build_tree(window)

    def __load_demo(self) -> int:
        """Load demonstration and return number of demonstrations."""
        info_file = os.path.join(self.demo_dir, 'info.yaml')
        if not os.path.isfile(info_file):
            with open(info_file, 'w') as f:
                yaml.dump(
                    {
                        'frames': self.all_frames,
                        'default_frame': self.default_frame
                    }, f)
            self.demonstrations = None
            return 0
        elif len(glob.glob(os.path.join(self.demo_dir, 'demo*'))) == 0:
            self.demonstrations = None
            return 0
        else:
            self.demonstrations = RobotDemonstration(
                self.demo_dir,
                {
                    'pick': yumi_actions.PickAction,
                    'place': yumi_actions.PlaceAction,
                    'drop': yumi_actions.PlaceAction,
                    'move': base_actions.MoveAction
                }
            )
            return self.demonstrations.n_demonstrations()

    def __build_tree(self, window: gui.Window):
        """Build the Behavior Tree once it is loaded."""
        window.find_element('__run__').update('Building tree...', disabled=True)
        window.find_element('__show__').update(disabled=True)
        window.find_element('__save__').update(disabled=True)
        pbar = window.find_element('__progress__')
        pbar.update(visible=True)
        window.read(timeout=0)

        self.tree_dir = TemporaryDirectory()
        settings_dir = os.path.join(self.tree_dir.name, 'settings')
        equivalent = find_equivalent_actions(
            self.demonstrations,
            {
                'pick': yumi_actions.EquivalentPick,
                'place': yumi_actions.EquivalentPlace,
                'drop': yumi_actions.EquivalentPlace,
                'move': base_actions.EquivalentMove
            }
        )
        behaviors = MobileYuMiBehaviors(settings_dir)
        # Send the offline interface to the planner to expand the BT
        offline_interface = OfflineInterface(
            self.demonstrations.frames,
            self.demonstrations.frames,
            self.demonstrations.default_frame,
            False
        )
        if not self.has_robot:
            # Use a simulated world interface if we are not connected to the real robot
            self.world_interface = offline_interface

        # tree is a PyTree object!
        tree = learn_tree(
            settings_dir,
            equivalent,
            behaviors,
            offline_interface,
            iterations=50,
            callback=pbar.update_bar
        )
        Blackboard.clear()
        string_tree = tree.bt.bt

        if self.use_hri:
            # now a tree is learnt from demo, so from now on we can modify it to add stuff related
            # to the execution, like HRI behaviors.
            behaviors = LfDBehaviors(settings_dir)
            post_prox.update_bt_settings(settings_dir, equivalent, behaviors)

            # add the HRI subtree on top of the learned tree
            hri_tree = ['s(', 'f(', 'clear', 'disambiguate', ')']
            hri_tree = hri_tree + string_tree
            # since we added 1 depth level, close the tree
            hri_tree.append(')')

            string_tree = hri_tree

        # in string_tree, the string representation of the tree is stored
        with open(os.path.join(self.tree_dir.name, 'tree.yaml'), 'w') as f:
            yaml.dump(string_tree, f)

        self.bt = PyTree(
            string_tree, behaviors, world_interface=self.world_interface)

        # Print the BT with the Blackboard variables
        render_dot_tree(
            self.bt.root,
            name='debug_pytree',
            target_directory=self.tree_dir.name,
            with_blackboard_variables=True
        )

        tree_render.dot_graph(self.bt).write_svg(
            os.path.join(self.tree_dir.name, 'full.svg'), encoding='utf-8')
        positions = tree_render.write_tikz_tree(
            self.bt, os.path.join(self.tree_dir.name, 'full.tex'))
        tree_render.py_trees_dot(self.bt).write_svg(
            os.path.join(self.tree_dir.name, 'full_pytrees.svg'), encoding='utf-8'
        )

        with open(os.path.join(self.tree_dir.name, 'positions.yaml'), 'w') as f:
            yaml.dump(positions, f)

        window.find_element('__run__').update('Run', disabled=False)
        window.find_element('__show__').update(disabled=False)
        window.find_element('__save__').update(disabled=False)
        window.find_element('__n_nodes__').update('Number of nodes: %d' % self.bt.bt.length())
        pbar.update(visible=False)


def main():
    rclpy.init()
    gui = LfDGUI()
    t = threading.Thread(target=lambda: rclpy.spin(gui))
    t.start()
    gui.show()
    gui.destroy_node()
    rclpy.shutdown()
    t.join()


if __name__ == '__main__':
    main()
