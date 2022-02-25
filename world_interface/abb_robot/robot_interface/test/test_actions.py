"""Test simple YuMi actions and general action classes."""

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


from bt_learning.learning_from_demo.tests.util import TESTDIR
from robot_behaviors.yumi_behaviors.lfd_actions import PickAction, PlaceAction
from robot_interface.demonstration import RobotAction, RobotDemonstration


def test_pick_demo():
    demos = RobotDemonstration(TESTDIR + '/demo_test', {'pick': PickAction, 'place': PlaceAction})

    action = demos.demonstrations()[0][0]

    # In this action we picked 'object'
    assert action.frame == ['object']
    assert action.parameters == ['object']


def test_place_demo():
    demos = RobotDemonstration(TESTDIR + '/demo_test', {'pick': PickAction, 'place': PlaceAction})

    action = demos.demonstrations()[0][1]
    # In this action we placed 'object'
    assert action.parameters == ['object']


class ActionPickTest(RobotAction):

    def __init__(self, data, frames, default_frame, *args, **kwargs):
        super().__init__(data, frames, default_frame, *args, **kwargs)


def test_action_demo():
    demos = RobotDemonstration(TESTDIR + '/demo_test', {'pick': ActionPickTest})

    # All pick actions should use our derived class
    for action in demos.all_actions():
        if action.type == 'pick':
            assert isinstance(action, ActionPickTest)
