"""Node with a data structure that keeps track of the robot target during execution."""

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

from execution_interfaces.srv import GetTarget, SetTarget
import rclpy
import rclpy.node


class Tracker(rclpy.node.Node):

    def __init__(self):
        super().__init__('target_tracker')

        self.current_target = {
            'action_type': None,
            'target_name': None,
            'frame_id': None,
            'target_pose': None,
            'bounding_box': None
        }

        self.get_target_srv = self.create_service(GetTarget, 'get_target_srv', self.get_target_cb)
        self.set_target_srv = self.create_service(SetTarget, 'set_target_srv', self.set_target_cb)

    def get_target_cb(self, request, response):
        """Get the current robot target."""
        if self.current_target['action_type'] is None:
            response.success = False
            response.message = 'Target not set!'
            response.action = None
        else:
            response.success = True
            response.message = ''
            response.action.action_type = self.current_target['action_type']
            response.action.target_name = self.current_target['target_name']
            response.action.frame_id = self.current_target['frame_id']
            response.action.target_pose = self.current_target['target_pose']
            response.action.bounding_box = self.current_target['bounding_box']

        return response

    def set_target_cb(self, request, response):
        """Set the current robot target."""
        response.success = True
        response.message = 'Target set!'

        self.current_target['action_type'] = request.action.action_type
        self.current_target['target_name'] = request.action.target_name
        self.current_target['frame_id'] = request.action.frame_id
        self.current_target['target_pose'] = request.action.target_pose
        self.current_target['bounding_box'] = request.action.bounding_box

        return response


def main():
    rclpy.init()
    tracker_service = Tracker()
    rclpy.spin(tracker_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
