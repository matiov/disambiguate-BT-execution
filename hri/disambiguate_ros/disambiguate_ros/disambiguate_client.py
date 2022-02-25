"""Client to test the services in disambiguate_ros."""

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

from hri_interfaces.srv import Disambiguate
import rclpy
from rclpy.node import Node


class DisambiguateClient(Node):
    """Call the disambiguation framework."""

    def __init__(self):
        super().__init__('disambiguate_client')

        self.declare_parameter('query', 'banana')

        self.target = self.get_parameter('query').get_parameter_value().string_value

        self.namespace = '/abb/sensors/camera'

        self.disambiguate = self.create_client(
            Disambiguate, self.namespace + '/disambiguate_srv')
        while not self.disambiguate.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service Disambiguate not available, waiting again...')

    def disambiguation_request(self):
        request = Disambiguate.Request()
        request.category_str = self.target
        self.future = self.disambiguate.call_async(request)


def main(args=None):
    rclpy.init(args=args)

    minimal_client = DisambiguateClient()
    minimal_client.disambiguation_request()

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.future.done():
            try:
                response = minimal_client.future.result()
            except Exception as e:
                minimal_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                minimal_client.get_logger().info(f'Result: {response.result}')
                minimal_client.get_logger().info(f'Bounding Box: {list(response.bounding_box)}')
                break

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
