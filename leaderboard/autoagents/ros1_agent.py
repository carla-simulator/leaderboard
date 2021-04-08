#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a ROS autonomous agent interface to control the ego vehicle via a ROS stack
"""

from __future__ import print_function

import time

from leaderboard.autoagents.ros_base_agent import BridgeHelper, ROSBaseAgent, ROSLauncher

import carla
import roslibpy
import transforms3d

client = roslibpy.Ros(host='localhost', port=9090)
client.connect()


def wait_for_message(client, topic, topic_type, timeout=None):

    class _WFM(object):
        def __init__(self):
            self.msg = None
        def cb(self, msg):
            if self.msg is None:
                self.msg = msg

    wfm = _WFM()
    s = None
    try:
        s = roslibpy.Topic(client, topic, topic_type, reconnect_on_close=False)
        s.subscribe(wfm.cb)
        if timeout is not None:
            timeout_t = time.time() + timeout
        while client.is_connected and wfm.msg is None:
            time.sleep(0.1)
            if timeout is not None and time.time() >= timeout_t:
                raise TimeoutError("timeout exceeded while waiting for message on topic {}".format(topic))

    finally:
        if s is not None:
            s.unsubscribe()

    return wfm.msg


def wait_for_service(client, service, timeout=None):

    if timeout is not None:
        timeout_t = time.time() + timeout

    services = client.get_services()
    while service not in services:
        time.sleep(0.1)
        if timeout is not None and time.time() >= timeout_t:
            raise TimeoutError("timeout exceeded while waiting for service {} to be ready".format(service))
        services = client.get_services()


class ROS1Agent(ROSBaseAgent):

    ROS_VERSION = 1

    def __init__(self, carla_host, carla_port, debug):
        super(ROS1Agent, self).__init__(self.ROS_VERSION, carla_host, carla_port, debug)

        self._server_process = ROSLauncher("server", self.ROS_VERSION)
        self._server_process.run(
            package="rosbridge_server",
            launch_file="rosbridge_websocket.launch",
            wait=True
        )

        client.run(30)

        #self._step_once_service = roslibpy.Service(client, "/carla/simulation_control", "carla_msgs/CarlaControl", reconnect_on_close=False)

        self._spawn_object_service = roslibpy.Service(client, "/carla/spawn_object", "carla_msgs/SpawnObject", reconnect_on_close=False)
        self._destroy_object_service = roslibpy.Service(client, "/carla/destroy_object", "carla_msgs/DestroyObject", reconnect_on_close=False)

        wait_for_service(client, "/carla/spawn_object")
        wait_for_service(client, "/carla/destroy_object")

        self._control_subscriber = roslibpy.Topic(client, "/carla/hero/vehicle_control_cmd", "carla_msgs/CarlaEgoVehicleControl", queue_length=1, reconnect_on_close=False)
        self._control_subscriber.subscribe(self._vehicle_control_cmd_callback)

        self._path_publisher = roslibpy.Topic(client, "/carla/hero/global_plan", "nav_msgs/Path", latch=True, reconnect_on_close=False)

        status = wait_for_message(client, "/carla/hero/status", "std_msgs/Bool")

    def spawn_object(self, type_, id_, transform, attributes, attach_to=0):
        spawn_point = BridgeHelper.carla2ros_pose(
            transform.location.x, transform.location.y, transform.location.z,
            transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw,
            to_quat=True
        )
        attributes = [{"key": str(key), "value": str(value)} for key, value in attributes.items()]

        request = roslibpy.ServiceRequest({
            "type": type_,
            "id": id_,
            "attach_to": attach_to,
            "transform": spawn_point,
            "random_pose": False,
            "attributes": attributes,
        })

        response = self._spawn_object_service.call(request)
        # TODO: raise error when a error ocurred (i.e. response["id"] == -1)?
        return response["id"]

    def destroy_object(self, uid):
        request = roslibpy.ServiceRequest({"id": uid})
        response = self._destroy_object_service.call(request)
        # TODO: raise error when a error ocurred (i.e. response["success"] == False)?
        return response["success"]

    def run_step(self, input_data, timestamp):
        assert self._server_process.is_alive()
        return super(ROS1Agent, self).run_step(input_data, timestamp)

    # TODO: Create custom message for the global plan (not only coordinates but RoadOption)
    # TODO: Two publishers. One for world coordinates and another one for gps coordinates
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super(ROS1Agent, self).set_global_plan(global_plan_gps, global_plan_world_coord)

        poses = []
        for wp in self._global_plan_world_coord:
            poses.append(BridgeHelper.carla2ros_pose(
                wp[0].location.x, wp[0].location.y, wp[0].location.z,
                wp[0].rotation.roll, wp[0].rotation.pitch, wp[0].rotation.yaw,
                to_quat=True
            ))

        self._path_publisher.publish(roslibpy.Message(
            {
                "header": {
                    "frame_id": "/map"
                },
                "poses": [{ "pose": pose } for pose in poses]
            }))

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        self._control_subscriber.unsubscribe()
        self._path_publisher.unadvertise()
        self._spawn_object_service.unadvertise()
        self._destroy_object_service.unadvertise()

        self._server_process.terminate()

        assert not self._server_process.is_alive()

        super(ROS1Agent, self).destroy()
