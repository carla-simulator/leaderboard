#!/usr/bin/env python
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
This module provides a ROS autonomous agent interface to control the ego vehicle via a ROS stack
"""

import os
import carla

import carla_common.transforms as trans
from ros_compatibility import CompatibleNode, latch_on, ros_ok, QoSProfile, ros_init, ros_timestamp, ROSException, ROS_VERSION
from nav_msgs.msg import Path
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from geometry_msgs.msg import PoseStamped
from carla_msgs.msg import CarlaEgoVehicleControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

ROS_INITIALIZED = False

def get_entry_point():
    return 'RosAgent'

class RosAgent(AutonomousAgent):

    """
    Base class for ROS-based stacks.
    """
    def __init__(self, path_to_conf_file):
        super(RosAgent, self).__init__(path_to_conf_file)
        global ROS_INITIALIZED
        if not ROS_INITIALIZED:
            print("Init ROS")
            ROS_INITIALIZED = True
            ros_init()

        
        self.ros_node = CompatibleNode("LeaderboardAgent") 
        qos_profile=QoSProfile(depth=1, durability=latch_on)
        self.waypoint_publisher = self.ros_node.new_publisher(Path, "/carla/hero/waypoints", qos_profile=qos_profile)

        self.spawn_object_service = None
        self.spawned_objects = []


    def setup(self, path_to_conf_file):
        
        print(CarlaDataProvider.get_world())

        # for sensor in self.sensors():
        #     print(sensor)
        #     spawn_object_type = None
        #     if sensor['type']=="sensor.speedometer":
        #         spawn_object_type = "sensor.pseudo.speedometer"

        #     if spawn_object_type is not None:
        #         if ROS_VERSION == 1:
        #             spawn_object_request = SpawnObjectRequest()
        #         elif ROS_VERSION == 2:
        #             spawn_object_request = SpawnObject.Request()
        #         spawn_object_request.type =spawn_object_type
        #         spawn_object_request.id = sensor["id"]
        #         spawn_object_request.attach_to = 0
        #         spawn_object_request.random_pose = False
                   
                
                
        #         response = self.ros_node.call_service(self.spawn_object_service, spawn_object_request)
        #         if response.id == -1:
        #             raise Exception("Could not spawn object in ros bridge.")
        #         else:
        #             print("Successfully spawned object ")
        #         self.spawned_objects.append(response.id)
                
                
        #1. start ros bridge
        #2. start agent


    def destroy(self):
        """
        Cleanup of all ROS publishers
        """
        self.ros_node.loginfo("Cleaning up")


        del self.waypoint_publisher
        del self.ros_node

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super(RosAgent, self).set_global_plan(global_plan_gps, global_plan_world_coord)
        
        msg = Path()
        msg.header.frame_id = "/map"
        msg.header.stamp = ros_timestamp(self.ros_node.get_time(), from_sec=True)
        for wp in self._global_plan_world_coord:
            msg.poses.append(PoseStamped(pose=trans.carla_transform_to_ros_pose(wp[0])))

        self.ros_node.loginfo("Publishing Plan...")
        self.waypoint_publisher.publish(msg)


    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        cmd = carla.VehicleControl()
        #print("waiting for ctrl...")
        try:
            data = self.ros_node.wait_for_one_message("/carla/hero/vehicle_control_cmd", CarlaEgoVehicleControl, timeout=1.0,
                                        qos_profile=QoSProfile(depth=1, durability=False))
            cmd.throttle = data.throttle
            cmd.steer = data.steer
            cmd.brake = data.brake
            cmd.hand_brake = data.hand_brake
            cmd.reverse = data.reverse
            cmd.gear = data.gear
            cmd.manual_gear_shift = data.manual_gear_shift
        except ROSException:
            print("No ctrl received.")
            pass
        return cmd