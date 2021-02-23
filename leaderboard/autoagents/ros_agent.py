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
from ros_compatibility import CompatibleNode, latch_on, ros_ok, QoSProfile, ros_init, ros_timestamp, ROSException, ROS_VERSION, ros_shutdown
from nav_msgs.msg import Path
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from geometry_msgs.msg import PoseStamped
from carla_msgs.msg import CarlaEgoVehicleControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from carla_ros_scenario_runner.application_runner import ApplicationRunner, ApplicationStatus

from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import time
import rclpy

def get_entry_point():
    return 'RosAgent'

class RosAgent(AutonomousAgent):

    """
    Base class for ROS-based stacks.
    """
    def __init__(self, path_to_conf_file):
        super(RosAgent, self).__init__(path_to_conf_file)

        if not ros_ok():
            print("Initializing ROS")
            ros_init()

        self.ros_node = CompatibleNode("LeaderboardAgent")
        qos_profile=QoSProfile(depth=1, durability=latch_on)
        self.waypoint_publisher = self.ros_node.new_publisher(Path, "/carla/hero/global_plan", qos_profile=qos_profile)
        self.ctrl_subscriber = self.ros_node.create_subscriber(CarlaEgoVehicleControl, "/carla/hero/vehicle_control_cmd", self.new_control_callback, qos_profile=QoSProfile(depth=1, durability=False))

        self.leaderboard_executor = MultiThreadedExecutor()
        self.leaderboard_executor.add_node(self.ros_node)

        self.ros_spin_thread = Thread(target=self.spin_executor)

        self.new_control_cmd = None
        self.old_control_cmd = carla.VehicleControl()

    def get_launch_cmdline(self):
        """
        This method should be overriden in the custom agent
        Return command to launch the stack as a string
        """
        raise NotImplementedError

    def spin_executor(self):
        self.spinning = True
        while True and self.spinning:
            try:
                self.leaderboard_executor.spin_once(timeout_sec=0.01)
            except Exception as e:
                self.ros_node.logerr(f'Problem in ros_agent spin thread: {e}')

    def setup(self, path_to_conf_file):
        
        self.ros_spin_thread.start()
        self.agent_runner = ApplicationRunner(self.app_runner_status_updated,
                                        lambda log: print(f'ROS STACK: {log}'),
                                        "Passive mode is enabled")
        cmd_line = self.get_launch_cmdline()
        execute_stack = self.agent_runner.execute(cmd_line, env=os.environ)

    def app_runner_status_updated(self, status):
        """
        Executed from application runner whenever the status changed
        """
        print("ROS STACK: Status updated to {}".format(status))

    def new_control_callback(self, new_control):
        self.new_control_cmd = new_control

    def destroy(self):
        """
        Cleanup of all ROS publishers
        """
        self.ros_node.loginfo("Cleaning up")
        self.agent_runner.shutdown()
        self.ros_node.loginfo('joining spin thread')
        self.spinning = False
        self.ros_spin_thread.join()
        self.ros_node.loginfo('spin thread joined')

        self.leaderboard_executor.shutdown()
        del self.waypoint_publisher
        del self.ros_node
        del self.ros_spin_thread
        ros_shutdown()

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
        try:
            limit_count = 0 
            while self.new_control_cmd is None and limit_count < 50 :
                limit_count +=1
                time.sleep(0.01)
            if self.new_control_cmd is None:
                print('after 1 sec no new cmd, applying previous one')
                self.new_control_cmd = self.old_control_cmd
            else:
                self.old_control_cmd = self.new_control_cmd
            cmd.throttle = self.new_control_cmd.throttle
            cmd.steer = self.new_control_cmd.steer
            cmd.brake = self.new_control_cmd.brake
            cmd.hand_brake = self.new_control_cmd.hand_brake
            cmd.reverse = self.new_control_cmd.reverse
            cmd.gear = self.new_control_cmd.gear
            cmd.manual_gear_shift = self.new_control_cmd.manual_gear_shift

            self.new_control_cmd = None
        except ROSException:
            print("No ctrl received.")
            pass
        return cmd