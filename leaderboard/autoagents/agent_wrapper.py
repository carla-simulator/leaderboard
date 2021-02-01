#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function
import math
import os
import time

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.autonomous_agent import Track

from ros_compatibility import CompatibleNode, latch_on, ros_ok, QoSProfile, ros_init, ros_timestamp, ROSException, ros_shutdown, get_service_request, ROS_VERSION
from transforms3d.euler import euler2quat
from threading import Thread
import functools, asyncio
import concurrent
import time

from carla_msgs.srv import SpawnObject, DestroyObject
from geometry_msgs.msg import Pose
from diagnostic_msgs.msg import KeyValue

MAX_ALLOWED_RADIUS_SENSOR = 3.0

SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1
}


class AgentError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentError, self).__init__(message)


class CommunicationThread(Thread):

    def __init__(self, node):
        self.node = node
        super(CommunicationThread, self).__init__()

    def run(self):
        try:
            self.node.spin()
        except KeyboardInterrupt:
            #self.stop()
            raise

    def stop(self):
        ros_shutdown()

class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """

    allowed_sensors = [
        'sensor.opendrive_map',
        'sensor.speedometer',
        'sensor.camera.rgb',
        'sensor.camera',
        'sensor.lidar.ray_cast',
        'sensor.other.radar',
        'sensor.other.gnss',
        'sensor.other.imu'
    ]

    _agent = None
    _sensors_list = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        
        #Init ros once
            
        self.ros_node = CompatibleNode("AgentWrapper") 
        self.spawn_object_service = self.ros_node.create_service_client("/carla/spawn_object", SpawnObject)
        self.spawned_objects = []
        self._agent = agent        
        self.ros_spin_thread = Thread(target=self.ros_node.spin)
        self.ros_spin_thread.start()

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        return self._agent()

    def setup_sensors(self, vehicle, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        response_futures = []
        requests = []
        for sensor_spec in self._agent.sensors():
            spawn_object_type = None
            sensor_location = Pose()
            attributes = []
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                spawn_object_type = "sensor.pseudo.opendrive_map"
            elif sensor_spec['type'].startswith("sensor.speedometer"):
                spawn_object_type = "sensor.pseudo.speedometer"
            elif sensor_spec['type'].startswith('sensor.camera'):
                spawn_object_type = sensor_spec['type']
                attributes.append(KeyValue(key='image_size_x', value=str(sensor_spec['width'])))
                attributes.append(KeyValue(key='image_size_y', value=str(sensor_spec['height'])))
                attributes.append(KeyValue(key='fov', value=str(sensor_spec['fov'])))
                attributes.append(KeyValue(key='lens_circle_multiplier', value="3.0"))
                attributes.append(KeyValue(key='lens_circle_falloff', value="3.0"))
                attributes.append(KeyValue(key='chromatic_aberration_intensity', value="0.5"))
                attributes.append(KeyValue(key='chromatic_aberration_offset', value="0"))
                sensor_location = AgentWrapper.get_pose_from_sensor_spec(sensor_spec)
            elif sensor_spec['type'].startswith('sensor.lidar'):
                spawn_object_type = sensor_spec['type']
                attributes.append(KeyValue(key='range', value=str(85)))
                attributes.append(KeyValue(key='rotation_frequency', value=str(10)))
                attributes.append(KeyValue(key='channels', value=str(64)))
                attributes.append(KeyValue(key='upper_fov', value=str(10)))
                attributes.append(KeyValue(key='lower_fov', value=str(-30)))
                attributes.append(KeyValue(key='points_per_second', value=str(600000)))
                attributes.append(KeyValue(key='atmosphere_attenuation_rate', value=str(0.004)))
                attributes.append(KeyValue(key='dropoff_general_rate', value=str(0.45)))
                attributes.append(KeyValue(key='dropoff_intensity_limit', value=str(0.8)))
                attributes.append(KeyValue(key='dropoff_zero_intensity', value=str(0.4)))
                sensor_location = AgentWrapper.get_pose_from_sensor_spec(sensor_spec)
            elif sensor_spec['type'].startswith('sensor.other.radar'):
                spawn_object_type = sensor_spec['type']
                attributes.append(KeyValue(key='horizontal_fov', value=str(sensor_spec['fov'])))  # degrees
                attributes.append(KeyValue(key='vertical_fov', value=str(sensor_spec['fov']))) # degrees
                attributes.append(KeyValue(key='points_per_second', value='1500'))
                attributes.append(KeyValue(key='range', value='100'))  # meters
                sensor_location = AgentWrapper.get_pose_from_sensor_spec(sensor_spec)
            elif sensor_spec['type'].startswith('sensor.other.gnss'):
                spawn_object_type = sensor_spec['type']
                attributes.append(KeyValue(key='noise_alt_stddev', value=str(0.000005)))
                attributes.append(KeyValue(key='noise_lat_stddev', value=str(0.000005)))
                attributes.append(KeyValue(key='noise_lon_stddev', value=str(0.000005)))
                attributes.append(KeyValue(key='noise_alt_bias', value=str(0.0)))
                attributes.append(KeyValue(key='noise_lat_bias', value=str(0.0)))
                attributes.append(KeyValue(key='noise_lon_bias', value=str(0.0)))
                sensor_location = AgentWrapper.get_pose_from_sensor_spec(sensor_spec)
            elif sensor_spec['type'].startswith('sensor.other.imu'):
                spawn_object_type = sensor_spec['type']
                attributes.append(KeyValue(key='noise_accel_stddev_x', value=str(0.001)))
                attributes.append(KeyValue(key='noise_accel_stddev_y', value=str(0.001)))
                attributes.append(KeyValue(key='noise_accel_stddev_z', value=str(0.015)))
                attributes.append(KeyValue(key='noise_gyro_stddev_x', value=str(0.001)))
                attributes.append(KeyValue(key='noise_gyro_stddev_y', value=str(0.001)))
                attributes.append(KeyValue(key='noise_gyro_stddev_z', value=str(0.001)))
                sensor_location = AgentWrapper.get_pose_from_sensor_spec(sensor_spec)

            if spawn_object_type is not None:
                spawn_object_request = get_service_request(SpawnObject)
                spawn_object_request.type =spawn_object_type
                spawn_object_request.id = sensor_spec["id"]
                spawn_object_request.attach_to = vehicle.id
                spawn_object_request.random_pose = False
                spawn_object_request.attributes = attributes
                spawn_object_request.transform = sensor_location

                requests.append(spawn_object_request)
        
        result = True
        for req in requests:
            def done_callback(future):
                nonlocal result, self, response_futures # possible as, by default, service calls are processed sequentially
                if future.result().id == -1:
                    self.ros_node.logerr("Could not spawn object.")
                    result = False
                else:
                    self.ros_node.loginfo("Successfully spawned object.")
                    response_futures.remove(future)
                    self.spawned_objects.append(future.result().id)
                
            #calls.append(call_and_wait(node, spawn_service, req))
            self.ros_node.loginfo("Spawn object (type={}, id={})".format(req.type, req.id))
            future = self.ros_node.call_service_async(self.spawn_object_service, req)
            response_futures.append(future)
            future.add_done_callback(done_callback)

        # Tick once to spawn the sensors
        CarlaDataProvider.get_world().tick()
        
        while len(response_futures) != 0:
            print("Waiting for spawn_object responses....")
            time.sleep(0.5)
        if not result:
            raise RuntimeError("Error while spawning sensors.")
        

        # bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        # for sensor_spec in self._agent.sensors():
        #     # These are the pseudosensors (not spawned)
        #     if sensor_spec['type'].startswith('sensor.opendrive_map'):
        #         # The HDMap pseudo sensor is created directly here
        #         sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
        #     elif sensor_spec['type'].startswith('sensor.speedometer'):
        #         delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
        #         frame_rate = 1 / delta_time
        #         sensor = SpeedometerReader(vehicle, frame_rate)
        #     # These are the sensors spawned on the carla world
        #     else:
        #         bp = bp_library.find(str(sensor_spec['type']))
        #         if sensor_spec['type'].startswith('sensor.camera'):
        #             bp.set_attribute('image_size_x', str(sensor_spec['width']))
        #             bp.set_attribute('image_size_y', str(sensor_spec['height']))
        #             bp.set_attribute('fov', str(sensor_spec['fov']))
        #             bp.set_attribute('lens_circle_multiplier', str(3.0))
        #             bp.set_attribute('lens_circle_falloff', str(3.0))
        #             bp.set_attribute('chromatic_aberration_intensity', str(0.5))
        #             bp.set_attribute('chromatic_aberration_offset', str(0))

        #             sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
        #                                              z=sensor_spec['z'])
        #             sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
        #                                              roll=sensor_spec['roll'],
        #                                              yaw=sensor_spec['yaw'])
        #         elif sensor_spec['type'].startswith('sensor.lidar'):
        #             bp.set_attribute('range', str(85))
        #             bp.set_attribute('rotation_frequency', str(10))
        #             bp.set_attribute('channels', str(64))
        #             bp.set_attribute('upper_fov', str(10))
        #             bp.set_attribute('lower_fov', str(-30))
        #             bp.set_attribute('points_per_second', str(600000))
        #             bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
        #             bp.set_attribute('dropoff_general_rate', str(0.45))
        #             bp.set_attribute('dropoff_intensity_limit', str(0.8))
        #             bp.set_attribute('dropoff_zero_intensity', str(0.4))
        #             sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
        #                                              z=sensor_spec['z'])
        #             sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
        #                                              roll=sensor_spec['roll'],
        #                                              yaw=sensor_spec['yaw'])
        #         elif sensor_spec['type'].startswith('sensor.other.radar'):
        #             bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
        #             bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
        #             bp.set_attribute('points_per_second', '1500')
        #             bp.set_attribute('range', '100')  # meters

        #             sensor_location = carla.Location(x=sensor_spec['x'],
        #                                              y=sensor_spec['y'],
        #                                              z=sensor_spec['z'])
        #             sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
        #                                              roll=sensor_spec['roll'],
        #                                              yaw=sensor_spec['yaw'])

        #         elif sensor_spec['type'].startswith('sensor.other.gnss'):
        #             bp.set_attribute('noise_alt_stddev', str(0.000005))
        #             bp.set_attribute('noise_lat_stddev', str(0.000005))
        #             bp.set_attribute('noise_lon_stddev', str(0.000005))
        #             bp.set_attribute('noise_alt_bias', str(0.0))
        #             bp.set_attribute('noise_lat_bias', str(0.0))
        #             bp.set_attribute('noise_lon_bias', str(0.0))

        #             sensor_location = carla.Location(x=sensor_spec['x'],
        #                                              y=sensor_spec['y'],
        #                                              z=sensor_spec['z'])
        #             sensor_rotation = carla.Rotation()

        #         elif sensor_spec['type'].startswith('sensor.other.imu'):
        #             bp.set_attribute('noise_accel_stddev_x', str(0.001))
        #             bp.set_attribute('noise_accel_stddev_y', str(0.001))
        #             bp.set_attribute('noise_accel_stddev_z', str(0.015))
        #             bp.set_attribute('noise_gyro_stddev_x', str(0.001))
        #             bp.set_attribute('noise_gyro_stddev_y', str(0.001))
        #             bp.set_attribute('noise_gyro_stddev_z', str(0.001))

        #             sensor_location = carla.Location(x=sensor_spec['x'],
        #                                              y=sensor_spec['y'],
        #                                              z=sensor_spec['z'])
        #             sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
        #                                              roll=sensor_spec['roll'],
        #                                              yaw=sensor_spec['yaw'])
        #         # create sensor
        #         sensor_transform = carla.Transform(sensor_location, sensor_rotation)
        #         sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
        #     # setup callback
        #     sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self._agent.sensor_interface))
        #     self._sensors_list.append(sensor)

        # # Tick once to spawn the sensors
        # CarlaDataProvider.get_world().tick()

    @staticmethod
    def get_pose_from_sensor_spec(sensor_spec):
        pose = Pose()
        pose.position.x = sensor_spec['x']
        pose.position.y = sensor_spec['y']
        pose.position.z = sensor_spec['z']
        if 'pitch' in sensor_spec and 'roll' in sensor_spec and 'yaw' in sensor_spec:
            quat = euler2quat(sensor_spec['roll'], sensor_spec['pitch'], sensor_spec['yaw'])
            pose.orientation.w=quat[0]
            pose.orientation.x=quat[1]
            pose.orientation.y=quat[2]
            pose.orientation.z=quat[3]
        return pose

    # def setup_sensors(self, vehicle, debug_mode=False):
    #     """
    #     Create the sensors defined by the user and attach them to the ego-vehicle
    #     :param vehicle: ego vehicle
    #     :return:
    #     """
    #     bp_library = CarlaDataProvider.get_world().get_blueprint_library()
    #     for sensor_spec in self._agent.sensors():
    #         # These are the pseudosensors (not spawned)
    #         if sensor_spec['type'].startswith('sensor.opendrive_map'):
    #             # The HDMap pseudo sensor is created directly here
    #             sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
    #         elif sensor_spec['type'].startswith('sensor.speedometer'):
    #             delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
    #             frame_rate = 1 / delta_time
    #             sensor = SpeedometerReader(vehicle, frame_rate)
    #         # These are the sensors spawned on the carla world
    #         else:
    #             bp = bp_library.find(str(sensor_spec['type']))
    #             if sensor_spec['type'].startswith('sensor.camera'):
    #                 bp.set_attribute('image_size_x', str(sensor_spec['width']))
    #                 bp.set_attribute('image_size_y', str(sensor_spec['height']))
    #                 bp.set_attribute('fov', str(sensor_spec['fov']))
    #                 bp.set_attribute('lens_circle_multiplier', str(3.0))
    #                 bp.set_attribute('lens_circle_falloff', str(3.0))
    #                 bp.set_attribute('chromatic_aberration_intensity', str(0.5))
    #                 bp.set_attribute('chromatic_aberration_offset', str(0))

    #                 sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
    #                                                  z=sensor_spec['z'])
    #                 sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
    #                                                  roll=sensor_spec['roll'],
    #                                                  yaw=sensor_spec['yaw'])
    #             elif sensor_spec['type'].startswith('sensor.lidar'):
    #                 bp.set_attribute('range', str(85))
    #                 bp.set_attribute('rotation_frequency', str(10))
    #                 bp.set_attribute('channels', str(64))
    #                 bp.set_attribute('upper_fov', str(10))
    #                 bp.set_attribute('lower_fov', str(-30))
    #                 bp.set_attribute('points_per_second', str(600000))
    #                 bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
    #                 bp.set_attribute('dropoff_general_rate', str(0.45))
    #                 bp.set_attribute('dropoff_intensity_limit', str(0.8))
    #                 bp.set_attribute('dropoff_zero_intensity', str(0.4))
    #                 sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
    #                                                  z=sensor_spec['z'])
    #                 sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
    #                                                  roll=sensor_spec['roll'],
    #                                                  yaw=sensor_spec['yaw'])
    #             elif sensor_spec['type'].startswith('sensor.other.radar'):
    #                 bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
    #                 bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
    #                 bp.set_attribute('points_per_second', '1500')
    #                 bp.set_attribute('range', '100')  # meters

    #                 sensor_location = carla.Location(x=sensor_spec['x'],
    #                                                  y=sensor_spec['y'],
    #                                                  z=sensor_spec['z'])
    #                 sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
    #                                                  roll=sensor_spec['roll'],
    #                                                  yaw=sensor_spec['yaw'])

    #             elif sensor_spec['type'].startswith('sensor.other.gnss'):
    #                 bp.set_attribute('noise_alt_stddev', str(0.000005))
    #                 bp.set_attribute('noise_lat_stddev', str(0.000005))
    #                 bp.set_attribute('noise_lon_stddev', str(0.000005))
    #                 bp.set_attribute('noise_alt_bias', str(0.0))
    #                 bp.set_attribute('noise_lat_bias', str(0.0))
    #                 bp.set_attribute('noise_lon_bias', str(0.0))

    #                 sensor_location = carla.Location(x=sensor_spec['x'],
    #                                                  y=sensor_spec['y'],
    #                                                  z=sensor_spec['z'])
    #                 sensor_rotation = carla.Rotation()

    #             elif sensor_spec['type'].startswith('sensor.other.imu'):
    #                 bp.set_attribute('noise_accel_stddev_x', str(0.001))
    #                 bp.set_attribute('noise_accel_stddev_y', str(0.001))
    #                 bp.set_attribute('noise_accel_stddev_z', str(0.015))
    #                 bp.set_attribute('noise_gyro_stddev_x', str(0.001))
    #                 bp.set_attribute('noise_gyro_stddev_y', str(0.001))
    #                 bp.set_attribute('noise_gyro_stddev_z', str(0.001))

    #                 sensor_location = carla.Location(x=sensor_spec['x'],
    #                                                  y=sensor_spec['y'],
    #                                                  z=sensor_spec['z'])
    #                 sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
    #                                                  roll=sensor_spec['roll'],
    #                                                  yaw=sensor_spec['yaw'])
    #             # create sensor
    #             sensor_transform = carla.Transform(sensor_location, sensor_rotation)
    #             sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
    #         # setup callback
    #         sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self._agent.sensor_interface))
    #         self._sensors_list.append(sensor)

    #     # Tick once to spawn the sensors
    #     CarlaDataProvider.get_world().tick()
        

    @staticmethod
    def validate_sensor_configuration(sensors, agent_track, selected_track):
        """
        Ensure that the sensor configuration is valid, in case the challenge mode is used
        Returns true on valid configuration, false otherwise
        """
        if Track(selected_track) != agent_track:
            raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))

        sensor_count = {}
        sensor_ids = []

        for sensor in sensors:

            # Check if the is has been already used
            sensor_id = sensor['id']
            if sensor_id in sensor_ids:
                raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
            else:
                sensor_ids.append(sensor_id)

            # Check if the sensor is valid
            if agent_track == Track.SENSORS:
                if sensor['type'].startswith('sensor.opendrive_map'):
                    raise SensorConfigurationInvalid("Illegal sensor used for Track [{}]!".format(agent_track))

            # Check the sensors validity
            if sensor['type'] not in AgentWrapper.allowed_sensors:
                raise SensorConfigurationInvalid("Illegal sensor used. {} are not allowed!".format(sensor['type']))

            # Check the extrinsics of the sensor
            if 'x' in sensor and 'y' in sensor and 'z' in sensor:
                if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                    raise SensorConfigurationInvalid(
                        "Illegal sensor extrinsics used for Track [{}]!".format(agent_track))

            # Check the amount of sensors
            if sensor['type'] in sensor_count:
                sensor_count[sensor['type']] += 1
            else:
                sensor_count[sensor['type']] = 1


        for sensor_type, max_instances_allowed in SENSORS_LIMITS.items():
            if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
                raise SensorConfigurationInvalid(
                    "Too many {} used! "
                    "Maximum number allowed is {}, but {} were requested.".format(sensor_type,
                                                                                  max_instances_allowed,
                                                                                  sensor_count[sensor_type]))

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []
        
        ros_shutdown()
        self.ros_spin_thread.join()
