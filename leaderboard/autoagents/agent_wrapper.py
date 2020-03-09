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

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.envs.sensor_interface import CallBack, OpenDirveMapReader
from leaderboard.autoagents.autonomous_agent import Track

MAX_ALLOWED_RADIUS_SENSOR = 3.0

SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1
}


class SensorConfigurationInvalid(Exception):

    """Base class for other exceptions"""

    def __init__(self, message):
        print(message)
        super(SensorConfigurationInvalid, self).__init__()


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """

    _agent = None
    _sensors_list = []
    _challenge_mode = False

    def __init__(self, agent, challenge_mode):
        """
        Set the autonomous agent
        """
        self._agent = agent
        self._challenge_mode = challenge_mode

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        return self._agent()

    def setup_sensors(self, vehicle, debug_mode=False, track=None):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = OpenDirveMapReader(vehicle, sensor_spec['reading_frequency'])
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(200))
                    bp.set_attribute('rotation_frequency', str(10))
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(560000))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])


                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    bp.set_attribute('noise_alt_stddev', str(1.5))
                    bp.set_attribute('noise_lat_stddev', str(0.1))
                    bp.set_attribute('noise_lon_stddev', str(0.1))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.1))
                    bp.set_attribute('noise_accel_stddev_y', str(0.1))
                    bp.set_attribute('noise_accel_stddev_z', str(0.15))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.01))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.01))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.01))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)

        self._validate_sensor_configuration(self._agent.track)

        while not self._agent.all_sensors_ready():
            if debug_mode:
                print(" waiting for one data reading from sensors...")
            CarlaDataProvider.get_world().tick()

    def _validate_sensor_configuration(self, selected_track):
        """
        Ensure that the sensor configuration is valid, in case the challenge mode is used
        Returns true on valid configuration, false otherwise
        """

        if Track(selected_track) != self._agent.track:
            raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))

        sensor_count = {}

        for sensor in self._agent.sensors():
            if self._agent.track == Track.SENSORS:
                if sensor['type'].startswith('sensor.opendrive_map'):
                    raise SensorConfigurationInvalid("Illegal sensor used for Track [{}]!".format(self._agent.track))

            # let's check the extrinsics of the sensor
            if 'x' in sensor and 'y' in sensor and 'z' in sensor:
                if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                    raise SensorConfigurationInvalid(
                        "Illegal sensor extrinsics used for Track [{}]!".format(self._agent.track))

            if sensor['type'] in sensor_count:
               sensor_count[sensor['type']] += 1
            else:
               sensor_count[sensor['type']] = 0

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
