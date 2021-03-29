#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example of an agent that doesn't use the carla.Map in order to
navigate through the town, by using the (https://github.com/carla-simulator/map) library,
which doesn't access any privileged information.

This library isn't installed automatically by the leaderboard.

This agent ignores all dynamic elements of the scene as well as traffic lights so it is
recommended to remove the background activity when checking this example (to do so, go to
"leaderboard/scenarios/route_scenario.py" and change the dictionary at "_initialize_actors")
"""

from __future__ import print_function

import carla
import numpy as np
import math
import os
import xml.etree.ElementTree as ET

from leaderboard.autoagents.map_agent_controller import VehiclePIDController
from leaderboard.autoagents.map_helper import (get_route_segment,
                                              to_ad_paraPoint,
                                              get_lane_interval_list,
                                              enu_to_carla_loc,
                                              get_route_lane_list)
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

import ad_map_access as ad

def get_entry_point():
    return 'MapAgent'

class MapAgent(AutonomousAgent):
    """
    Autonomous agent to control the ego vehicle using the AD Map library
    to parse the opendrive map information
    """

    def setup(self, path_to_conf_file):
        """Setup the agent parameters"""
        self.track = Track.MAP

        # Route
        self._route = []  # List of [carla.Waypoint, RoadOption]
        self._route_index = 0  # Index of the closest route point to the vehicle
        self._route_buffer = 3  # Amount of route points checked
        self._min_target_dist = 2  # How far will the target waypoint be [in meters]

        # AD map library
        self._map_initialized = False
        self._txt_name = "LeaderboardMap.txt"
        self._xodr_name = "LeaderboardMap.xodr"

        # Controller
        self._controller = None
        self._target_speed = 20
        self._prev_location = None
        self._prev_heading = None
        self._weight = 0.95
        self._args_lateral_pid = {'K_P': 1.95, 'K_D': 0.2, 'K_I': 0.07, 'dt': 0.05}
        self._args_longitudinal_pid = {'K_P': 1.0, 'K_D': 0, 'K_I': 0.05, 'dt': 0.05}

        self.world = carla.Client('127.0.0.1', 2000).get_world()

    def sensors(self):
        """Define the sensors required by the agent"""
        sensors = [
            {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'ODM'},
            {'type': 'sensor.speedometer', 'id': 'Speed'},
            {'type': 'sensor.other.gnss', 'x': 0, 'y': 0, 'z': 0, 'id': 'GNSS'},
            {'type': 'sensor.other.imu', 'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'id': 'IMU'}
        ]

        return sensors

    def run_step(self, data, timestamp):
        """Execute one step of navigation."""
        control = carla.VehicleControl()

        # Initialize the map library
        if not self._map_initialized:
            if 'ODM' in data:
                self._map_initialized = self._initialize_map(data['ODM'][1]['opendrive'])
            else:
                return control

        # Create the route
        if not self._route:
            self._get_route()

        # Create the controller, or run one step of it
        if not self._controller:
            self._controller = VehiclePIDController(self._args_lateral_pid, self._args_longitudinal_pid)
        else:
            current_location = self._get_current_location(data)
            current_heading = self._get_current_heading(data)
            current_speed = self._get_current_speed(data)
            target_location = self._get_target_location(current_location, current_speed)
            target_speed = self._get_target_speed(current_location)

            control = self._controller.run_step(
                target_speed, current_speed,
                target_location, current_location, current_heading
            )

            self._prev_heading = current_heading
            self._prev_location = current_location

        return control

    def _get_current_speed(self, data):
        """Calculates the speed of the vehicle"""
        return 3.6 * data['Speed'][1]['speed']

    def _get_target_speed(self, current_location):
        """Returns the target speed"""
        # # get para point of the ego location
        # para_point = to_ad_paraPoint(current_location)

        # # get all speed limits of the lane
        # lane = ad.map.lane.getLane(para_point.laneId)
        # speed_limits = ad.map.lane.getSpeedLimits(lane, ad.physics.ParametricRange())

        # # get the one that is affecting the ego
        # offset = float(para_point.parametricOffset)
        # for sl in speed_limits:
        #     min_piece = float(sl.lanePiece.minimum)
        #     max_piece = float(sl.lanePiece.maximum)

        #     if min_piece < offset < max_piece:
        #         print(float(sl.speedLimit))
        #         return float(sl.speedLimit)

        return self._target_speed

    def _get_current_location(self, data):
        """Calculates the transform of the vehicle"""
        R = 6378135
        lat_rad = (np.deg2rad(data['GNSS'][1][0]) + np.pi) % (2 * np.pi) - np.pi
        lon_rad = (np.deg2rad(data['GNSS'][1][1]) + np.pi) % (2 * np.pi) - np.pi
        x = R * np.sin(lon_rad) * np.cos(lat_rad) 
        y = R * np.sin(-lat_rad)
        z = data['GNSS'][1][2]
        current_loc = carla.Location(x, y, z)

        # Remove some of the GNSS noise
        if self._prev_location and self._prev_heading:
            location_vec = current_loc - self._prev_location  # Displacement vector

            dot1 = location_vec.x * self._prev_heading.x + \
                   location_vec.y * self._prev_heading.y + \
                   location_vec.z * self._prev_heading.z

            dot2 = self._prev_heading.x * self._prev_heading.x + \
                   self._prev_heading.y * self._prev_heading.y + \
                   self._prev_heading.z * self._prev_heading.z

            # Compute expected location
            expected_loc = self._prev_location + self._prev_heading * dot1 / dot2

            # Get the mean of the two locations
            new_current_loc = carla.Location(
                self._weight * expected_loc.x + (1 - self._weight) * current_loc.x,
                self._weight * expected_loc.y + (1 - self._weight) * current_loc.y,
                self._weight * expected_loc.z + (1 - self._weight) * current_loc.z,
            )
            return new_current_loc

        else:
            return current_loc

        # TODO: use this one if changed
        # geo_point = ad.map.point.createGeoPoint(
        #     data['GNSS'][1][1],  # Long
        #     data['GNSS'][1][0],  # Lat
        #     data['GNSS'][1][2]   # Alt
        # )
        # enu_point = ad.map.point.toENU(geo_point)
        # return  enu_to_carla_loc(enu_point)

    def _get_current_heading(self, data):
        """Transform the compass data (radiants) into the vehicle heading"""
        compass_data = data['IMU'][1][6]
        compass_rad = (compass_data - math.pi / 2) % (2 * math.pi)  # Substract 90º and clip it
        return  carla.Vector3D(x=math.cos(compass_rad), y=math.sin(compass_rad))

    def _get_target_location(self, current_location, current_speed):
        """Returns the target location of the controller"""
        min_distance = float('inf')
        start_index = self._route_index
        end_index = min(start_index + self._route_buffer, len(self._route))

        for i in range(start_index, end_index):
            route_location = self._route[i]
            distance = current_location.distance(route_location)
            if distance < min_distance:
                self._route_index = i
                min_distance = distance

        added_target = max(int(self._min_target_dist + current_speed / (2*3.6)), 1)
        target_index = min(self._route_index + added_target, len(self._route) - 1)

        return self._route[target_index]

    def _initialize_map(self, opendrive_contents):
        """Initialize the AD map library and, creates the file needed to do so."""
        lat_ref = 0.0
        lon_ref = 0.0

        # Save the opendrive data into a file
        with open(self._xodr_name, 'w') as f:
            f.write(opendrive_contents)

        # Get geo reference
        xml_tree = ET.parse(self._xodr_name)
        for geo_elem in xml_tree.find('header').find('geoReference').text.split(' '):
            if geo_elem.startswith('+lat_0'):
                lat_ref = float(geo_elem.split('=')[-1])
            elif geo_elem.startswith('+lon_0'):
                lon_ref = float(geo_elem.split('=')[-1])

        # Save the previous info 
        with open(self._txt_name, 'w') as f:
            txt_content = "[ADMap]\n" \
                          "map=" + self._xodr_name + "\n" \
                          "[ENUReference]\n" \
                          "default=" + str(lat_ref) + " " + str(lon_ref) + " 0.0"
            f.write(txt_content)

        return ad.map.access.init(self._txt_name)

    def _get_route(self):
        """Creates a route with waypoints every meter."""

        for i in range (1, len(self._global_plan_world_coord)):

            # Get the starting and end location of the route segment
            start_location = self._global_plan_world_coord[i-1][0].location
            end_location = self._global_plan_world_coord[i][0].location

            # self.world.debug.draw_point(start_location + carla.Location(z=1.5), size=0.2, color=carla.Color(0,255,255))
            # self.world.debug.draw_point(end_location + carla.Location(z=1.5), size=0.2, color=carla.Color(255,255,0))
            # self.world.debug.draw_string(start_location + carla.Location(z=2), str(i), life_time=10000, color=carla.Color(0,0,0))

            # Ignore the lane change parts
            start_option = self._global_plan_world_coord[i-1][1]
            end_option = self._global_plan_world_coord[i][1]
            if start_option.value in (5, 6) and start_option == end_option:
                if to_ad_paraPoint(start_location).laneId != to_ad_paraPoint(end_location).laneId:
                    continue  # Ignore the lane change parts

            # Get the route
            route_segment, start_lane_id = get_route_segment(start_location, end_location)
            if not route_segment:
                continue  # No route found, move to the next segment

            # Transform the AD map route representation into waypoints
            for segment in get_route_lane_list(route_segment, start_lane_id):
                for param in get_lane_interval_list(segment.laneInterval):
                    para_point = ad.map.point.createParaPoint(
                        segment.laneInterval.laneId, ad.physics.ParametricValue(param)
                    )

                    enu_point = ad.map.lane.getENULanePoint(para_point)
                    self._route.append(enu_to_carla_loc(enu_point))

                    # carla_location = enu_to_carla_loc(enu_point)
                    # self.world.debug.draw_point(carla_location + carla.Location(z=1.5), life_time=-1)
                    # self._route.append(carla_location)

    def destroy(self):
        """Remove the AD map library files"""
        for fname in [self._txt_name, self._xodr_name]:
            if os.path.exists(fname):
                os.remove(fname)

        super(MapAgent, self).destroy()
