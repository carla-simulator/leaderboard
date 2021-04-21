# Copyright (c) # Copyright (c) 2018-2021 CVC.
#
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
from leaderboard.autoagents.map_helper import (enu_to_carla_loc,
                                               get_shortest_route,
                                               get_route_lane_list,
                                               to_ad_paraPoint,
                                               get_lane_interval_list)

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

        # AD map library
        self._map_initialized = False
        self._txt_name = "LeaderboardMap.txt"
        self._xodr_name = "LeaderboardMap.xodr"

        # Controller constants
        self._controller = None
        self._lateral_pid = {'K_P': 1.95, 'K_D': 0.2, 'K_I': 0.05, 'dt': 0.05}
        self._longitudinal_pid = {'K_P': 1.0, 'K_D': 0, 'K_I': 0.05, 'dt': 0.05}
        self._max_brake = 0.75

        # Current location constants
        self._weight = 0.95  # Weight of the expected location. Between 0 and 1

        # Target location constants
        self._target_min_dist = 1  # How far will the target waypoint be [in meters]
        self._target_index_rate = 0.75  # Target distance increase rate w.r.t ego's velocity

        # Target speed constants
        self._target_speed = 30 / 3.6 

        # Obstacle detection constants
        self._obstacle_min_dist = 5  # Min obstacle distance to be checked [in meters]
        self._obstacle_index_rate = 1  # Target distance increase rate w.r.t ego's velocity
        self._obstacle_threshold = 0.5  # Threshold used to discern between obstacles

        self._prev_location = None
        self._prev_heading = None

        self._world = carla.Client('127.0.0.1', 2000).get_world()

    def sensors(self):
        """Define the sensors required by the agent. IMU and GNSS are setup at the
        same position to avoid having to change between those two coordinate references"""
        self._sensor_z = 1.8
        sensors = [
            {
                'type': 'sensor.opendrive_map',
                'reading_frequency': 1,
                'id': 'ODM'
            },
            {
                'type': 'sensor.speedometer',
                'id': 'Speed'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0, 'y': 0, 'z': self._sensor_z,
                'id': 'GNSS'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0, 'y': 0, 'z': 0,
                'roll': 0, 'pitch': 0, 'yaw': 0,
                'id': 'IMU'
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 0.7, 'y': 0.0, 'z': self._sensor_z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'id': 'LIDAR'
            }
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
            self._controller = VehiclePIDController(self._lateral_pid, self._longitudinal_pid, max_brake=self._max_brake)
        else:
            current_location = self._get_current_location(data)
            current_heading = self._get_current_heading(data)
            current_transform = self._get_current_transform(current_location, current_heading)
            current_speed = self._get_current_speed(data)
            target_location = self._get_target_location(current_location, current_speed)
            target_speed = self._get_target_speed(current_location)

            # Traffic Light tests, missing some attributes at the AD map library
            # ego_lane_id = to_ad_paraPoint(current_location).laneId
            # print(ego_lane_id)
            # tls = ad.map.landmark.getVisibleTrafficLights(ego_lane_id)
            # for tl in tls:
            #     print(tl)

            control = self._controller.run_step(
                target_speed, current_speed,
                target_location, current_location, current_heading
            )
            if self._is_obstacle_detected(data, current_transform, current_speed):
                control = self._controller.emergency_brake(control)

            self._prev_heading = current_heading
            self._prev_location = current_location

        return control

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

    def _get_current_heading(self, data):
        """Transform the compass data (radiants) into the vehicle heading"""
        compass_data = data['IMU'][1][6]
        compass_rad = (compass_data - math.pi / 2) % (2 * math.pi)  # Substract 90º and clip it
        return  carla.Vector3D(x=math.cos(compass_rad), y=math.sin(compass_rad))

    def _get_current_transform(self, location, heading):
        """Returns the current ego vehicle transform"""
        yaw = math.degrees(math.atan2(heading.y, heading.x))
        return carla.Transform(location, carla.Rotation(yaw=yaw))

    def _get_current_speed(self, data):
        """Calculates the speed of the vehicle"""
        return data['Speed'][1]['speed']

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

        # Get the target waypoint for both route and obstacle detection
        route_added_target = int(self._target_min_dist + self._target_index_rate * current_speed)
        target_route_index = min(self._route_index + route_added_target, len(self._route) - 1)

        return self._route[target_route_index]

    def _get_target_speed(self, current_location):
        """Returns the desired target speed"""
        return self._target_speed

    def _is_obstacle_detected(self, data, base_transform, current_speed):
        """Detects whether there is an obstacle in front of the ego"""
        target_locations = []
        obstacle_range = [[0, 0], [0, 0]]  # Simplify the route lcoation to a square [[min_x, max_x], [min_y, max_y]]

        obstacle_added_target = int(self._obstacle_min_dist + self._obstacle_index_rate * current_speed)
        obstacle_target_index = min(self._route_index + obstacle_added_target, len(self._route) - 1)

        # Displace the route up / down depending on the terrain 
        route_z_increase = (base_transform.location.z - self._sensor_z) - self._route[self._route_index].z
        route_z_increase += 2 * self._obstacle_threshold  # And a bit more to avoid false positives with the ground

        # Get a list of the route waypoints (in sensor coordinates)
        for i in range(self._route_index + 1, obstacle_target_index):

            # Copy the route location as some of its attributes will be changed (upwards shift)
            temp = self._route[i]
            route_location = carla.Location(temp.x , temp.y, temp.z)
            route_location.z += route_z_increase

            # Change it to sensor coordinates
            route_location_ = np.array([[route_location.x, route_location.y, route_location.z, 1]])
            self._world.debug.draw_point(route_location, size=0.2, life_time=0.25, color=carla.Color(0,255,255))
            target_location_ = np.matmul(base_transform.get_inverse_matrix(), np.transpose(route_location_))
            target_location = carla.Location(target_location_[0][0], target_location_[1][0], target_location_[2][0])
            target_locations.append(target_location)

            # Update the square dimensions
            if target_location.x < obstacle_range[0][0]:
                obstacle_range[0][0] = target_location.x
            elif target_location.x > obstacle_range[0][1]:
                obstacle_range[0][1] = target_location.x
            if target_location.y < obstacle_range[1][0]:
                obstacle_range[1][0] = target_location.y
            elif target_location.y > obstacle_range[1][1]:
                obstacle_range[1][1] = target_location.y

        # Check all LIDAR points for possible obstacles
        for lidar_point in data['LIDAR'][1]:
            lidar_location = carla.Location(float(lidar_point[0]), float(lidar_point[1]), float(lidar_point[2]))

            # Points outside the interest zone are ignored
            if lidar_location.x < obstacle_range[0][0] - self._obstacle_threshold:
                continue
            if lidar_location.x > obstacle_range[0][1] + self._obstacle_threshold:
                continue
            if lidar_location.y < obstacle_range[1][0] - self._obstacle_threshold:
                continue
            if lidar_location.y > obstacle_range[1][1] + self._obstacle_threshold:
                continue

            if self._is_location_obstacle(target_locations, lidar_location):
                # loc__ = np.array([[lidar_location.x, lidar_location.y, lidar_location.z, 1]])
                # loc_ = np.matmul(base_transform.get_matrix(), np.transpose(loc__))
                # loc = carla.Location(loc_[0][0], loc_[1][0], loc_[2][0])
                # self._world.debug.draw_point(loc, size=0.2, life_time=0.25, color=carla.Color(0,255,255))
                return True

        return False

    def _is_location_obstacle(self, targets, location):
        """Calculates whether or not a location can be considered an obstacle"""
        location_ = np.array([location.x, location.y, location.z])
        for target in targets:
            target_ = np.array([target.x, target.y, target.z])
            if np.linalg.norm(target_ - location_) < self._obstacle_threshold:
                return True
        return False

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

            # Ignore the lane change parts
            start_option = self._global_plan_world_coord[i-1][1]
            end_option = self._global_plan_world_coord[i][1]
            if start_option.value in (5, 6) and start_option == end_option:
                if to_ad_paraPoint(start_location).laneId != to_ad_paraPoint(end_location).laneId:
                    continue  # Two points signifying a lane change, move to the next segment

            # Get the route
            route_segment, start_lane_id = get_shortest_route(start_location, end_location)
            if not route_segment:
                continue  # No route found, move to the next segment

            # Transform the AD map route representation into waypoints (When needed to parse the altitude).
            # The parameters must be precomputed to know its final length, and interpolate the height
            params = []
            route_lanes = get_route_lane_list(route_segment, start_lane_id)
            for i, segment in enumerate(route_lanes):
                for param in get_lane_interval_list(segment.laneInterval):
                    params.append([param, i])

            altitudes = self._get_lane_altitude_list(start_location.z, end_location.z, len(params))
            for i, param in enumerate(params):
                lane_id = route_lanes[param[1]].laneInterval.laneId
                para_point = ad.map.point.createParaPoint(lane_id, ad.physics.ParametricValue(param[0]))
                enu_point = ad.map.lane.getENULanePoint(para_point)
                carla_point = enu_to_carla_loc(enu_point)
                carla_point.z = altitudes[i]  # AD Map doesn't parse the altitude
                self._route.append(carla_point)
                self._world.debug.draw_point(carla_point, size=0.1, life_time=100, color=carla.Color(0,0,0))

            # # Transform the AD map route representation into waypoints (If not needed to parse the altitude)
            # for segment in get_route_lane_list(route_segment, start_lane_id):
            #     lane_id = segment.laneInterval.laneId
            #     param_list = get_lane_interval_list(segment.laneInterval)
            #     for i in range(len(param_list)):
            #         para_point = ad.map.point.createParaPoint(lane_id, ad.physics.ParametricValue(param_list[i]))
            #         enu_point = ad.map.lane.getENULanePoint(para_point)
            #         carla_point = enu_to_carla_loc(enu_point)
            #         self._route.append(carla_point)
            #         # self._world.debug.draw_point(carla_point, size=0.1, life_time=100, color=carla.Color(0,0,0))

    def _get_lane_altitude_list(self, start_z, end_z, length):
        """Gets the z values of a lane. This is a simple linear interpolation
        and it won't be necessary whenever the AD map parses the altitude"""
        if length == 0:
            return []
        if start_z == end_z:
            return start_z*np.ones(length)
        return np.arange(start_z, end_z, (end_z - start_z) / length)

    def destroy(self):
        """Remove the AD map library files"""
        for fname in [self._txt_name, self._xodr_name]:
            if os.path.exists(fname):
                os.remove(fname)

        super(MapAgent, self).destroy()
