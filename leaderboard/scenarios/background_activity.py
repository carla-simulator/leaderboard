#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

import math
import pprint

import carla
import numpy as np
import numpy.random as random
import math
import py_trees
from collections import OrderedDict

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.timer import GameTime

from agents.navigation.local_planner import RoadOption


class BackgroundActivity(BasicScenario):

    """
    Implementation of a scenario to spawn a set of background actors,
    and to remove traffic jams in background traffic

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicle, config, route, debug_mode=False, timeout=0):
        """
        Setup all relevant parameters and create scenario
        """
        self._map = CarlaDataProvider.get_map()
        self.ego_vehicle = ego_vehicle
        self.route = route
        self.config = config
        self.debug = debug_mode
        self.timeout = timeout  # Timeout of scenario in seconds

        super(BackgroundActivity, self).__init__("BackgroundActivity",
                                                 [ego_vehicle],
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=True,
                                                 criteria_enable=True)

    def _initialize_actors(self, config):
        """
        Create the necessary initial actors
        """
        pass

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        # Check if a vehicle is further than X, destroy it if necessary and respawn it
        background_checker = py_trees.composites.Sequence()
        background_checker.add_child(BackgroundBehavior(
            self.ego_vehicle,
            self.route
        ))
        return background_checker

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        pass

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        pass


class BackgroundBehavior(AtomicBehavior):
    """
    Handles the background activity
    """
    def __init__(self, ego_actor, route, debug=True, name="BackgroundBehavior"):
        """
        Setup class members
        """
        super(BackgroundBehavior, self).__init__(name)
        self.debug = debug
        self._map = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self._rng = random.RandomState(2000)

        self._fake_junction_ids = []
        self._fake_lane_pair_keys = []
        self._micro_lane_pair_keys = []

        # Ego variables
        self._ego_actor = ego_actor
        self._ego_state = 'road'

        # Route variables
        self._waypoints = []
        self._route_accum_dist = []
        prev_trans = None
        for trans, _ in route:
            self._waypoints.append(self._map.get_waypoint(trans.location))
            if prev_trans:
                dist = trans.location.distance(prev_trans.location)
                self._route_accum_dist.append(dist + self._route_accum_dist[-1])
            else:
                self._route_accum_dist.append(0)
            prev_trans = trans

        self._route_length = len(route)
        self._route_index = 0
        self._route_buffer = 3

        # Road variables (these are very much hardcoded so watch out when changing them)
        self._road_actors = []
        self._road_front_vehicles = 3  # Amount of vehicles in front of the ego
        self._road_back_vehicles = 3  # Amount of vehicles behind the ego
        self._road_vehicle_dist = 10  # Starting distance between spawned vehicles
        self._base_min_radius = 32  # Vehicles further than this will start to slow down
        self._base_max_radius = 37  # Must be higher than nº_veh * veh_dist or the furthest vehicle will never activate
        self._radius_increase_ratio = 1.8  # Meters the radius increases per m/s of the ego
        self._leading_dist_interval = [1, 6]

        # Opposite lane variables
        self._opposite_actors = []
        self._opposite_sources = []
        self._opposite_source_dist = 60
        self._opposite_vehicle_dist = 10
        self._opposite_sources_max_actors = 6  # Maximum vehicles alive at the same time per source

        # Break scenario variables
        self._is_scenario_active = False
        self._break_actors = []
        self._break_time_interval = [30, 40]  # Min, max time between scenarios
        self._break_duration = 7  # Duration of the scenario
        self._get_next_scenario_time()

        # Lane change variables
        self._is_lane_change_active = False
        self._lane_change_actors = []
        self._lane_change_dist = 50
        self._lane_change_leading_dist_interval = [8, 14]

        self._lane_changes = []
        self._lane_change_index = 0
        ignore_point = False
        for i in range(len(route) - 1):
            if ignore_point:
                ignore_point = False
                continue
            if route[i][1] in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
                mean_accum_dist = (self._route_accum_dist[i] + self._route_accum_dist[i+1]) / 2
                self._lane_changes.append(mean_accum_dist)
                ignore_point = True

        # Junction variables
        self._junctions = []
        self._active_junctions = []
        self._previous_actors = []  # Actors created at the previous junction
        self._previous_sources = []

        self._junction_join_dist = 34  # Two junctions closer than this will be considered as one
        self._junction_detection_dist = 45  # Distance from which junction actors are spawned
        self._junction_exit_dist = 35  # Max distance actors travel after exiting a junction
        self._entry_sources_dist = 25  # Distance from the entry sources to the junction
        self._entry_sources_max_actors = 6  # Maximum vehicles alive at the same time per source


    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        ego_wp = self._waypoints[0]
        same_dir_wps = self._get_same_dir_lanes(ego_wp)
        opposite_dir_wps = self._get_opposite_dir_lanes(ego_wp)
        self._initialise_road_behavior(same_dir_wps, ego_wp)
        self._initialise_opposite_sources(opposite_dir_wps)
        self._create_junction_actor_dict()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        prev_ego_index = self._route_index

        # TODO: Use Junction, Road and Source class
        # TODO: Check capacity of connecting lanes
        # TODO: Micro lanes from the scratch
        # TODO: Recycle heroes
        #   1) 360º opposite-exit
        #   2) 360º opposite-previous
        #   3) Entry - exit connecting lanes

        # Get ego's odometry. For robustness, the closest route point will be used
        location = CarlaDataProvider.get_location(self._ego_actor)
        ego_wp = self._update_ego_route_location(location)
        ego_transform = ego_wp.transform
        if self.debug:
            self._world.debug.draw_string(location, self._ego_state, False, carla.Color(0,0,0), 0.05)

        # Monitor the background activity's state
        if self._ego_state == 'road':
            self._update_road_radius()
            self._update_road_actors(ego_transform)
            self._move_opposite_sources(prev_ego_index, self._route_index)

        elif self._ego_state == 'junction':
            self._update_junction_actors()
            self._update_junction_entrances()
            self._monitor_ego_junction_exit(ego_wp)

        # Manage non-state related behaviors
        if self._is_scenario_active or self._ego_state == 'road':
            self._manage_break_scenario()

        self._monitor_nearby_junctions()
        self._update_exit_sources(ego_transform.location)

        self._manage_lane_change_scenario()
    
        self._update_previous_actors(ego_transform)
        self._update_opposite_actors(ego_transform)
        self._update_opposite_sources()

        return new_status

    def terminate(self, new_status):
        """Destroy all actors"""
        all_actors = self._get_actors()
        for actor in list(all_actors):
            self._destroy_actor(actor)
        super(BackgroundBehavior, self).terminate(new_status)

    def _get_actors(self):
        """returns a list of all actors part of the background activity"""
        actors = self._road_actors + self._previous_actors + self._opposite_actors
        for junction in self._active_junctions:
            if 'actor_dict' in junction:
                actors.extend(list(junction['actor_dict']))
        return actors

    def _get_relevant_actors(self):
        """returns a list of all actors part of the background activity"""
        relevant_actors = self._road_actors
        for junction in self._active_junctions:
            if 'actor_dict' in junction:
                relevant_actors.extend(list(junction['actor_dict']))
        return relevant_actors


    ################################
    ##       Junction cache       ##
    ################################

    def _create_junction_actor_dict(self):
        """Extracts the junctions the ego vehicle will pass through"""
        data = self._get_junctions_data()
        fake_data, filtered_data = self._filter_fake_junctions(data)
        self._get_fake_lane_pairs(fake_data)
        route_data = self._join_roundabout_junctions(filtered_data)
        self._add_junctions_topology(route_data)
        self._junctions = route_data

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self._junctions)

    def _get_junctions_data(self):
        """Gets all the junctions the ego passes through"""
        junction_data = []
        index = 0

        # Ignore the junction the ego spawns at
        for i in range(0, self._route_length - 1):
            if not self._is_junction(self._waypoints[i]):
                index = i
                break

        for i in range(index, self._route_length - 1):
            wp = self._waypoints[i]
            next_wp = self._waypoints[i+1]

            # Searching for the junction exit
            if len(junction_data) != 0 and junction_data[-1]['route_exit_index'] is None:
                if not self._is_junction(next_wp) or next_wp.get_junction().id != junction_id:
                    junction_data[-1]['route_exit_index'] = i+1
            # Searching for a junction
            elif self._is_junction(next_wp):
                junction_id = next_wp.get_junction().id
                if len(junction_data) != 0 and junction_data[-1]['junctions'][-1].id == junction_id:
                    junction_data[-1]['junctions'].append(next_wp.get_junction())
                    junction_data[-1]['route_exit_index'] = None
                else:
                    junction_data.append({
                        'junctions': [next_wp.get_junction()],  # For multijunction later on
                        'route_enter_index': i,
                        'route_exit_index': None
                    })

        return junction_data

        # Cause micro lanes are fun, check that the route points aren't part of them
        # for junction_data in junctions:

        #     # Entry. Compare it with its previous point (minimum distance of 1m)
        #     enter_index = junction_data['route_enter_index']
        #     enter_wp = self._waypoints[enter_index]
        #     enter_accum_dist = self._route_accum_dist[enter_index]
        #     for i in range(enter_index, 0, -1):
        #         if enter_accum_dist - self._route_accum_dist[i] < 1:
        #             continue
        #         prev_enter_wp = self._waypoints[i]
        #         if not prev_enter_wp.is_junction and prev_enter_wp.road_id != enter_wp.road_id:
        #             junction_data['route_enter_index'] = i
        #         break

        #     # Exit. Compare it with its next point (minimum distance of 1m)
        #     exit_index = junction_data['route_enter_index']
        #     exit_wp = self._waypoints[exit_index]
        #     exit_accum_dist = self._route_accum_dist[exit_index]
        #     for i in range(exit_index, self._route_length - 1):
        #         if self._route_accum_dist[i] - exit_accum_dist < 1:
        #             continue
        #         next_exit_wp = self._waypoints[i]
        #         if not next_exit_wp.is_junction and next_exit_wp.road_id != exit_wp.road_id:
        #             junction_data['route_enter_index'] = i
        #         break

    def _filter_fake_junctions(self, data):
        """Filters fake junctions. As a general note, a fake junction is that where no road lane divide in two.
        However, this fails for CARLA maps, so check junctions which have all lanes straight."""
        fake_data = []
        filtered_data = []
        threshold = math.radians(15)

        for junction_data in data:
            found_turn = False
            for enter_wp, exit_wp in junction_data['junctions'][0].get_waypoints(carla.LaneType.Driving):
                enter_heading = enter_wp.transform.get_forward_vector()
                exit_heading = exit_wp.transform.get_forward_vector()
                dot = enter_heading.x * exit_heading.x + enter_heading.y * exit_heading.y
                if dot < math.cos(threshold):
                    found_turn = True
                    break

            if not found_turn:
                fake_data.append(junction_data)
            else:
                filtered_data.append(junction_data)

        return fake_data, filtered_data

    def _get_roundabouts_info(self):
        """Function to hardcode the roundabout topology, as the current API doesn't offer that info"""
        roundabout_junctions = []
        fake_lane_keys = []

        if 'Town03' in self._map.name:
            roundabout_junctions.extend([
                self._map.get_waypoint_xodr(1100, -5, 16.6).get_junction(),
                self._map.get_waypoint_xodr(1624, -5, 25.3).get_junction(),
                self._map.get_waypoint_xodr(1655, -5, 8.3).get_junction(),
                self._map.get_waypoint_xodr(1772, 3, 16.2).get_junction(),
                self._map.get_waypoint_xodr(1206, -5, 5.9).get_junction()])
            fake_lane_keys.extend([
                ['37*-4','36*-4'], ['36*-4','37*-4'],
                ['37*-5','36*-5'], ['36*-5','37*-5'],
                ['38*-4','12*-4'], ['12*-4','38*-4'],
                ['38*-5','12*-5'], ['12*-5','38*-5']])
            # Micro lanes at the roundabout. Better to just hardcode them as fake junction

        self._fake_lane_pair_keys.extend(fake_lane_keys)

        return roundabout_junctions

    def _join_roundabout_junctions(self, filtered_data):
        """Joins roundabout junctions into one, adding the whole roundabout"""
        print(len(filtered_data))
        route_data = []
        roundabout_junctions = self._get_roundabouts_info()

        # If entering a roundabout, add all its junctions to the list
        roundabout_ids = [j.id for j in roundabout_junctions]
        for junction_data in filtered_data:
            junction = junction_data['junctions'][0]

            # Join the same junction in one, or two consecutive junctions part of the same roundabout
            if len(route_data) > 0:
                prev_junction_ids = [j.id for j in route_data[-1]['junctions']]
                if junction.id in prev_junction_ids:
                    route_data[-1]['route_exit_index'] = junction_data['route_exit_index']
                    continue

            # Add all roundabout junctions
            if junction.id in roundabout_ids:
                junction_ids = [j.id for j in junction_data['junctions']]
                for roundabout_junction in roundabout_junctions:
                    if roundabout_junction.id not in junction_ids:
                        junction_data['junctions'].append(roundabout_junction)

            route_data.append(junction_data)

        return route_data

    def _get_fake_lane_pairs(self, fake_data):
        """Gets a list of enter-exit lanes of the fake junctions"""
        for fake_junctions_data in fake_data:
            for junction in fake_junctions_data['junctions']:
                for enter_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
                    while self._is_junction(enter_wp):
                        enter_wps = enter_wp.previous(0.5)
                        if len(enter_wps) == 0:
                            break  # Stop when there's no prev
                        enter_wp = enter_wps[0]
                    if self._is_junction(enter_wp):
                        continue  # Triggered by the loops break

                    while self._is_junction(exit_wp):
                        exit_wps = exit_wp.next(0.5)
                        if len(exit_wps) == 0:
                            break  # Stop when there's no prev
                        exit_wp = exit_wps[0]
                    if self._is_junction(exit_wp):
                        continue  # Triggered by the loops break

                    self._fake_junction_ids.append(junction.id)
                    self._fake_lane_pair_keys.append([self._lane_key(enter_wp), self._lane_key(exit_wp)])

    def _get_junction_entry_wp(self, enter_wp):
        """For a junction waypoint, returns a waypoint outside of it that enters into its lane"""
        # Exit the junction
        while self._is_junction(enter_wp):
            enter_wps = enter_wp.previous(0.2)
            if len(enter_wps) == 0:
                break  # Stop when there's no prev
            enter_wp = enter_wps[0]
        if self._is_junction(enter_wp):
            return None  # Triggered by the loops break

        return enter_wp
        # Cause micro lanes, move 'a bit' further back
        # enter_wps_1 = enter_wp.previous(1)
        # if len(enter_wps_1) == 0:
        #     return None  # Stop when there's no prev
        # enter_wp_1 = enter_wps_1[0]
        # enter_wps_2 = enter_wp_1.previous(2)
        # if len(enter_wps_2) == 0:
        #     return None  # Stop when there's no prev
        # enter_wp_2 = enter_wps_2[0]

        # if not self._is_junction(enter_wp_1) and enter_wp_1.road_id != enter_wp.road_id:
        #     print("WARNING: Found micro lane at an entrance")
        #     self._micro_lane_pair_keys.append([self._lane_key(enter_wp), self._lane_key(enter_wp_1)])
        #     return enter_wp_1  # enter_wp is part of a micro lane
        # elif not self._is_junction(enter_wp_2) and enter_wp_2.road_id != enter_wp.road_id:
        #     print("WARNING: Found micro lane at an entrance")
        #     self._micro_lane_pair_keys.append([self._lane_key(enter_wp), self._lane_key(enter_wp_2)])
        #     return enter_wp_2  # enter_wp is part of a micro lane
        # else:
        #     return enter_wp  # enter_wp is not part of a micro lane

    def _get_junction_exit_wp(self, exit_wp):
        """For a junction waypoint, returns a waypoint outside of it from which the lane exits the junction"""
        while self._is_junction(exit_wp):
            exit_wps = exit_wp.next(0.2)
            if len(exit_wps) == 0:
                break  # Stop when there's no prev
            exit_wp = exit_wps[0]
        if self._is_junction(exit_wp):
            return None  # Triggered by the loops break

        return exit_wp
        # Cause micro lanes, move 'a bit' further back
        # exit_wps_1 = exit_wp.next(1)
        # if len(exit_wps_1) == 0:
        #     return None  # Stop when there's no prev
        # exit_wp_1 = exit_wps_1[0]
        # exit_wps_2 = exit_wp_1.next(1)
        # if len(exit_wps_2) == 0:
        #     return None  # Stop when there's no prev
        # exit_wp_2 = exit_wps_2[0]

        # if not self._is_junction(exit_wp_1) and exit_wp_1.road_id != exit_wp.road_id:
        #     print("WARNING: Found micro lane at an exit")
        #     self._micro_lane_pair_keys.append([self._lane_key(exit_wp), self._lane_key(exit_wp_1)])
        #     return exit_wp_1  # exit_wp is part of a micro lane
        # elif not self._is_junction(exit_wp_2) and exit_wp_2.road_id != exit_wp.road_id:
        #     print("WARNING: Found micro lane at an exit")
        #     self._micro_lane_pair_keys.append([self._lane_key(exit_wp), self._lane_key(exit_wp_2)])
        #     return exit_wp_2  # exit_wp is part of a micro lane
        # else:
        #     return exit_wp  # exit_wp is not part of a micro lane

    def _get_closest_junction_waypoint(self, route_wp, junction_wps):
        """a"""
        closest_dist = float('inf')
        closest_wp = None
        route_location = route_wp.transform.location
        for wp in junction_wps:
            distance = wp.transform.location.distance(route_location)
            if distance < closest_dist:
                closest_dist = distance
                closest_wp = wp

        return closest_wp

    def _is_route_wp_behind_junction_wp(self, route_wp, junction_wp):
        """Checks if an actor is behind the ego. Uses the route transform"""
        route_location = route_wp.transform.location
        junction_transform = junction_wp.transform
        junction_heading = junction_transform.get_forward_vector()
        wps_vec = route_location - junction_transform.location
        if junction_heading.x * wps_vec.x + junction_heading.y * wps_vec.y < - 0.17:  # 100º
            return True
        return False

    def _add_junctions_topology(self, route_data):
        """Gets the entering and exiting lanes of a multijunction"""
        for junction_data in route_data:
            used_entering_lanes = []
            used_exiting_lanes = []
            entering_lane_wps = []
            exiting_lane_wps = []

            if self.debug:
                print(' --------------------- ')
            for junction in junction_data['junctions']:
                for enter_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving): 

                    enter_wp = self._get_junction_entry_wp(enter_wp)
                    if not enter_wp:
                        continue
                    if self._lane_key(enter_wp) not in used_entering_lanes:
                        used_entering_lanes.append(self._lane_key(enter_wp))
                        entering_lane_wps.append(enter_wp)
                        if self.debug:
                            self._world.debug.draw_point(
                                enter_wp.transform.location + carla.Location(z=1), size=0.1,
                                color=carla.Color(255,255,0), life_time=10000)

                    exit_wp = self._get_junction_exit_wp(exit_wp)
                    if not exit_wp:
                        continue
                    if self._lane_key(exit_wp) not in used_exiting_lanes:
                        used_exiting_lanes.append(self._lane_key(exit_wp))
                        exiting_lane_wps.append(exit_wp)
                        if self.debug:
                            self._world.debug.draw_point(
                                exit_wp.transform.location + carla.Location(z=1), size=0.1,
                                color=carla.Color(0,255,255), life_time=10000)

            # Check for route junction overlapping information (aka micro lanes)
            entering_lane_keys = [self._lane_key(wp) for wp in entering_lane_wps]
            route_entry_wp = self._waypoints[junction_data['route_enter_index']]
            if self._lane_key(route_entry_wp) not in entering_lane_keys:
                print('WARNING: Found a junction with a route entry lane different than isnt its entry lanes')
                # junction_entry_wp = self._get_closest_junction_waypoint(route_entry_wp, entering_lane_wps)
                # entering_lane_wps = [route_entry_wp if i == junction_entry_wp else i for i in entering_lane_wps]
                # self._micro_lane_pair_keys.append([self._lane_key(route_entry_wp), self._lane_key(junction_entry_wp)])

            exiting_lane_keys = [self._lane_key(wp) for wp in exiting_lane_wps]
            if junction_data['route_exit_index']:
                route_exit_wp = self._waypoints[junction_data['route_exit_index']]
            else:
                route_exit_wp = None
            if route_exit_wp and self._lane_key(route_exit_wp) not in exiting_lane_keys:
                print('WARNING: Found a junction with a route exit lane different than isnt its exit lanes')
                # junction_exit_wp = self._get_closest_junction_waypoint(route_exit_wp, exiting_lane_wps)
                # exiting_lane_wps = [route_exit_wp if i == junction_exit_wp else i for i in exiting_lane_wps]
                # self._micro_lane_pair_keys.append([self._lane_key(route_exit_wp), self._lane_key(junction_exit_wp)])

            ################# This is pretty much for the roundabouts. But better to check for all
            # Connecting lanes (they exit one junction and enter another or viceversa)
            exiting_lane_keys = [self._lane_key(wp) for wp in exiting_lane_wps]
            entering_lane_keys = [self._lane_key(wp) for wp in entering_lane_wps]
            for wp in list(entering_lane_wps):
                if self._lane_key(wp) in exiting_lane_keys:
                    entering_lane_wps.remove(wp)
                    if self.debug:
                        self._world.debug.draw_point(
                            wp.transform.location + carla.Location(z=1), size=0.1,
                            color=carla.Color(255,0,255), life_time=10000)
            for wp in list(exiting_lane_wps):
                if self._lane_key(wp) in entering_lane_keys:
                    exiting_lane_wps.remove(wp)
                    if self.debug:
                        self._world.debug.draw_point(
                            wp.transform.location + carla.Location(z=1), size=0.1,
                            color=carla.Color(255,0,255), life_time=10000)

            # Connecting lanes with a fake junction in the middle (same as before but a bit more complex)
            for enter_key, exit_key in self._fake_lane_pair_keys:
                entry_wp = None
                for wp in entering_lane_wps:
                    if self._lane_key(wp) == exit_key:  # A junction exit is a fake junction entry
                        entry_wp = wp
                        break
                exit_wp = None
                for wp in exiting_lane_wps:
                    if self._lane_key(wp) == enter_key:  # A junction entry is a fake junction exit
                        exit_wp = wp
                        break
                if entry_wp and exit_wp:
                    entering_lane_wps.remove(entry_wp)
                    exiting_lane_wps.remove(exit_wp)
                    if self.debug:
                        self._world.debug.draw_point(
                            entry_wp.transform.location + carla.Location(z=1), size=0.1,
                            color=carla.Color(255,0,255), life_time=10000)
                        self._world.debug.draw_point(
                            exit_wp.transform.location + carla.Location(z=1), size=0.1,
                            color=carla.Color(255,0,255), life_time=10000)

            ####################

            junction_data['enter_wps'] = entering_lane_wps
            junction_data['exit_wps'] = exiting_lane_wps

            if self.debug:
                exit_lane = self._waypoints[junction_data['route_exit_index']] if junction_data['route_exit_index'] else None
                print('> R Enter Lane: {}'.format(self._lane_key(self._waypoints[junction_data['route_enter_index']])))
                print('> R Exit  Lane: {}'.format(self._lane_key(exit_lane)))
                entry = '> J Enter Lanes: '
                for enter_wp in entering_lane_wps:
                    key = self._lane_key(enter_wp)
                    entry += key + ' ' * (6 - len(key))
                print(entry)
                exit = '> J Exit  Lanes: '
                for exit_wp in exiting_lane_wps:
                    key = self._lane_key(exit_wp)
                    exit += key + ' ' * (6 - len(key))
                print(exit)

    ################################
    ## Waypoint related functions ##
    ################################

    def _is_junction(self, wp):
        if not wp.is_junction or wp.junction_id in self._fake_junction_ids:
            return False
        return True

    def _get_same_dir_lanes(self, waypoint):
        """Gets all the lanes with the same direction of the road of a wp"""
        same_dir_wps = [waypoint]

        # Check roads on the right
        right_wp = waypoint
        while True:
            possible_right_wp = right_wp.get_right_lane()
            if possible_right_wp is None or possible_right_wp.lane_type != carla.LaneType.Driving:
                break
            right_wp = possible_right_wp
            same_dir_wps.append(right_wp)

        # Check roads on the left
        left_wp = waypoint
        while True:
            possible_left_wp = left_wp.get_left_lane()
            if possible_left_wp is None or possible_left_wp.lane_type != carla.LaneType.Driving:
                break
            if possible_left_wp.lane_id * left_wp.lane_id < 0:
                break
            left_wp = possible_left_wp
            same_dir_wps.append(left_wp)

        return same_dir_wps

    def _get_opposite_dir_lanes(self, ego_wp):
        """Gets all the lanes with opposite direction of the road of a wp"""
        other_dir_wps = []
        other_dir_wp = None

        # Get the first lane of the opposite direction
        left_wp = ego_wp
        while True:
            possible_left_wp = left_wp.get_left_lane()
            if possible_left_wp is None:
                break
            if possible_left_wp.lane_id * left_wp.lane_id < 0:
                other_dir_wp = possible_left_wp
                break
            left_wp = possible_left_wp

        if not other_dir_wp:
            return other_dir_wps

        # Check roads on the right
        right_wp = other_dir_wp
        while True:
            if right_wp.lane_type == carla.LaneType.Driving:
                other_dir_wps.append(right_wp)
            possible_right_wp = right_wp.get_right_lane()
            if possible_right_wp is None:
                break
            right_wp = possible_right_wp

        return other_dir_wps

    def _lane_key(self, wp):
        """Returns a key corresponding to the waypoint lane"""
        return '' if wp is None else self._road_key(wp) + '*' + str(wp.lane_id)

    def _road_key(self, wp):
        """Returns a key corresponding to the waypoint road"""
        return '' if wp is None else str(wp.road_id)

    def _ids_from_key(self, lane_key):
        """Returns the road and lane ids from a key"""
        road_id, lane_id = lane_key.split('*')
        return road_id, lane_id

    def _is_actor_a_road_actor(self, actor, ego_road_key, junction):
        """Searches the exit lanes for a specific actor. Faster than comparing road and lane ids"""
        exit_dict = junction['exit_dict']
        for lane_key in exit_dict:
            if actor in exit_dict[lane_key]:
                road_id, _ = self._ids_from_key(lane_key)
                return road_id == ego_road_key

        return False

    ################################
    ##       Mode functions       ##
    ################################
    def _clear_opposite_traffic(self, junction):
        """Sometimes, such as at roundabouts, the route exit will overlap with the opposite traffic road.
        Detect these cases to avoid weird interactions, removing the opposite traffic"""
        if len(self._opposite_sources) == 0:
            return

        route_exit_wp = self._waypoints[junction['route_exit_index']]
        route_exit_key = self._road_key(route_exit_wp)
        if not self._is_junction(self._opposite_sources[0]['wp']):
            source_wp = self._opposite_sources[0]['wp']
        else:
            source_wp = self._opposite_sources[0]['wp']
            while self._is_junction(source_wp):
                next_wps = source_wp.next(0.5)
                if len(next_wps) == 0:
                    break
                source_wp = next_wps[0]

        if self._road_key(source_wp) == route_exit_key:
            for actor in list(self._opposite_actors):
                self._destroy_actor(actor)

    def _switch_to_junction_mode(self, junction):
        """Prepares the junction mode, storing some junction values and changing the state of the actors"""
        self._ego_state = 'junction'
        for actor in list(self._road_actors):
            junction['actor_dict'][actor] = {'state': 'junction_entry', 'ref_wp': None}
            self._road_actors.remove(actor)
            if not self._is_scenario_active:
                self._tm.vehicle_percentage_speed_difference(actor, 0)

        self._clear_opposite_traffic(junction)
        self._opposite_sources.clear()

    def _end_junction_behavior(self, ego_wp, junction):
        """Destroying unneeded actors, remembering the rest and cleaning up variables of the exited junction.
        If no other junctions are active, starts road mode"""
        for actor in list(junction['actor_dict']):
            location = CarlaDataProvider.get_location(actor)
            if not location or self._is_location_behind_ego(location):
                self._destroy_actor(actor)
                continue

            self._tm.vehicle_percentage_speed_difference(actor, 0)
            if self._is_actor_a_road_actor(actor, self._road_key(ego_wp), junction):
                if len(self._active_junctions) > 1:
                    # Move actors to the next junctions
                    self._active_junctions[1]['actor_dict'][actor] = {'state': 'junction_entry', 'ref_wp': None}
                else:
                    self._road_actors.append(actor)
            else:
                self._previous_actors.append(actor)

        self._switch_junction_exit_sources(junction)

        if len(self._active_junctions) <= 1:
            self._ego_state = 'road'
            opposite_dir_wps = self._get_opposite_dir_lanes(ego_wp)
            self._initialise_opposite_sources(opposite_dir_wps)

        self._active_junctions.pop(0)

    def _switch_junction_exit_sources(self, junction):
        """Removes the active sources (part of the previous road) and get the ones of the exitted junction.
        Does nothing if another junction is active"""
        self._previous_sources.clear()
        self._previous_sources.extend(junction['exit_sources'])

    def _search_for_next_junction(self):
        """Check if closeby to a junction. The closest one will always be the first"""
        if not self._junctions:
            return None

        ego_accum_dist = self._route_accum_dist[self._route_index]
        junction_accum_dist = self._route_accum_dist[self._junctions[0]['route_enter_index']]
        if junction_accum_dist - ego_accum_dist < self._junction_detection_dist:  # Junctions closeby
            return self._junctions.pop(0)

        return None

    def _monitor_nearby_junctions(self):
        """Monitors when the ego approaches a junction. If that's the case, prepares the junction mode.
        This can be triggered even if there is another junction behavior happening"""
        junction = self._search_for_next_junction()
        if not junction:
            return

        junction['actor_dict'] = OrderedDict()
        if self._ego_state == 'road':
            self._switch_to_junction_mode(junction)

        route_enter_wp = self._waypoints[junction['route_enter_index']]
        self._initialise_junction_entrances(junction, route_enter_wp)

        route_exit_wp = self._waypoints[junction['route_exit_index']] if junction['route_exit_index'] else None
        self._initialise_junction_exits(junction, route_exit_wp)

        self._active_junctions.append(junction)

    def _monitor_ego_junction_exit(self, ego_wp):
        """Monitors when the ego exits the junctions. If that's the case, prepares the road mode"""
        current_junction = self._active_junctions[0]
        if self._lane_key(ego_wp) in current_junction['exit_dict']:
            self._end_junction_behavior(ego_wp, current_junction)

    def _update_exit_sources(self, ego_location):
        """Manages the sources that spawn actors behind the ego. Sources are destroyed after their actors are spawned"""
        for source_info in list(self._previous_sources):
            transform = source_info['transform']
            actors = source_info['actors']

            if self.debug:
                self._world.debug.draw_point(
                    transform.location + carla.Location(z=1), size=0.1,
                    color=carla.Color(255,0,0), life_time=0.2)

            if len(actors) >= self._road_back_vehicles:
                self._previous_sources.remove(source_info)
                continue

            if len(actors) == 0:
                location = ego_location
                max_dist = self._road_vehicle_dist
            else:
                location = actors[0].get_location()
                max_dist = 0.5 * self._road_vehicle_dist

            distance = location.distance(transform.location)

            # Spawn a new actor if the last one is far enough
            if distance > max_dist:
                actor = CarlaDataProvider.request_new_actor(
                    'vehicle.*', transform, rolename='background',
                    autopilot=True, random_location=False, safe_blueprint=True, tick=False
                )
                if actor is None:
                    continue

                if self._ego_state == 'road':
                    self._save_actor_info(actor)
                    self._road_actors.append(actor)
                elif self._ego_state == 'junction':
                    self._save_actor_info(actor)
                    self._active_junctions[0]['actor_dict'][actor] = {'state': 'junction_entry', 'ref_wp': None}
                source_info['actors'].append(actor)


    ################################
    ## Behavior related functions ##
    ################################
    def _get_leading_distance(self, min_value=None, max_value=None):
        if not min_value:
            min_value = self._leading_dist_interval[0]
        if not max_value:
            max_value = self._leading_dist_interval[1]
        return min_value + (max_value - min_value) * self._rng.rand()

    def _save_actor_info(self, actor):
        """Saved the actor information"""
        self._tm.distance_to_leading_vehicle(actor, self._get_leading_distance())
        self._tm.auto_lane_change(actor, False)

    def _initialise_road_behavior(self, road_wps, ego_wp):
        """Intialises the road behavior, consisting on several vehicle in front of the ego"""
        spawn_points = []
        # Vehicles in front
        for wp in road_wps:
            next_wp = wp
            for _ in range(self._road_front_vehicles):
                next_wps = next_wp.next(self._road_vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                spawn_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=0.2), next_wp.transform.rotation))

        # Vehicles on the side
        for wp in road_wps:
            if wp.lane_id != ego_wp.lane_id:
                spawn_points.append(carla.Transform(
                    wp.transform.location + carla.Location(z=0.2), wp.transform.rotation))

        # Vehicles behind
        for wp in road_wps:
            prev_wp = wp
            for _ in range(self._road_back_vehicles):
                prev_wps = prev_wp.previous(self._road_vehicle_dist)
                if len(prev_wps) == 0:
                    break  # Stop when there's no next
                prev_wp = prev_wps[0]
                spawn_points.append(carla.Transform(
                    prev_wp.transform.location + carla.Location(z=0.2), prev_wp.transform.rotation))

        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_points), spawn_points, True, False, 'background', safe_blueprint=True, tick=False)

        for actor in actors:
            self._save_actor_info(actor)
            self._road_actors.append(actor)

    def _initialise_opposite_sources(self, road_wps):
        """Creates actor sources that spawn actors in the opposite direction"""
        for wp in road_wps:
            moved_dist = 0
            prev_wp = wp
            while moved_dist < self._opposite_source_dist and not self._is_junction(prev_wp):
                prev_wps = prev_wp.previous(2)
                if len(prev_wps) == 0:
                    break
                prev_wp = prev_wps[0]
                moved_dist += 2
            self._opposite_sources.append({'wp': prev_wp, 'actors':[]})

    def _initialise_junction_entrances(self, junction, route_enter_wp):
        """Initializes the actor sources to ensure the junction is always populated"""
        enter_wps = junction['enter_wps']
        junction['entry_sources'] = []

        for wp in enter_wps:
            if wp.road_id == route_enter_wp.road_id:
                continue  # Ignore the road from which the route enters

            prev_wps = wp.previous(self._entry_sources_dist)
            if len(prev_wps) == 0:
                continue  # Stop when there's no prev
            prev_wp = prev_wps[0]

            source_transform = carla.Transform(
                prev_wp.transform.location + carla.Location(z=0.2), prev_wp.transform.rotation)
            junction['entry_sources'].append({'transform': source_transform, 'actors':[]})

    def _initialise_junction_exits(self, junction, route_exit_wp):
        """Spawns the road mode actors at the route's exit road and creates the pseudo actor sink dictionary"""
        exit_wps = junction['exit_wps']
        exit_route_key = self._road_key(route_exit_wp)
        junction['exit_dict'] = OrderedDict()
        junction['exit_sources'] = []

        for wp in exit_wps:
            junction['exit_dict'][self._lane_key(wp)] = []
            if self._road_key(wp) != exit_route_key:
                continue  # Nothing to spawn

            source_transform = carla.Transform(
                wp.transform.location + carla.Location(z=0.2), wp.transform.rotation)
            junction['exit_sources'].append(
                {'transform': source_transform, 'actors': []})

            exiting_points = []
            next_wp = wp
            for _ in range(self._road_front_vehicles):
                next_wps = next_wp.next(self._road_vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                exiting_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=0.15), next_wp.transform.rotation))

            actors = CarlaDataProvider.request_new_batch_actors(
                'vehicle.*', len(exiting_points), exiting_points, True, False, 'background',
                safe_blueprint=True, tick=False
            )
            for actor in actors:
                self._save_actor_info(actor)
                junction['actor_dict'][actor] = {'state': 'junction_exit', 'ref_wp': wp}
                junction['exit_dict'][self._lane_key(wp)].insert(0, actor)  # The front most has to be index 0

    def _update_junction_entrances(self):
        """Checks the actor sources to see if new actors have to be created"""
        for junction in self._active_junctions:
            actor_dict = junction['actor_dict']
            for source_info in junction['entry_sources']:
                transform = source_info['transform']
                actors = source_info['actors']

                # Cap the amount of alive actors
                if len(actors) >= self._entry_sources_max_actors:
                    continue

                # Calculate distance to the last created actor
                if len(actors) == 0:
                    distance = self._road_vehicle_dist + 1
                else:
                    actor_location = CarlaDataProvider.get_location(actors[0])
                    if not actor_location:
                        continue
                    distance = transform.location.distance(actor_location)

                # Spawn a new actor if the last one is far enough
                if distance > self._road_vehicle_dist:
                    actor = CarlaDataProvider.request_new_actor(
                        'vehicle.*', transform, rolename='background',
                        autopilot=True, random_location=False, safe_blueprint=True, tick=False
                    )
                    if actor is None:
                        continue

                    self._save_actor_info(actor)
                    actor_dict[actor] = {'state': 'junction_entry', 'ref_wp': None}
                    source_info['actors'].append(actor)

    def _move_opposite_sources(self, prev_index, current_index):
        """Moves the sources of the opposite direction back the same amount as the ego moved"""
        if self.debug:
            for info in self._opposite_sources:
                wp = info['wp']
                self._world.debug.draw_point(
                    wp.transform.location + carla.Location(z=2), size=0.1,
                    color=carla.Color(0,0,255), life_time=0.1)

        if prev_index == current_index:
            return

        prev_accum_dist = self._route_accum_dist[prev_index]
        current_accum_dist = self._route_accum_dist[current_index]
        move_dist = current_accum_dist - prev_accum_dist

        for i, info in enumerate(self._opposite_sources):
            wp = info['wp']
            if not self._is_junction(wp):
                prev_wps = wp.previous(move_dist)
                if len(prev_wps) == 0:
                    continue
                prev_wp = prev_wps[0]
                self._opposite_sources[i]['wp'] = prev_wp

    def _update_opposite_sources(self):
        """Checks the opposite actor sources to see if new actors have to be created"""
        for source_info in self._opposite_sources:
            wp = source_info['wp']
            actors = source_info['actors']

            # Cap the amount of alive actors
            if len(actors) >= self._opposite_sources_max_actors:
                continue

            # Calculate distance to the last created actor
            if len(actors) == 0:
                distance = self._opposite_vehicle_dist + 1
            else:
                actor_location = CarlaDataProvider.get_location(actors[0])
                if not actor_location:
                    continue
                distance = wp.transform.location.distance(actor_location)

            # Spawn a new actor if the last one is far enough
            if distance > self._opposite_vehicle_dist:
                actor = CarlaDataProvider.request_new_actor(
                    'vehicle.*', wp.transform, rolename='background',
                    autopilot=True, random_location=False, safe_blueprint=True, tick=False
                )
                if actor is None:
                    continue

                self._save_actor_info(actor)
                self._opposite_actors.append(actor)
                source_info['actors'].insert(0, actor)

    def _update_road_radius(self):
        """Changed the radius dependent on the speed of the ego"""
        speed = CarlaDataProvider.get_velocity(self._ego_actor)
        self._min_radius = self._base_min_radius + self._radius_increase_ratio*speed
        self._max_radius = self._base_max_radius + self._radius_increase_ratio*speed

    def _get_next_scenario_time(self):
        """Gets the time for the next scenario"""
        a = self._break_time_interval[0]
        b = self._break_time_interval[1] - self._break_time_interval[0]
        self._next_scenario_time = a + b*(self._rng.rand())

    def _manage_break_scenario(self):
        """Randomly makes the vehicles in front of the ego suddenly break. Use a timer to avoid hving to
        reset it after junctions, making the scenario never trigger when multiple junctions are nearby"""
        self._next_scenario_time -= self._world.get_snapshot().timestamp.delta_seconds
        if self._is_scenario_active and self._next_scenario_time <= 0:
            self._is_scenario_active = False
            self._get_next_scenario_time()

            # Reset vehicles to normal behavior
            for actor in self._break_actors:
                self._tm.vehicle_percentage_speed_difference(actor, 0)
                actor.set_light_state(carla.VehicleLightState.NONE)

            self._break_actors = []

        elif not self._is_scenario_active and self._next_scenario_time <= 0:
            self._is_scenario_active = True
            self._next_scenario_time = self._break_duration

            # Stop all road actors in front of the ego
            for actor in self._road_actors:
                location = CarlaDataProvider.get_location(actor)
                if location and not self._is_location_behind_ego(location):
                    self._break_actors.append(actor)
                    self._tm.vehicle_percentage_speed_difference(actor, 100)
                    actor.set_light_state(carla.VehicleLightState.Brake)

    def _manage_lane_change_scenario(self):
        """Tracks if there are any route lane changes near the ego, and if so,
        increase a bit the distance between vehicles to help the ego"""
        def add_space(actor):
            min_dist = self._lane_change_leading_dist_interval[0]
            max_dist = self._lane_change_leading_dist_interval[1]
            self._tm.distance_to_leading_vehicle(actor, self._get_leading_distance(min_dist, max_dist))
        def remove_space(actor):
            self._tm.distance_to_leading_vehicle(actor, self._get_leading_distance())
        if len(self._lane_changes) == 0:
            return

        current_accum_dist = self._route_accum_dist[self._route_index]
        if self._lane_change_index < len(self._lane_changes):
            next_lane_change_accum_dist = self._lane_changes[self._lane_change_index]
            next_lane_change_dist = next_lane_change_accum_dist - current_accum_dist
        else:
            next_lane_change_accum_dist = float('inf')
            next_lane_change_dist = float('inf')

        if self._lane_change_index > 0:
            prev_lane_change_dist = current_accum_dist - self._lane_changes[self._lane_change_index - 1]
        else:
            prev_lane_change_dist = -float('inf')

        if not self._is_lane_change_active:
            if next_lane_change_dist < self._lane_change_dist:
                self._is_lane_change_active = True

        else:
            for actor in self._get_relevant_actors():
                location = CarlaDataProvider.get_location(actor)
                if not location:
                    continue
                if actor not in self._lane_change_actors and self._is_location_behind_ego(location):
                    add_space(actor)
                    self._lane_change_actors.append(actor)
                elif actor in self._lane_change_actors and not self._is_location_behind_ego(location):
                    remove_space(actor)

            if current_accum_dist > next_lane_change_accum_dist:
                self._lane_change_index += 1
            elif next_lane_change_dist < self._lane_change_dist:
                pass  # Lane change close in front
            elif prev_lane_change_dist < self._lane_change_dist:
                pass # Lane change close behind
            else:
                self._is_lane_change_active = False
                for actor in self._lane_change_actors:
                    remove_space(actor)
                self._lane_change_actors.clear()

    #############################
    ##     Actor functions     ##
    #############################

    def _is_location_behind_ego(self, location):
        """Checks if an actor is behind the ego. Uses the route transform"""
        ego_transform = self._waypoints[self._route_index].transform
        ego_heading = ego_transform.get_forward_vector()
        ego_actor_vec = location - ego_transform.location
        if ego_heading.x * ego_actor_vec.x + ego_heading.y * ego_actor_vec.y < - 0.17:  # 100º
            return True
        return False

    def _update_road_actors(self, route_transform):
        """Dynamically controls the actor speed in front of the ego. Not applied to those behind
        so they can catch up the ego"""
        for actor in self._road_actors:
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            if self.debug:
                self._world.debug.draw_string(location, 'road', False, carla.Color(0,0,0), 0.05)
            if not self._is_scenario_active and not self._is_location_behind_ego(location):
                distance = location.distance(route_transform.location)
                speed_red = (distance - self._min_radius) / (self._max_radius - self._min_radius) * 100
                speed_red = np.clip(speed_red, 0, 100)
                self._tm.vehicle_percentage_speed_difference(actor, speed_red)

    def _handle_actor_junction_exit(self, active_junction, actor, wp):
        """Handles all checks after an actors exits a junction. This removing actors in that lane if it 
        has maximum capacity as well as changing the actor state"""
        actor_lane_key = self._lane_key(wp)
        exit_dict = active_junction['exit_dict']

        if actor_lane_key in exit_dict:
            actors = exit_dict[actor_lane_key]
            if len(actors) >= self._road_front_vehicles:
                self._destroy_actor(actors[0])  # This is always the front most vehicle
            actors.append(actor)

            active_junction['actor_dict'][actor] = {'state': 'junction_exit', 'ref_wp': wp}

    def _update_junction_actors(self):
        """Handles an actor depending on their previous state"""
        for i, junction in enumerate(self._active_junctions):
            actor_dict = junction['actor_dict']
            for actor in list(actor_dict):
                if actor not in actor_dict:
                    continue  # Actor was removed during the loop
                location = CarlaDataProvider.get_location(actor)
                if not location:
                    continue
                if self.debug:
                    self._world.debug.draw_string(location, 'junction' + str(i+1), False, carla.Color(0,0,0), 0.05)

                state, ref_wp = actor_dict[actor].values()

                # Monitor its exit and destroy an actor if needed
                if state == 'junction_entry':
                    actor_wp = self._map.get_waypoint(location)
                    if not self._is_junction(actor_wp):
                        self._handle_actor_junction_exit(junction, actor, actor_wp)

                # Deactivate them when far from the junction
                elif state == 'junction_exit':
                    distance = location.distance(ref_wp.transform.location)
                    if distance > self._junction_exit_dist:
                        self._tm.vehicle_percentage_speed_difference(actor, 100)
                        actor_dict[actor]['state'] = 'junction_inactive'

                # Destroy it if behind
                elif state == 'junction_inactive':
                    pass

    def _update_previous_actors(self, ref_transform):
        """Actors part of the previous junctions will be destroyed when far from the ego"""
        for actor in self._previous_actors:
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            if self.debug:
                self._world.debug.draw_string(location, 'previous', False, carla.Color(0,0,0), 0.05)
            distance = location.distance(ref_transform.location)
            if distance > self._max_radius and self._is_location_behind_ego(location):
                self._destroy_actor(actor)

    def _update_opposite_actors(self, ref_transform):
        """"""
        for actor in list(self._opposite_actors):
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            if self.debug:
                self._world.debug.draw_string(location, 'opposite', False, carla.Color(0,0,0), 0.05)
            distance = location.distance(ref_transform.location)
            if distance > self._max_radius and self._is_location_behind_ego(location):
                self._destroy_actor(actor)

    def _destroy_actor(self, actor):
        """Destroy the actor and all its references"""
        if actor in self._road_actors:
            self._road_actors.remove(actor)
        if actor in self._opposite_actors:
            self._opposite_actors.remove(actor)
        if actor in self._previous_actors:
            self._previous_actors.remove(actor)

        for opposite_source in self._opposite_sources:
            if actor in opposite_source['actors']:
                opposite_source['actors'].remove(actor)
                break

        for junction in self._active_junctions:
            junction['actor_dict'].pop(actor, None)

            for exit_source in junction['exit_sources']:
                if actor in exit_source['actors']:
                    exit_source['actors'].remove(actor)
                    break

            for entry_source in junction['entry_sources']:
                if actor in entry_source['actors']:
                    entry_source['actors'].remove(actor)
                    break

            for exit_keys in junction['exit_dict']:
                exit_actors = junction['exit_dict'][exit_keys]
                if actor in exit_actors:
                    exit_actors.remove(actor)
                    break

        actor.destroy()

    def _update_ego_route_location(self, location):
        """Returns the closest route location to the ego"""
        shortest_distance = float('inf')
        closest_index = -1

        for index in range(self._route_index, min(self._route_index + self._route_buffer, self._route_length)):
            ref_location = self._waypoints[index].transform.location
            dist_to_route = ref_location.distance(location)
            if dist_to_route <= shortest_distance:
                closest_index = index
                shortest_distance = dist_to_route

        if closest_index != -1:
            self._route_index = closest_index

        return self._waypoints[self._route_index]
