#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

import math

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
        self._timestep = self._world.get_settings().fixed_delta_seconds
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self._random_seed_range = range(2000, 4000, 100)
        self._rng = random.RandomState(2000)

        # Actor variables
        self._ego_actor = ego_actor
        self._ego_state = 'road'
        self._background_actors = []
        self._actor_dict = OrderedDict()  # Dict with {actor: {'state', 'ref_wp'}}

        # Road variables (these are very much hardcoded so watch out when changing them)
        self._road_front_vehicles = 3  # Amount of vehicles in front of the ego
        self._road_back_vehicles = 3  # Amount of vehicles behind the ego
        self._road_vehicle_dist = 10  # Starting distance between spawned vehicles
        self._base_min_radius = 32  # Vehicles further than this will start to slow down
        self._base_max_radius = 37  # Must be higher than nº_veh * veh_dist or the furthest vehicle will never activate
        self._radius_increase_ratio = 1.8  # Meters the radius increases per m/s of the ego

        # Scenario variables
        self._is_scenario_active = False
        self._scenario_actors = [] 
        self._scenario_min_time = 30  # Min time between scenarios
        self._scenario_max_time = 40  # Max time between scenarios
        self._scenario_duration = 7  # Duration of the scenario
        self._scenario_timer = 0  # Counts the time between scenarios
        self._get_next_scenario_time()

        # Junction variables
        self._junction_join_dist = 34  # Junction detection distance.
        self._junction_detection_dist = 45  # Distance from a junction at which the junction mode starts
        self._junction_exit_dist = 35  # There needs to be space enough for self._num_road_vehicles plus another
        self._junction_exits = OrderedDict()  # Keep track of the amount of actors per lane
        self._junction_entry_sources = []  # Tracks actors spawned by the entry sources
        self._junction_exit_sources = []  # Tracks actors spawned by the exit sources
        self._junction_exit_key = ""  # Key of the road through which the route exits the junction
        self._sources_dist = 25  # Distance from the sources to the junction
        self._prev_actors = []  # Actors created at the previous junction
        self._max_actors_per_source = 6  # Maximum vehicles alive at the same time per source

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

    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        ego_wp = self._waypoints[0]
        self._initialise_road_behavior(self._get_lanes(ego_wp), ego_wp)
        self._create_junction_dict()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        # Get ego location and waypoint
        ego_location = CarlaDataProvider.get_location(self._ego_actor)
        if ego_location is None:
            return new_status
        self._update_ego_route_location(ego_location)

        # Manage non-state related behaviors
        if self._ego_state == 'road' or self._is_scenario_active:
            self._manage_road_scenario()
        self._manage_exit_sources()

        # Monitor the background activity's state
        if self._ego_state == 'road':
            self._manage_road_radius()
            self._monitor_nearby_junctions()
        elif self._ego_state == 'junction':
            self._monitor_ego_junction_exit()
            self._manage_junction_entrances()
        else:
            raise ValueError("ego state is " + str(self._ego_state))

        # Manage the background actors
        for actor in self._background_actors:
            location = CarlaDataProvider.get_location(actor)
            if location is None:
                continue
            if self.debug:
                self._world.debug.draw_string(
                    location, self._actor_dict[actor]['state'], False, carla.Color(0,0,0), 0.05)
            self._manage_background_actor(actor, location)

        return new_status

    def terminate(self, new_status):
        """Destroy all actors"""
        for actor in self._background_actors:
            self._destroy_actor(actor)
        super(BackgroundBehavior, self).terminate(new_status)

    ################################
    ##       Junction cache       ##
    ################################

    def _get_roundabouts_info(self):
        """Function to hardcode the roundabout topology, as current API doesn't offer that info"""

        if "Town03" in self._map.name:
            self._roundabout_junctions = [
                self._map.get_waypoint_xodr(1100, -5, 16.6).get_junction(),
                self._map.get_waypoint_xodr(1624, -5, 25.3).get_junction(),
                self._map.get_waypoint_xodr(1655, -5, 8.3).get_junction(),
                self._map.get_waypoint_xodr(1772, 3, 16.2).get_junction(),
                self._map.get_waypoint_xodr(1206, -5, 5.9).get_junction()
            ]
            self._roundabout_fake_lane_keys = [
                ["37*-4","36*-4"],["37*-5","36*-5"],
                ["32*-2","33*-2"], ["32*-1","33*-1"],
                ["32*4","33*4"], ["32*5","33*5"],
                ["33*-2","32*-2"], ["33*-1","32*-1"],
                ["33*4","32*4"], ["33*5","32*5"]]
            # The last four are a weird double micro lane near gas station
        else:
            self._roundabout_junctions = []
            self._roundabout_fake_lane_keys = []

    def _create_junction_dict(self):
        """Extracts the junctions the ego vehicle will pass through"""
        self._get_roundabouts_info()
        data = self._get_junctions_data()
        filtered_data, fake_junctions = self._filter_fake_junctions(data)
        lane_pair_keys = self._get_fake_lane_pairs(fake_junctions)
        multi_data = self._join_junctions(filtered_data)
        self._add_junctions_topology(multi_data, lane_pair_keys)
        self._junctions_data = multi_data

    def _get_junctions_data(self):
        """Gets all the junctions the ego passes through"""
        junctions = []
        index = 0

        # Ignore the junction the ego spawns at
        for i in range(0, self._route_length - 1):
            if not self._waypoints[i].is_junction:
                index = i
                break

        for i in range(index, self._route_length - 1):
            wp = self._waypoints[i]
            next_wp = self._waypoints[i+1]

            # Searching for the junction exit
            if len(junctions) != 0 and junctions[-1]['route_exit_index'] is None:
                if not next_wp.is_junction or next_wp.get_junction().id != junction_id:
                    junctions[-1]['route_exit_index'] = i+1
            # Searching for a junction
            elif next_wp.is_junction:
                junction_id = next_wp.get_junction().id
                if len(junctions) != 0 and junctions[-1]['junctions'][-1].id == junction_id:
                    junctions[-1]['junctions'].append(next_wp.get_junction())
                    junctions[-1]['route_exit_index'] = None
                else:
                    junctions.append({
                        'junctions': [next_wp.get_junction()],  # For multijunction later on
                        'route_enter_index': i,
                        'route_exit_index': None
                    })

        if len(junctions) > 0 and junctions[-1]['route_exit_index'] is None:
            junctions[-1]['route_exit_index'] = self._route_length - 1

        # Cause micro lanes are fun, check that the route points aren't part of them
        for junction_data in junctions:

            # Entry. Compare it with its previous point (minimum distance of 1m)
            enter_index = junction_data['route_enter_index']
            enter_wp = self._waypoints[enter_index]
            enter_accum_dist = self._route_accum_dist[enter_index]
            for i in range(enter_index, 0, -1):
                if enter_accum_dist - self._route_accum_dist[i] < 1:
                    continue
                prev_enter_wp = self._waypoints[i]
                if not prev_enter_wp.is_junction and prev_enter_wp.road_id != enter_wp.road_id:
                    junction_data['route_enter_index'] = i
                break

            # Exit. Compare it with its next point (minimum distance of 1m)
            exit_index = junction_data['route_enter_index']
            exit_wp = self._waypoints[exit_index]
            exit_accum_dist = self._route_accum_dist[exit_index]
            for i in range(exit_index, self._route_length - 1):
                if self._route_accum_dist[i] - exit_accum_dist < 1:
                    continue
                next_exit_wp = self._waypoints[i]
                if not next_exit_wp.is_junction and next_exit_wp.road_id != exit_wp.road_id:
                    junction_data['route_enter_index'] = i
                break

        return junctions

    def _filter_fake_junctions(self, junctions_data):
        """Filters fake junctions. For CARLA maps, these are junctions which have all lanes straight"""
        filtered_junctions_data = []
        fake_junctions_data = []
        threshold = math.radians(15)

        for junction_data in junctions_data:
            found_turn = False

            for enter_wp, exit_wp in junction_data['junctions'][0].get_waypoints(carla.LaneType.Driving):
                enter_heading = enter_wp.transform.get_forward_vector()
                exit_heading = exit_wp.transform.get_forward_vector()
                dot = enter_heading.x * exit_heading.x + enter_heading.y * exit_heading.y
                if dot < math.cos(threshold):
                    found_turn = True
                    break

            if not found_turn:
                fake_junctions_data.append(junction_data)
            else:
                filtered_junctions_data.append(junction_data)
        return filtered_junctions_data, fake_junctions_data

    def _join_junctions(self, junctions):
        """Joins closeby junctions into one and if entering a roudnabout, the whole roundabout is added"""
        if len(junctions) == 0:
            return []
        multi_junctions = [junctions[0]]

        for i in range(1, len(junctions)):
            junction_data = junctions[i]
            enter_accum_dist = self._route_accum_dist[junction_data['route_enter_index']]
            prev_exit_accum_dist = self._route_accum_dist[multi_junctions[-1]['route_exit_index']]

            # Group closeby junctions
            if enter_accum_dist - prev_exit_accum_dist < self._junction_join_dist:
                multi_junctions[-1]['junctions'].append(junction_data['junctions'][0])
                multi_junctions[-1]['route_exit_index'] = junction_data['route_exit_index']
            # New junction entrance
            else:
                multi_junctions.append(junction_data)

        # If entering a roundabout, add all its junctions to the list
        roundabout_ids = [j.id for j in self._roundabout_junctions]
        for junction_data in multi_junctions:
            at_roundabout = False
            for junction in junction_data['junctions']:
                if junction.id in roundabout_ids:
                    at_roundabout = True
                    break

            if at_roundabout:
                junction_ids = [j.id for j in junction_data['junctions']]
                for roundabout_junction in self._roundabout_junctions:
                    if roundabout_junction.id not in junction_ids:
                        junction_data['junctions'].append(roundabout_junction)

        return multi_junctions

    def _get_fake_lane_pairs(self, junctions):
        """Gets a list of enter-exit lanes of the fake junctions"""
        lane_pairs = self._roundabout_fake_lane_keys
        for junction_data in junctions:
            for junction in junction_data['junctions']:
                for enter_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
                    while enter_wp.is_junction:
                        enter_wps = enter_wp.previous(0.5)
                        if len(enter_wps) == 0:
                            break  # Stop when there's no prev
                        enter_wp = enter_wps[0]
                    if enter_wp.is_junction:
                        continue  # Triggered by the loops break

                    while exit_wp.is_junction:
                        exit_wps = exit_wp.next(0.5)
                        if len(exit_wps) == 0:
                            break  # Stop when there's no prev
                        exit_wp = exit_wps[0]
                    if exit_wp.is_junction:
                        continue  # Triggered by the loops break

                    lane_pairs.append([self._lane_key(enter_wp), self._lane_key(exit_wp)])

        return lane_pairs

    def _get_junction_entry_wp(self, enter_wp):
        """For a junction waypoint, returns a waypoint outside of it that enters into its lane"""
        # Exit the junction
        while enter_wp.is_junction:
            enter_wps = enter_wp.previous(0.2)
            if len(enter_wps) == 0:
                break  # Stop when there's no prev
            enter_wp = enter_wps[0]
        if enter_wp.is_junction:
            return None  # Triggered by the loops break

        # Cause micro lanes, move a bit further back
        enter_wps_ = enter_wp.previous(1)
        if len(enter_wps_) == 0:
            return None  # Stop when there's no prev
        enter_wp_ = enter_wps_[0]

        if not enter_wp_.is_junction and enter_wp_.road_id != enter_wp.road_id:
            return enter_wp_  # enter_wp is part of a micro lane
        else:
            return enter_wp  # enter_wp is not part of a micro lane

    def _get_junction_exit_wp(self, exit_wp):
        """For a junction waypoint, returns a waypoint outside of it from which the lane exits the junction"""
        while exit_wp.is_junction:
            exit_wps = exit_wp.next(0.2)
            if len(exit_wps) == 0:
                break  # Stop when there's no prev
            exit_wp = exit_wps[0]
        if exit_wp.is_junction:
            return None  # Triggered by the loops break

        # Cause micro lanes, move a bit further back
        exit_wps_ = exit_wp.next(1)
        if len(exit_wps_) == 0:
            return None  # Stop when there's no prev
        exit_wp_ = exit_wps_[0]

        if not exit_wp_.is_junction and exit_wp_.road_id != exit_wp.road_id:
            return exit_wp_  # exit_wp is part of a micro lane
        else:
            return exit_wp  # exit_wp is not part of a micro lane

    def _add_junctions_topology(self, junctions, lane_pair_keys):
        """Gets the entering and exiting lanes of a multijunction"""
        for junction_data in junctions:
            used_entering_lanes = []
            used_exiting_lanes = []
            entering_lane_wps = []
            exiting_lane_wps = []
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
            for enter_key, exit_key in lane_pair_keys:
                entry_wp = None
                exit_wp = None
                for wp in entering_lane_wps:
                    if self._lane_key(wp) == exit_key:  # A junction exit is a fake jucntion entry
                        entry_wp = wp
                        break
                for wp in exiting_lane_wps:
                    if self._lane_key(wp) == enter_key:  # A junction entry is a fake jucntion exit
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

            junction_data['enter_wps'] = entering_lane_wps
            junction_data['exit_wps'] = exiting_lane_wps

            if self.debug:
                print(" --------------------- ")
                print("> R Enter Lane: {}".format(self._lane_key(self._waypoints[junction_data['route_enter_index']])))
                print("> R Exit  Lane: {}".format(self._lane_key(self._waypoints[junction_data['route_exit_index']])))
                entry = "> J Enter Lanes: "
                for enter_wp in entering_lane_wps:
                    key = self._lane_key(enter_wp)
                    entry += key + " " * (6 - len(key))
                print(entry)
                exit = "> J Exit  Lanes: "
                for exit_wp in exiting_lane_wps:
                    key = self._lane_key(exit_wp)
                    exit += key + " " * (6 - len(key))
                print(exit)

            # Some debugging information
            if self._lane_key(self._waypoints[junction_data['route_enter_index']]) not in entering_lane_keys:
                print("WARNING: Found a junction with a route entry lane different than its entry lanes")
            if self._lane_key(self._waypoints[junction_data['route_exit_index']]) not in exiting_lane_keys:
                print("WARNING: Found a junction with a route exit lane different than its exit lanes")

    ################################
    ## Waypoint related functions ##
    ################################

    def _get_lanes(self, waypoint):
        """Gets all the lanes of the road the ego starts at"""
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

    def _lane_key(self, wp):
        """Returns a key corresponding to the lane"""
        return "" if wp is None else self._road_key(wp) + str(wp.lane_id)

    def _road_key(self, wp):
        """Returns a key corresponding to the road"""
        return "" if wp is None else str(wp.road_id) + "*"

    def _is_actor_at_exit_road(self, actor, ego_road):
        """Searches the exit lanes dictionary for a specific actor (Faster than checking road and lanes"""
        road = None
        for lane_key in self._junction_exits:
            if actor in self._junction_exits[lane_key]:
                road = int(lane_key.split('*')[0])
                break

        if road != ego_road:
            return False
        return True

    ################################
    ##       Mode functions       ##
    ################################

    def _switch_to_junction_mode(self, junction):
        """Prepares the junction mode, storing some junction values and changing the state of the actors"""
        self._ego_state = 'junction'

        for actor in self._background_actors:
            if actor not in self._scenario_actors:
                self._tm.vehicle_percentage_speed_difference(actor, 0)
            if actor not in self._prev_actors:
                self._actor_dict[actor] = {'state': 'junction_entry', 'ref_wp': None}
            else:
                self._actor_dict[actor] = {'state': 'road', 'ref_wp': None}

        self._junction_exit_index = junction['route_exit_index']
        self._junction_exit_key = self._road_key(self._waypoints[junction['route_exit_index']])

    def _switch_to_road_mode(self):
        """Prepares the road mode, destroying unneeded actors, remembering the rest and cleaning up variables"""
        self._ego_state = 'road'

        ego_wp = self._waypoints[self._route_index]
        ego_road = ego_wp.road_id
        for actor in list(self._background_actors):
            location = CarlaDataProvider.get_location(actor)
            if not location or self._is_actor_behind(location, ego_wp.transform):
                self._destroy_actor(actor)
                continue

            if not self._is_actor_at_exit_road(actor, ego_road):
                self._prev_actors.append(actor)
            self._tm.vehicle_percentage_speed_difference(actor, 0)
            self._actor_dict[actor] = {'state': 'road', 'ref_wp': None}

        self._junction_exit_key = ""
        self._junction_exit_index = None
        self._junction_exits.clear()
        self._junction_entry_sources.clear()
        self._junctions_data.pop(0)  # Remove the junction data
        self._switch_junction_exit_sources()

    def _switch_junction_exit_sources(self):
        """Removes the active sources (part of the previous road) and activates the inactive ones"""
        for i, source_info in enumerate(list(self._junction_exit_sources)):
            if source_info["active"]:
                self._junction_exit_sources.remove(source_info)

        for i in range(len(self._junction_exit_sources)): 
            self._junction_exit_sources[i]["active"] = True


    def _search_for_junction(self):
        """Check if closeby to a junction. The closest one will always be the first"""
        if not self._junctions_data:
            return None

        ego_accum_dist = self._route_accum_dist[self._route_index]
        junction_accum_dist = self._route_accum_dist[self._junctions_data[0]['route_enter_index']]
        if junction_accum_dist - ego_accum_dist < self._junction_detection_dist:  # Junctions closeby
            return self._junctions_data[0]

        return None

    def _monitor_nearby_junctions(self):
        """Monitors when the ego approaches a junction. If that's the case, prepares the junction mode"""
        junction = self._search_for_junction()
        if not junction:
            return

        self._switch_to_junction_mode(junction)
        route_enter_wp = self._waypoints[junction['route_enter_index']]
        route_exit_wp = self._waypoints[junction['route_exit_index']] if junction['route_exit_index'] else None
        self._initialise_junction_entrances(junction['enter_wps'], route_enter_wp)
        self._initialise_junction_exits(junction['exit_wps'], route_exit_wp)

    def _monitor_ego_junction_exit(self):
        """Monitors when the ego exits the junctions. If that's the case, prepares the road mode"""
        if self._route_index > self._junction_exit_index:
            self._switch_to_road_mode()

    def _manage_exit_sources(self):
        """Manages the sources that spawn actors behind the ego"""
        for i, info in enumerate(self._junction_exit_sources):
            transform = info["transform"]
            active = info["active"]
            last_actor = info["last_actor"]
            num_actors = info["num_actors"]

            if self.debug:
                self._world.debug.draw_point(
                    transform.location + carla.Location(z=1), size=0.1,
                    color=carla.Color(255,0,0), life_time=0.2)
                self._world.debug.draw_string(
                    transform.location, str(active), False, carla.Color(0,0,0), life_time=0.2)

            if num_actors >= self._road_back_vehicles or not active:
                continue

            if not last_actor:
                location = self._waypoints[self._route_index].transform.location
                max_dist = self._road_vehicle_dist
            else:
                location = last_actor.get_location()
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

                actor_state = 'road' if self._ego_state == 'road' else 'junction_entry'
                self._save_actor_info(actor, actor_state, None, leading_distance=self._get_random_number(13,17))
                self._junction_exit_sources[i]["last_actor"] = actor
                self._junction_exit_sources[i]["num_actors"] += 1

        # Remove the ones that have already spawned the actors
        for info in list(self._junction_exit_sources):
            if info["num_actors"] >= self._road_back_vehicles:
                self._junction_exit_sources.remove(info)

    ################################
    ## Behavior related functions ##
    ################################
    def _get_random_number(self, min_value, max_value):
        return min_value + (max_value - min_value) * self._rng.rand()

    def _save_actor_info(self, actor, state='road', ref_wp=None, leading_distance=None):
        """Saved the actor information"""
        if not leading_distance:
            leading_distance = self._get_random_number(1,6)

        self._background_actors.append(actor)
        self._actor_dict[actor] = {'state': state, 'ref_wp': ref_wp}

        self._tm.distance_to_leading_vehicle(actor, leading_distance)
        self._tm.auto_lane_change(actor, False)
        # self._tm.set_random_device_seed(int(self._rng.choice(self._random_seed_range)))

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

        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_points), spawn_points, True, False, 'background', safe_blueprint=True, tick=False)

        for actor in actors:
            self._save_actor_info(actor, 'road')

        spawn_points = []
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
            self._save_actor_info(actor, 'road', leading_distance=self._get_random_number(13,17))

    def _initialise_junction_entrances(self, enter_wps, route_enter_wp):
        """Initializes the actor sources to ensure the junction is always populated"""
        for wp in enter_wps:
            if wp.road_id == route_enter_wp.road_id:
                continue  # Ignore the road from which the route exits

            prev_wps = wp.previous(self._sources_dist)
            if len(prev_wps) == 0:
                continue  # Stop when there's no prev
            prev_wp = prev_wps[0]

            source_transform = carla.Transform(
                prev_wp.transform.location + carla.Location(z=0.2), prev_wp.transform.rotation)
            self._junction_entry_sources.append({"transform": source_transform, "actors":[]})

    def _initialise_junction_exits(self, exit_wps, route_exit_wp):
        """Spawns the road mode actors at the route's exit road and creates the pseudo actor sink dictionary"""
        exit_route_key = self._road_key(route_exit_wp)

        for wp in exit_wps:
            self._junction_exits[self._lane_key(wp)] = []
            if self._road_key(wp) != exit_route_key:
                continue  # Nothing to spawn

            source_transform = carla.Transform(
                wp.transform.location + carla.Location(z=0.2), wp.transform.rotation)
            self._junction_exit_sources.append(
                {"transform": source_transform, "active": False, "last_actor": None, "num_actors": 0})

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
                self._save_actor_info(actor, 'junction_exit', wp)
                self._junction_exits[self._lane_key(wp)].append(actor)

    def _manage_junction_entrances(self):
        """Checks the actor sources to see if new actors have to be created"""
        for i, info in enumerate(self._junction_entry_sources):
            transform = info["transform"]
            actors = info["actors"]

            # Cap the amount of alive actors
            if len(actors) >= self._max_actors_per_source:
                continue

            # Calculate distance to the last created actor
            if len(actors) == 0:
                distance = self._road_vehicle_dist + 1
            else:
                actor_location = CarlaDataProvider.get_location(actors[0])
                if actor_location is None:
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

                self._save_actor_info(actor, 'junction_entry', None)
                self._junction_entry_sources[i]["actors"].insert(0, actor)

    def _manage_road_radius(self):
        """Changed the radius dependent on the speed of the ego"""
        speed = CarlaDataProvider.get_velocity(self._ego_actor)
        self._min_radius = self._base_min_radius + self._radius_increase_ratio*speed
        self._max_radius = self._base_max_radius + self._radius_increase_ratio*speed

    def _get_next_scenario_time(self):
        """Gets the time for the next scenario"""
        a = self._scenario_min_time
        b = self._scenario_max_time - self._scenario_min_time
        self._next_scenario_time = a + b*(self._rng.rand())

    def _manage_road_scenario(self):
        """Randomly makes the vehicles in front of the ego suddenly break. Use a timer to avoid hving to
        reset it after junctions, making the scenario never trigger when multiple junctions are nearby"""
        self._next_scenario_time -= self._timestep  # TODO: This just works on sync mode
        # print(self._next_scenario_time)
        if self._is_scenario_active and self._next_scenario_time <= 0:
            self._is_scenario_active = False
            for actor in self._scenario_actors:
                self._tm.vehicle_percentage_speed_difference(actor, 0)
                actor.set_light_state(carla.VehicleLightState.NONE)
            if self._ego_state == 'road':
                self._get_next_scenario_time()
            self._scenario_actors = []

        elif not self._is_scenario_active and self._next_scenario_time <= 0:
            self._is_scenario_active = True
            self._scenario_actors = []
            ego_transform = self._waypoints[self._route_index].transform
            for actor in self._background_actors:
                location = CarlaDataProvider.get_location(actor)
                if not location:
                    continue
                if not self._is_actor_behind(location, ego_transform):  # Don't stop actors behind the ego
                    self._scenario_actors.append(actor)
                    self._tm.vehicle_percentage_speed_difference(actor, 100)
                    actor.set_light_state(carla.VehicleLightState.Brake)
            self._next_scenario_time = self._scenario_duration

    #############################
    ##     Actor functions     ##
    #############################

    def _is_actor_behind(self, location, ego_transform):
        """Checks if an actor is behind the ego. Uses the route transform"""
        ego_heading = ego_transform.get_forward_vector()
        ego_actor_vec = location - ego_transform.location
        if ego_heading.x * ego_actor_vec.x + ego_heading.y * ego_actor_vec.y < - 0.17:  # 100º
            return True
        return False

    def _manage_background_actor(self, actor, location):
        """Handles an actor depending on their previous state"""
        def set_actor_max_speed(distance):
            speed_red = (distance - self._min_radius) / (self._max_radius - self._min_radius) * 100
            speed_red = np.clip(speed_red, 0, 100)
            self._tm.vehicle_percentage_speed_difference(actor, speed_red)

        state, ref_wp = self._actor_dict[actor].values()
        route_transform = self._waypoints[self._route_index].transform

        if self._is_scenario_active and actor in self._scenario_actors:
            # Scenario actors are stopped, ignore them
            return

        # Calculate the distance to the reference point (by default, to the route)
        if ref_wp is None:
            ref_location = route_transform.location
        else:
            ref_location = ref_wp.transform.location
        distance = location.distance(ref_location)

        # For prev junction actors, remove them if far away. For the rest, control their speed
        if state == 'road':
            if actor in self._prev_actors:
                if distance > self._max_radius and self._is_actor_behind(location, route_transform):
                    self._destroy_actor(actor)
            elif not self._is_actor_behind(location, route_transform):
                set_actor_max_speed(distance)

        # Deactivate them when far from the junction
        elif state == 'junction_exit':
            if distance > self._junction_exit_dist:
                self._tm.vehicle_percentage_speed_difference(actor, 100)
                if self._junction_exit_key and self._road_key(ref_wp) == self._junction_exit_key:
                    self._actor_dict[actor]['state'] = 'junction_standby'
                else:
                    self._actor_dict[actor]['state'] = 'junction_inactive'

        # Destroy it if behind
        elif state == 'junction_inactive':
            if self._is_actor_behind(location, route_transform):
                self._destroy_actor(actor)

        # Monitor its exit and destroy an actor if needed
        elif state == 'junction_entry':
            actor_wp = self._map.get_waypoint(location)
            actor_lane_key = self._lane_key(actor_wp)
            if actor_lane_key in self._junction_exits:
                self._actor_dict[actor] = {'state': 'junction_exit', 'ref_wp': actor_wp}
                if len(self._junction_exits[actor_lane_key]) >= self._road_front_vehicles:
                    self._destroy_actor(self._junction_exits[actor_lane_key][-1])  # Remove last vehicle
                self._junction_exits[actor_lane_key].insert(0, actor)

        # Special state to avoid destroying the road behavior actors at the end of the junction
        elif state == 'junction_standby':
            pass

    def _destroy_actor(self, actor):
        """Destroy the actor and all its references"""
        if actor in self._background_actors:
            self._background_actors.remove(actor)

        if actor in self._prev_actors:
            self._prev_actors.remove(actor)

        if self._ego_state == 'junction':
            for lane_key in self._junction_exits:
                if actor in self._junction_exits[lane_key]:
                    self._junction_exits[lane_key].remove(actor)
                    break
            for i, info in enumerate(self._junction_entry_sources):
                if actor in info["actors"]:
                    self._junction_entry_sources[i]["actors"].remove(actor)
                    break

        for i, info in enumerate(self._junction_exit_sources):
            if actor == info["last_actor"]:
                self._junction_entry_sources[i]["last_actor"].remove(actor)
                break

        self._actor_dict.pop(actor, None)
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
