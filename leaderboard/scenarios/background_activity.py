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
import math
import py_trees
from collections import OrderedDict

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenarios.basic_scenario import BasicScenario


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
    def __init__(self, ego_actor, route, name="BackgroundBehavior"):
        """
        Setup class members
        """
        super(BackgroundBehavior, self).__init__(name)
        self._map = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self._tm.set_global_distance_to_leading_vehicle(5)

        # Actor variables
        self._ego_actor = ego_actor
        self._ego_state = 'road'
        self._background_actors = []
        self._actor_dict = OrderedDict()  # Dict with {actor: {'state', 'ref_wp'}}

        # Road variables (these are very much hardcoded so watch out when changing them)
        self._num_road_vehicles = 3  # Amount of vehicles in front of the ego
        self._road_vehicle_dist = 10  # Starting distance between spawned vehicles
        self._radius = 35  # Must be higher than nº_veh * veh_dist or the furthest vehicle will never activate 

        # Junction variables
        self._junction_dist = 34  # Junction detection distance. TODO: Higher breaks roundabout (fake junctions)
        self._junction_exits = OrderedDict()  # Keep track of the amount of actors per lane
        self._junction_sources = []  # Tracks actors spawned by the sources
        self._junction_ids = []
        self._road_exit_key = ""  # Key of the road through which the route exits the junction
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
        ego_location = self._ego_actor.get_transform().location
        ego_wp = self._map.get_waypoint(ego_location)
        self._initialise_road_behavior(self._get_lanes(ego_wp))
        self._create_junction_dict()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        # Get ego location and waypoint
        ego_location = CarlaDataProvider.get_location(self._ego_actor)
        if ego_location is None:
            return new_status
        self._update_ego_route_location(ego_location)

        # Monitor the background activity's state
        if self._ego_state == 'road':
            self._monitor_nearby_junctions()
        elif self._ego_state == 'junction':
            self._monitor_ego_junction_exit()
            self._manage_junction_entrances()

        # Manage the background actors
        for actor in self._background_actors:
            location = CarlaDataProvider.get_location(actor)
            if location is None:
                continue
            self._world.debug.draw_string(location, self._actor_dict[actor]['state'], False, carla.Color(0,0,0), 0.05)
            self._manage_background_actor(actor, location)

        return new_status

        # TODO 2: Check for roundabout and all junctions
        # TODO 3: Road behavior

    def terminate(self, new_status):
        """Destroy all actors"""
        for actor in self._background_actors:
            self._destroy_actor(actor)
        super(BackgroundBehavior, self).terminate(new_status)

    def _create_junction_dict(self):
        """Extracts the junction th ego vehicle will pass through"""
        # Get the junction data the route passes through, filter, group them and get their topology
        data = self._get_junctions_data()
        filtered_data, _ = self._filter_fake_junctions(data)
        multi_data = self._join_junctions(filtered_data)
        self._add_junctions_topology(multi_data)
        self._junctions_data = multi_data

    def _get_junctions_data(self):
        """Gets all the junctions the ego passes through"""
        junctions = []
        index = 0
        last_junction_id = None

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
                if not next_wp.is_junction or next_wp.get_junction().id != last_junction_id:
                    junctions[-1]['route_exit_index'] = i+1
            # Searching for a junction
            elif next_wp.is_junction:
                junctions.append({
                    'junctions': [next_wp.get_junction()],  # For multijunction later on
                    'route_enter_index': i,
                    'route_exit_index': None
                })
                last_junction_id = next_wp.get_junction().id

        return junctions

    def _filter_fake_junctions(self, junctions_data):
        """Filters fake junctions. At CARLA, these are junctions which have all lanes straight."""
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
        """Joins closeby junctions into one"""
        multi_junctions = [junctions[0]]

        for i in range(1, len(junctions)):
            junction_data = junctions[i]

            enter_accum_dist = self._route_accum_dist[junction_data['route_enter_index']]
            prev_exit_accum_dist = self._route_accum_dist[multi_junctions[-1]['route_exit_index']]

            # Group closeby junctions
            if enter_accum_dist - prev_exit_accum_dist < self._junction_dist:
                multi_junctions[-1]['junctions'].append(junction_data['junctions'][0])
                multi_junctions[-1]['route_exit_index'] = junction_data['route_exit_index']
                if junction_data['junctions'][0].id != multi_junctions[-1]['junctions'][-1].id:
                    # Due to topology, the same junction might be detected twice
                    multi_junctions[-1]['route_exit_index'] = junction_data['route_exit_index']

            # New junction entrance
            else:
                multi_junctions.append(junction_data)

        return multi_junctions

    def _add_junctions_topology(self, junctions):
        """Gets the entering and exiting lanes of a multijunction"""
        for junction_data in junctions:
            used_entering_lanes = []
            used_exiting_lanes = []
            entering_lane_wps = []
            exiting_lane_wps = []

            for junction in junction_data['junctions']:
                for enter_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):

                    while enter_wp.is_junction:
                        enter_wps = enter_wp.previous(0.5)
                        if len(enter_wps) == 0:
                            break  # Stop when there's no prev
                        enter_wp = enter_wps[0]
                    if enter_wp.is_junction:
                        continue  # Triggered by the loops break
                    if self._lane_key(enter_wp) not in used_entering_lanes:
                        used_entering_lanes.append(self._lane_key(enter_wp))
                        entering_lane_wps.append(enter_wp)

                    while exit_wp.is_junction:
                        exit_wps = exit_wp.next(0.5)
                        if len(exit_wps) == 0:
                            break  # Stop when there's no prev
                        exit_wp = exit_wps[0]
                    if exit_wp.is_junction:
                        continue  # Triggered by the loops break
                    if self._lane_key(exit_wp) not in used_exiting_lanes:
                        used_exiting_lanes.append(self._lane_key(exit_wp))
                        exiting_lane_wps.append(exit_wp)

            if len(junctions) > 1:
                # TODO: This fails if there are fake junction betweens two junctions
                exiting_lane_keys = [self._lane_key(wp) for wp in exiting_lane_wps]
                entering_lane_wps_ = [wp for wp in entering_lane_wps if self._lane_key(wp) not in exiting_lane_keys]

                entering_lane_keys = [self._lane_key(wp) for wp in entering_lane_wps]
                exiting_lane_wps_ = [wp for wp in exiting_lane_wps if self._lane_key(wp) not in entering_lane_keys]

            else:
                entering_lane_wps_ = entering_lane_wps
                exiting_lane_wps_ = exiting_lane_wps

            junction_data['enter_wps'] = entering_lane_wps_
            junction_data['exit_wps'] = exiting_lane_wps_

            # for enter_wp in entering_lane_wps_:
            #     self._world.debug.draw_point(enter_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(255,255,0), life_time=10000)
            # for exit_wp in exiting_lane_wps_:
            #     self._world.debug.draw_point(exit_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(0,255,255), life_time=10000)

        # for enter_wp in junctions[2]['enter_wps']:
        #     self._world.debug.draw_point(enter_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(255,255,0), life_time=10000)
        # for exit_wp in junctions[2]['exit_wps']:
        #     self._world.debug.draw_point(exit_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(0,255,255), life_time=10000)

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
        """Searches the exit lanes dictionary for a specific actor. Returns False if not found"""
        road = None
        for lane_key in self._junction_exits:
            if actor in self._junction_exits[lane_key]:
                road = int(lane_key.split('*')[0])
                break

        if road != ego_road:
            return False
        return True

    def _switch_to_junction_mode(self, junction):
        """Prepares the background activity to work in junction mode. This involves storing some
        junction values as well changing the state of most (if not all) actors"""
        self._ego_state = 'junction'

        self._junction_ids = [j.id for j in junction['junctions']]
        self._junction_exit_index = junction['route_exit_index']
        self._road_exit_key = self._road_key(self._waypoints[junction['route_exit_index']])

        for actor in self._background_actors:
            self._tm.vehicle_percentage_speed_difference(actor, 0)
            if actor in self._prev_actors:  # Remnants of previous junction
                self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': None}
            else:
                self._actor_dict[actor] = {'state': 'junction', 'ref_wp': None}

    def _switch_to_road_mode(self):
        """Prepares the background activity to work in road mode. This involves destroying all actors
        behidng the ego, remembering the ones that are in front and cleaning up junction variables"""
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
            self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': None}

        self._road_exit_key = ""
        self._junction_exit_index = None
        self._junction_exits.clear()
        self._junction_sources.clear()
        self._junction_ids.clear()
        self._junctions_data.pop(0)

    def _search_for_junction(self):
        """Search for closest frontal junction. When junction move ends, the jucntion data is
        removed, so it is only needed to check the first junction"""
        junction_data = self._junctions_data[0]
        ego_accum_dist = self._route_accum_dist[self._route_index]
        junction_accum_dist = self._route_accum_dist[junction_data['route_enter_index']]

        if junction_accum_dist - ego_accum_dist < self._junction_dist:  # Junctions closeby
            return junction_data
        return None

    def _monitor_nearby_junctions(self):
        """Monitors when the ego approaches a junction"""

        junction = self._search_for_junction()
        if not junction:
            return
        self._switch_to_junction_mode(junction)

        route_enter_wp = self._waypoints[junction['route_enter_index']]
        route_exit_wp = self._waypoints[junction['route_exit_index']]
        self._initialise_junction_entrances(junction['enter_wps'], route_enter_wp)
        self._initialise_junction_exits(junction['exit_wps'], route_exit_wp)

    def _monitor_ego_junction_exit(self):
        """Monitors when the ego enters and exits the junctions"""
        if self._route_index > self._junction_exit_index:
            self._switch_to_road_mode()

    ################################
    ## Behavior related functions ##
    ################################

    def _initialise_road_behavior(self, road_wps):
        """Intialises the road behavior, consisting on several vehicle in front of the ego"""
        spawn_points = []
        for wp in road_wps:
            next_wp = wp
            for _ in range(self._num_road_vehicles):
                next_wps = next_wp.next(self._road_vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                spawn_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=0.2),
                    next_wp.transform.rotation
                ))

        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_points), spawn_points, True, False, 'background', safe_blueprint=True, tick=False)

        for actor in actors:
            self._background_actors.append(actor)
            self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': None}
            self._tm.auto_lane_change(actor, False)

    def _initialise_junction_entrances(self, enter_wps, route_enter_wp):
        """Manage the entrances of the junction, making sure the junction is always populated. This function
        initialises the actor sources"""
        for wp in enter_wps:
            if wp.road_id == route_enter_wp.road_id:
                continue  # Ignore the road from which the route exits

            distance = 0
            step = 5
            prev_wp = wp
            while distance < self._sources_dist and not prev_wp.is_junction:
                prev_wps = prev_wp.previous(step)
                if len(prev_wps) == 0:
                    continue  # Stop when there's no prev
                prev_wp = prev_wps[0]
                distance += step

            source_transform = carla.Transform(
                prev_wp.transform.location + carla.Location(z=0.2),
                prev_wp.transform.rotation
            )
            self._junction_sources.append([source_transform, []])

    def _initialise_junction_exits(self, exit_wps, route_exit_wp):
        """Manage the junction exits, making sure they are never crowded. This function initialises the lanes
        topology and spawns the actors at the route road to continue its road behavior after the ego exits the
        junction"""
        exit_route_key = self._road_key(route_exit_wp) if route_exit_wp else ""

        for wp in exit_wps:
            self._junction_exits[self._lane_key(wp)] = []
            if self._road_key(wp) != exit_route_key:
                continue  # Nothing to spawn

            exiting_points = []
            next_wp = wp
            for _ in range(self._num_road_vehicles):
                next_wps = next_wp.next(self._road_vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                exiting_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=0.2),
                    next_wp.transform.rotation
                ))

            actors = CarlaDataProvider.request_new_batch_actors(
                'vehicle.*', len(exiting_points), exiting_points, True, False, 'background',
                safe_blueprint=True, tick=False
            )
            for actor in actors:
                self._junction_exits[self._lane_key(wp)].append(actor)
                self._background_actors.append(actor)
                self._tm.auto_lane_change(actor, False)
                self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': wp}

    def _manage_junction_entrances(self):
        """Checks the actor sources to see if new actors have to be created"""
        for i, [source_transform, actors] in enumerate(self._junction_sources):

            if len(actors) >= self._max_actors_per_source:
                continue

            # Calculate distance to the last created actor
            if len(actors) == 0:
                distance = self._road_vehicle_dist + 1
            else:
                actor_location = CarlaDataProvider.get_location(actors[0])
                if actor_location is None:
                    continue
                distance = source_transform.location.distance(actor_location)

            # Spawn a new actor if the last one is far enough
            if distance > self._road_vehicle_dist:
                actor = CarlaDataProvider.request_new_actor(
                    'vehicle.*', source_transform, rolename='background',
                    autopilot=True, random_location=False, safe_blueprint=True, tick=False
                )
                if actor is None:
                    continue

                self._background_actors.append(actor)
                self._actor_dict[actor] = {'state': 'junction', 'ref_wp': None}
                self._tm.auto_lane_change(actor, False)
                self._junction_sources[i][1].insert(0, actor)

    #############################
    ## Actor related functions ##
    #############################

    def _is_actor_behind(self, location, ego_transform):
        """Checks if an actor is behind the ego. Uses the route transform"""
        ego_heading = ego_transform.get_forward_vector()
        ego_actor_vec = location - ego_transform.location
        if ego_heading.x * ego_actor_vec.x + ego_heading.y * ego_actor_vec.y < 0:
            return True
        return False

    def _manage_background_actor(self, actor, location):
        """Handle the background actors depending on their previous state"""
        state, ref_wp = self._actor_dict[actor].values()
        route_transform = self._waypoints[self._route_index].transform

        # Calculate the distance to the reference point (by default, to the route)
        if ref_wp is None:
            ref_location = route_transform.location
        else:
            ref_location = ref_wp.transform.location
        distance = location.distance(ref_location)

        # Deactivate if far from the reference point
        if state == 'road_active':
            if distance > self._radius + 0.5:
                self._tm.vehicle_percentage_speed_difference(actor, 100)
                if self._road_exit_key and self._road_key(ref_wp) == self._road_exit_key:
                    self._actor_dict[actor]['state'] = 'road_standby'
                else:
                    self._actor_dict[actor]['state'] = 'road_inactive'

        # Activate if closeby or destroy it if behind
        elif state == 'road_inactive':
            if distance < self._radius - 0.5:
                self._tm.vehicle_percentage_speed_difference(actor, 0)
                self._actor_dict[actor]['state'] = 'road_active'
            elif self._is_actor_behind(location, route_transform):
                self._destroy_actor(actor)

            elif self._ego_state == 'junction':
                # Actors that reenter the junctions reset their behavior (needed for roundabouts)
                actor_junction = self._map.get_waypoint(location).get_junction()
                if actor_junction and actor_junction.id in self._junction_ids:
                    self._remove_junction_data(actor)
                    self._tm.vehicle_percentage_speed_difference(actor, 0)
                    self._actor_dict[actor]['state'] = 'junction'

        # Monitor its exit
        elif state == 'junction':
            actor_wp = self._map.get_waypoint(location)
            actor_lane_key = self._lane_key(actor_wp)
            if actor_lane_key in self._junction_exits:
                self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': actor_wp}
                if len(self._junction_exits[actor_lane_key]) >= self._num_road_vehicles:
                    self._destroy_actor(self._junction_exits[actor_lane_key][-1])  # Remove last vehicle
                self._junction_exits[actor_lane_key].insert(0, actor)

        # Special state to avoid destroying the road behavior actors at the end of the junction
        elif state == 'road_standby':
            pass

    def _remove_junction_data(self, actor):
        """Removed all junction information about an actor"""
        for lane_key in self._junction_exits:
            if actor in self._junction_exits[lane_key]:
                self._junction_exits[lane_key].remove(actor)
                break

        # This will make sources possibly create infinite actors
        # for i, [_, actors] in enumerate(self._junction_sources):
        #     if actor in actors:
        #         self._junction_sources[i][1].remove(actor)
        #         break

    def _destroy_actor(self, actor):
        """Destroy the actor and all its references"""
        if actor in self._background_actors:
            self._background_actors.remove(actor)
        if actor in self._prev_actors:
            self._prev_actors.remove(actor)
        if self._ego_state == 'junction':
            self._remove_junction_data(actor)
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
