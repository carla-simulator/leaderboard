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

        # # Spawn the vehicles in the opposite lane
        # oppo_dir_points = []
        # for wp in oppo_dir_wps:
        #     # Spawn one in front
        #     next_wp = wp
        #     trans = next_wp.transform
        #     oppo_dir_points.append(carla.Transform(trans.location + carla.Location(z=1), trans.rotation))

        #     # Spawn several behind
        #     prev_wp = wp
        #     for _ in range(self._num_front_vehicles):
        #         prev_wps = prev_wp.previous(self._start_distance)
        #         if len(prev_wps) == 0:
        #             break  # Stop when there's no next
        #         prev_wp = prev_wps[0]
        #         trans = prev_wp.transform
        #         oppo_dir_points.append(carla.Transform(trans.location + carla.Location(z=1), trans.rotation))

        # oppo_lane_actors = CarlaDataProvider.request_new_batch_actors(
        #     'vehicle.*', len(oppo_dir_points), oppo_dir_points, True, False, 'background', safe=True)
        # self._oppo_dir_vehicles = [actor for actor in oppo_lane_actors]


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
        self.remove_all_actors()


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

        # Actor variables
        self._ego_actor = ego_actor
        self._same_dir_actors = []
        self._junction_actors = []
        self._actor_configuration = []
        self._actor_dict = {}
        self._actor_dict[ego_actor] = ['road', None]

        # Route variables
        self._route = route
        self._route_length = len(self._route)
        self._waypoints, _ = zip(*self._route)
        self._route_index = 0
        self._route_buffer = 3

        # Radius variables
        self._num_front_vehicles = 3
        self._start_distance = 10
        self._radius = self._num_front_vehicles * self._start_distance + 0.5 * self._start_distance
        self._junction_radius = self._radius * 2

    def initialise(self):
        """Creates the background activity actors"""
        # Get the road topology
        ego_location = self._ego_actor.get_transform().location
        ego_wp = self._map.get_waypoint(ego_location)
        same_dir_wps, oppo_dir_wps = self._get_lanes(ego_wp)

        # Spawn the vehicles with the same direction as the ego
        same_dir_points = []
        for wp in same_dir_wps:
            # Spawn one behind
            prev_wp = wp.previous(self._start_distance)[0]
            trans = prev_wp.transform
            same_dir_points.append(carla.Transform(trans.location + carla.Location(z=1), trans.rotation))

            # Spawn several in front
            next_wp = wp
            for _ in range(self._num_front_vehicles):
                next_wps = next_wp.next(self._start_distance)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                trans = next_wp.transform
                same_dir_points.append(carla.Transform(trans.location + carla.Location(z=1), trans.rotation))

        same_lane_actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(same_dir_points), same_dir_points, True, False, 'background', safe=True)
        self._same_dir_actors = [actor for actor in same_lane_actors]

        for actor in self._same_dir_actors:
            self._tm.auto_lane_change(actor, False)
            self._actor_dict[actor] = ['active', None]

    def _get_lanes(self, waypoint):
        """Gets all the lanes of the road the ego starts at"""
        same_dir_wps = []
        oppo_dir_wps = []
        opposite_lane_wp = None
        same_dir_wps.append(waypoint)

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
                opposite_lane_wp = possible_left_wp
                break
            left_wp = possible_left_wp
            same_dir_wps.append(left_wp)

        if opposite_lane_wp is None:
            return same_dir_wps, oppo_dir_wps

        # Check opposite direction lanes
        right_wp = opposite_lane_wp
        oppo_dir_wps.append(right_wp)
        while True:
            possible_right_wp = right_wp.get_right_lane()
            if possible_right_wp is None or possible_right_wp.lane_type != carla.LaneType.Driving:
                break
            right_wp = possible_right_wp
            oppo_dir_wps.append(right_wp)

        return same_dir_wps, oppo_dir_wps

    def _populate_junction(self, wp):
        """ Populate the junction with vehicle"""
        next_wp = wp
        while True:
            next_wp = next_wp.next(5)[0]
            if next_wp.is_junction:
                break
        junction = next_wp.get_junction()
        used_roads = [wp.road_id]
        enter_wps = []
        for enter, exit in junction.get_waypoints(carla.LaneType.Driving):
            enter_wp = enter.previous(1)[0]
            if enter_wp.road_id not in used_roads:
                used_roads.append(enter_wp.road_id)
                enter_wps.append(enter_wp)

        junction_points = []
        for wp in enter_wps:
            # Spawn several behind
            prev_wp = wp
            for _ in range(self._num_front_vehicles):
                prev_wps = prev_wp.previous(self._start_distance)
                if len(prev_wps) == 0:
                    break  # Stop when there's no next
                prev_wp = prev_wps[0]
                trans = prev_wp.transform
                junction_points.append(carla.Transform(trans.location + carla.Location(z=1), trans.rotation))

        junction_actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(junction_points), junction_points, True, False, 'background', safe=True)
        for actor in junction_actors:
            self._junction_actors.append(actor)
            self._tm.auto_lane_change(actor, False)
            self._actor_dict[actor] = ['immune', None]

        for actor in self._same_dir_actors:
            self._actor_dict[actor][0] = 'immune'

    def _prepare_junction_exit(self):
        """Populates the exit of the junction to continue with traffic after exiting the junction"""
        found_junction = False
        for i in range(self._route_index, self._route_length):
            route_tran = self._waypoints[i]
            route_wp = self._map.get_waypoint(route_tran.location)
            if found_junction and not route_wp.is_junction:
                break
            elif not found_junction and route_wp.is_junction:
                found_junction = True

        same_dir_wps, _ = self._get_lanes(route_wp)

        # Spawn the vehicles with the same direction as the ego
        self._actor_configuration = []
        for wp in same_dir_wps:

            same_dir_points = []
            # Spawn several in front
            next_wps = wp.next(self._radius - self._start_distance)
            if len(next_wps) == 0:
                continue  # Stop when there's no next
            next_wp = next_wps[0]
            for _ in range(self._num_front_vehicles):
                next_wps = next_wp.next(self._start_distance)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                trans = next_wp.transform
                same_dir_points.append(carla.Transform(trans.location + carla.Location(z=1), trans.rotation))

            same_lane_actors = CarlaDataProvider.request_new_batch_actors(
                'vehicle.*', len(same_dir_points), same_dir_points, True, False, 'background', safe=True)

            self._actor_configuration.append([wp.road_id, wp.lane_id, same_lane_actors])
            for actor in same_lane_actors:
                self._same_dir_actors.append(actor)
                self._tm.auto_lane_change(actor, False)
                self._actor_dict[actor] = ['active', wp.transform.location]

    def _end_junction_state(self):
        """Ends the junction state"""
        for actor in self._actor_dict:
            if self._actor_dict[actor][0] == 'immune':
                self._actor_dict[actor][0] = 'active'

    def _destroy_actor(self, actor):
        """Reposition an actor back to its lane"""
        if actor in self._same_dir_actors:
            self._same_dir_actors.remove(actor)
        elif actor in self._junction_actors:
            self._junction_actors.remove(actor)
        self._actor_dict.pop(actor, None)
        actor.destroy()

    def _deactivate_actor(self, actor):
        """Stops the actor from moving"""
        print("Deactivating one actor")
        self._tm.vehicle_percentage_speed_difference(actor, 100)
        self._actor_dict[actor][0] = 'inactive'

    def _activate_actor(self, actor):
        """Makes the vehicle move again"""
        print("Activating one actor")
        self._tm.vehicle_percentage_speed_difference(actor, 0)
        self._actor_dict[actor][0] = 'active'

    def _check_exit_lane(self, actor, wp):
        """asdasd"""
        for i, info in enumerate(self._actor_configuration):
            print(wp.road_id)
            print(wp.lane_id)
            if wp.road_id == info[0] and wp.lane_id == info[1]:
                # Remove the frontest vehicle and add the new one
                removed_actor = self._actor_configuration[i][2].pop(-1)
                self._actor_configuration[i][2].insert(0, actor)
                self._destroy_actor(removed_actor)

    def _get_route_ego_location(self, location):
        """Returns the closest route location to the ego"""
        shortest_distance = float('inf')
        closest_index = -1

        for index in range(self._route_index, min(self._route_index + self._route_buffer, self._route_length)):
            ref_waypoint = self._waypoints[index]
            ref_location = ref_waypoint.location

            dist_to_route = ref_location.distance(location)
            if dist_to_route <= shortest_distance:
                closest_index = index
                shortest_distance = dist_to_route

        if closest_index != -1:
            self._route_index = closest_index
        return self._waypoints[self._route_index]

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        ego_location = CarlaDataProvider.get_location(self._ego_actor)
        if ego_location is None:
            return new_status
        route_transform = self._get_route_ego_location(ego_location)
        route_location = route_transform.location
        route_wp = self._map.get_waypoint(route_location)

        # Ego state handler
        if self._actor_dict[self._ego_actor][0] == 'road' and len(route_wp.next(self._junction_radius)) > 1:
            # Ego close to a junction
            self._populate_junction(route_wp)
            self._prepare_junction_exit()
            self._actor_dict[self._ego_actor][0] = 'entering_junction'
        elif self._actor_dict[self._ego_actor][0] == 'entering_junction' and route_wp.is_junction:
            # Ego entered the junction
            self._actor_dict[self._ego_actor][0] = 'at_junction'
        elif self._actor_dict[self._ego_actor][0] == 'at_junction' and not route_wp.is_junction:
            # Ego exited the junction
            self._end_junction_state()
            self._actor_dict[self._ego_actor][0] = 'road'
            for actor in self._actor_dict:
                self._actor_dict[actor][1] = None

        # Handle lane actors
        for actor in self._same_dir_actors:
            location = CarlaDataProvider.get_location(actor)
            if location is None:
                continue

            # Calculate the distance to a specific point
            ref_location = route_location if self._actor_dict[actor][1] is None else self._actor_dict[actor][1]
            distance_to_route = location.distance(ref_location)

            if self._actor_dict[actor][0] == 'active':
                if distance_to_route > self._radius + 1:
                    self._deactivate_actor(actor)
            elif self._actor_dict[actor][0] == 'inactive':
                if distance_to_route < self._radius - 1:
                    self._activate_actor(actor)
                else:
                    # Remove the vehicle that have turned away from the route at junctions
                    route_heading = route_transform.get_forward_vector()
                    route_actor_vec = location - route_location
                    if route_heading.x * route_actor_vec.x + route_heading.y * route_actor_vec.y < 0:
                        self._destroy_actor(actor)
            elif self._actor_dict[actor][0] == 'immune':
                # Actor has entered the junction
                actor_wp = self._map.get_waypoint(location)
                if actor_wp.is_junction:
                    self._actor_dict[actor][0] = 'protected'
                continue
            elif self._actor_dict[actor][0] == 'protected':
                actor_wp = self._map.get_waypoint(location)
                if not actor_wp.is_junction:
                    self._check_exit_lane(actor, actor_wp)
                    self._actor_dict[actor] = ['active', actor_wp.transform.location]
                continue

        # Handle junction actors
        for actor in self._junction_actors:
            location = CarlaDataProvider.get_location(actor)
            if location is None:
                continue

            # Calculate the distance to a specific point
            ref_location = route_location if self._actor_dict[actor][1] is None else self._actor_dict[actor][1]
            distance_to_route = location.distance(ref_location)

            if self._actor_dict[actor][0] == 'active':
                if distance_to_route > self._radius + 1:
                    self._deactivate_actor(actor)
            elif self._actor_dict[actor][0] == 'inactive':
                if distance_to_route < self._radius - 1:
                    self._activate_actor(actor)
                else:
                    # Remove the vehicle that have turned away from the route at junctions
                    route_heading = route_transform.get_forward_vector()
                    route_actor_vec = location - route_location
                    if route_heading.x * route_actor_vec.x + route_heading.y * route_actor_vec.y < 0:
                        self._destroy_actor(actor)
            elif self._actor_dict[actor][0] == 'immune':
                # Actor has entered the junction
                actor_wp = self._map.get_waypoint(location)
                if actor_wp.is_junction:
                    self._actor_dict[actor][0] = 'protected'
                continue
            elif self._actor_dict[actor][0] == 'protected':
                actor_wp = self._map.get_waypoint(location)
                if not actor_wp.is_junction:
                    self._check_exit_lane(actor, actor_wp)
                    self._actor_dict[actor] = ['active', actor_wp.transform.location]
                continue

        # for actor in self._oppo_dir_actors:
        #     location = CarlaDataProvider.get_location(actor)
        #     if location is None:
        #         continue

        #     distance_to_ego = math.pow(location.x - ego_location.x, 2) + \
        #                       math.pow(location.y - ego_location.y, 2)

        #     if distance_to_ego > self._active_radius * self._active_radius:
        #         print("Teleporting one car"))

        return new_status

    def terminate(self, new_status):
        """
        Destroy all actors
        """
        super(BackgroundBehavior, self).terminate(new_status)
