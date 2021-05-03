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

        # Actor variables
        self._ego_actor = ego_actor
        self._background_actors = []
        self._actor_dict = OrderedDict()  # Dict with {actor: [state, reference location, at junction]}
        self._actor_dict[self._ego_actor] = {'state': 'road', 'ref_loc': None, 'at_junction': False}

        # Road variables (these are very much hardcoded so watch out when changing them)
        self._num_front_vehicles = 3  # Amount of vehicles in front of the ego
        self._vehicle_dist = 10  # Starting distance between spawned vehicles
        self._radius = 35  # Must be higher than nº_veh * veh_dist or the furthest vehicle will never activate 

        # Junction variables
        self._junction_dict = OrderedDict()  # Dictionary to keep track of the amount of actors per lane
        self._junction_sources = []  # List of [source transform, last spawned actor]
        self._junction_dist = 60  # Distance at which junction behavior starts

        # Route variables
        self._route = route
        self._route_length = len(self._route)
        self._waypoints, _ = zip(*self._route)
        self._route_index = 0
        self._route_buffer = 3

    def _spawn_road_actors(self, lane_wps):
        """Spawns several vehicles in front of the ego and in all adjacent lanes with the same direction."""
        spawn_points = []
        for wp in lane_wps:
            next_wp = wp
            for _ in range(self._num_front_vehicles):
                next_wps = next_wp.next(self._vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                spawn_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=1),
                    next_wp.transform.rotation
                ))

        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_points), spawn_points, True, False, 'background', safe=True)

        # Add the actors the database and remove their lane changes
        for actor in actors:
            self._background_actors.append(actor)
            self._actor_dict[actor] = {'state': 'road_active', 'ref_loc': None, 'at_junction': False}
            self._tm.auto_lane_change(actor, False)

    def _get_junction_sources_transform(self, lane_wps):
        """For each lane entering the junction, returns a transform where the actor source will be placed"""
        for wp in lane_wps:
            prev_wps = wp.previous(self._radius)
            if len(prev_wps) == 0:
                break  # Stop when there's no prev
            prev_wp = prev_wps[0]
            source_transform = carla.Transform(
                prev_wp.transform.location + carla.Location(z=1),
                prev_wp.transform.rotation
            )

            self._junction_sources.append([source_transform, None])

    def _spawn_junction_exit_actors(self, lane_wps):
        """Spawns several actors at the road from which the ego will exit the junction,
        to continue its road behavior."""
        for wp in lane_wps:
            exiting_points = []

            # Spawn the actors
            next_wp = wp
            for _ in range(self._num_front_vehicles):
                next_wps = next_wp.next(self._vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                exiting_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=1),
                    next_wp.transform.rotation
                ))

            actors = CarlaDataProvider.request_new_batch_actors(
                'vehicle.*', len(exiting_points), exiting_points, True, False, 'background', safe=True)

            for actor in actors:
                self._junction_dict[self._get_lane_key(wp)].append(actor)
                self._background_actors.append(actor)
                self._tm.auto_lane_change(actor, False)
                self._actor_dict[actor] = {'state': 'road_active', 'ref_loc': wp.transform.location, 'at_junction': False}

    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        ego_location = self._ego_actor.get_transform().location
        ego_wp = self._map.get_waypoint(ego_location)
        same_dir_wps = self._get_lanes(ego_wp)
        self._spawn_road_actors(same_dir_wps)

    def _handle_ego_state(self, route_location):
        """Handle the trace of the status of the ego. This serves as a way to know whether or not
        junction related behaviors have to start being triggered (or ended)"""
        route_wp = self._map.get_waypoint(route_location)
        ego_state = self._actor_dict[self._ego_actor]['state']

        # At a road -> monitor if there is a junction nearby
        if ego_state == 'road' and len(route_wp.next(self._junction_dist)) > 1:
            self._actor_dict[self._ego_actor]['state'] = 'junction'

            # Set all created vehicles to junction mode
            for actor in self._background_actors:
                # TODO: Check if this breaks vehicles already inside junctions
                self._actor_dict[actor] = {'state': 'junction', 'ref_loc': None, 'at_junction': False}

            # Prepare the junction and the route exit lane
            self._prepare_junction(route_wp)

        # Junction found -> monitor the entrance to the junction
        elif ego_state == 'junction':
            at_junction = self._actor_dict[self._ego_actor]['at_junction']

            if not at_junction and route_wp.is_junction:
                # Entering the junction
                self._actor_dict[self._ego_actor]['at_junction'] = True

            elif at_junction and not route_wp.is_junction:
                # Exiting the junction
                self._actor_dict[self._ego_actor]['state'] = 'road'

                # Clear all junction variables
                self._junction_dict.clear()
                self._junction_sources.clear()

                # Set all actors to road mode
                for actor in self._background_actors:
                    # Note: this activates all inactive actors but they'll be deactivated again at the same step
                    self._actor_dict[actor] = {'state': 'road_active', 'ref_loc': None, 'at_junction': False}

    def _handle_junction_sources(self):
        """Checks the actor sources to see if new actors have to be created"""
        for i, [source_transform, last_actor] in enumerate(self._junction_sources):

            if last_actor is None:
                distance = self._vehicle_dist + 1
            else:
                # Calculate distacne to last created actor
                actor_location = CarlaDataProvider.get_location(last_actor)
                if actor_location is None:
                    print("Failed")
                    continue
                distance = source_transform.location.distance(actor_location)

            # Spawn a new actor if the last one is far enough
            if distance > self._vehicle_dist:
                actors = CarlaDataProvider.request_new_batch_actors(
                    'vehicle.*', 1, [source_transform], autopilot=True,
                    random_location=False, rolename='background', safe=True
                )
                if not actors:
                    continue
                actor = actors[0]

                self._background_actors.append(actor)
                self._actor_dict[actor] = {'state': 'junction', 'ref_loc': None, 'at_junction': False}
                self._tm.auto_lane_change(actor, False)
                self._junction_sources[i][1] = actor

    def _handle_background_actor(self, actor, location, ego_transform):
        """Handle the background actors depending on their previous state"""
        current_state, current_ref_point, at_junction = self._actor_dict[actor].values()

        # Calculate the distance to the reference point (by default, to the route)
        if current_ref_point is None:
            ref_location = ego_transform.location
        else:
            ref_location = current_ref_point
        distance = location.distance(ref_location)

        # Active -> Deactivate if far from the reference point
        if current_state == 'road_active':
            if distance > self._radius + 1:
                self._deactivate_actor(actor)

        # Inactive -> Activate if close to the reference point or remove if behind it
        elif current_state == 'road_inactive':
            # Remove the vehicle that are behind the ego
            ego_heading = ego_transform.get_forward_vector()
            ego_actor_vec = location - ego_transform.location
            if ego_heading.x * ego_actor_vec.x + ego_heading.y * ego_actor_vec.y < 0:
                self._destroy_actor(actor)
                return
            # Or activate them if they are in front and closeby
            if distance < self._radius - 1:
                self._activate_actor(actor)

        # Junction -> Monitor its exit
        elif current_state == 'junction':
            actor_wp = self._map.get_waypoint(location)
            if at_junction and not actor_wp.is_junction:
                # Exiting lane
                self._manage_exiting_lane(actor, actor_wp)
                self._actor_dict[actor] = {
                    'state': 'road_active', 'ref_loc': actor_wp.transform.location, 'at_junction': False
                }
            elif not at_junction and actor_wp.is_junction:
                # Entering lane
                self._actor_dict[actor]['at_junction'] = True

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

    def _get_lane_key(self, wp):
        """Returns a key corresponding to the lane"""
        return self._get_road_key(wp) + str(wp.lane_id)

    def _get_road_key(self, wp):
        """Returns a key corresponding to the road"""
        return str(wp.road_id) + "*"

    def _get_junction_topology(self, junction, route_wp):
        """Gets the entering and exiting lanes of a junction. Needed as junction.get_waypoints
        gets the wps inside the junction. Needs some filter as two entering / exiting points
        might originate from the same lane"""
        used_entering_lanes = [self._get_lane_key(route_wp)]
        used_exiting_lanes = []
        entering_lane_wps = []
        exiting_lane_wps = []

        for enter_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):

            # Entering waypoints. Move them out of the junction and save them
            enter_wps = enter_wp.previous(1)
            if len(enter_wps) == 0:
                continue  # Stop when there's no prev
            enter_wp = enter_wps[0]
            if self._get_lane_key(enter_wp) not in used_entering_lanes:
                # The road hasn't been used
                used_entering_lanes.append(self._get_lane_key(enter_wp))
                entering_lane_wps.append(enter_wp)

            # Exiting waypoints. Move them out of the junction and save them
            exit_wps = exit_wp.next(1)
            if len(exit_wps) == 0:
                continue  # Stop when there's no prev
            exit_wp = exit_wps[0]
            if self._get_lane_key(exit_wp) not in used_exiting_lanes:
                # The road hasn't been used
                used_exiting_lanes.append(self._get_lane_key(exit_wp))
                exiting_lane_wps.append(exit_wp)

        return entering_lane_wps, exiting_lane_wps

    def _prepare_junction(self, route_wp):
        """Populate the junction with vehicle. This is divided in handling two:
        - Getting the junction topology
        - Creating actor sources to infinitely populate the intersection at the entering lanes
        - Create vehicles at the road the route exits the junction"""

        # Get the junction. Possible TODO: Use the route?
        next_wp = route_wp
        while True:
            next_wp = next_wp.next(5)[0]
            if next_wp.is_junction:
                break
        junction = next_wp.get_junction()
        if junction is None:
            raise ValueError("Couldn't find the approaching junction")

        # Get the junction topology
        entering_junction_wps, exiting_junction_wps = self._get_junction_topology(junction, route_wp)

        # Spawn the actors
        # self._spawn_junction_actors(entering_junction_wps)
        self._get_junction_sources_transform(entering_junction_wps)

        # Get the first route waypoint outside the junction
        found_junction_exit = False
        for i in range(self._route_index, self._route_length - 1):
            route_wp = self._map.get_waypoint(self._waypoints[i].location)
            route_next_wp = self._map.get_waypoint(self._waypoints[i+1].location)
            if route_wp.is_junction and not route_next_wp.is_junction:
                found_junction_exit = True
                break

        if not found_junction_exit:
            return

        for wp in exiting_junction_wps:
            self._junction_dict[self._get_lane_key(wp)] = []

        # Filter the exiting lanes to those of the same road as the route
        route_road = self._get_road_key(route_next_wp)
        exiting_route_wps = [wp for wp in exiting_junction_wps if self._get_road_key(wp) == route_road]

        self._spawn_junction_exit_actors(exiting_route_wps)

        # TODO 1: Don't tick! check sensor frame
        # TODO 2: TM removes vehicles!
        # TODO 3: Check for roundabout, fake junctions and all junctions

    def _manage_exiting_lane(self, actor, wp):
        """When a vehicle exits the junction, check the state of that lane, removing
        a vehicle if needed to avoid crowding the exit"""
        lane_key = self._get_lane_key(wp)
        if lane_key in self._junction_dict:
            # Remove the frontest vehicle and add the new one at the end
            if len(self._junction_dict[lane_key]) >= self._num_front_vehicles:
                removed_actor = self._junction_dict[lane_key][-1]
                self._destroy_actor(removed_actor)
            self._junction_dict[lane_key].insert(0, actor)

    def _destroy_actor(self, actor):
        """Destroy the actor and all its references"""
        # Remove it from actor list
        if actor in self._background_actors:
            self._background_actors.remove(actor)
        
        # From the dictionary
        self._actor_dict.pop(actor, None)

        # And from the junction dictionary, if ego is at a junction
        if self._actor_dict[self._ego_actor]['state'] == 'junction':
            for lane_key in self._junction_dict:
                if actor in self._junction_dict[lane_key]:
                    self._junction_dict[lane_key].remove(actor)

        actor.destroy()

    def _deactivate_actor(self, actor):
        """Stops the actor from moving"""
        self._tm.vehicle_percentage_speed_difference(actor, 100)
        self._actor_dict[actor]['state'] = 'road_inactive'

    def _activate_actor(self, actor):
        """Makes the vehicle move again"""
        self._tm.vehicle_percentage_speed_difference(actor, 0)
        self._actor_dict[actor]['state'] = 'road_active'

    def _update_ego_route_location(self, location):
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

        # Get ego location and waypoint
        ego_location = CarlaDataProvider.get_location(self._ego_actor)
        if ego_location is None:
            return new_status
        route_transform = self._update_ego_route_location(ego_location)
        self._handle_ego_state(route_transform.location)
        if self._actor_dict[self._ego_actor]['state'] == 'junction':
            self._handle_junction_sources()

        for actor in self._background_actors:
            # Get its location and Handle its behavior
            location = CarlaDataProvider.get_location(actor)
            if location is None:
                continue
            self._world.debug.draw_string(location, self._actor_dict[actor]['state'], False, carla.Color(0,0,0), 0.05)
            self._handle_background_actor(actor, location, route_transform)

        return new_status

    def terminate(self, new_status):
        """
        Destroy all actors
        """
        super(BackgroundBehavior, self).terminate(new_status)
