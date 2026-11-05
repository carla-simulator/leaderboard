#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic scenario behaviors required to realize
complex, realistic scenarios such as "follow a leading vehicle", "lane change",
etc.

The atomic behaviors are implemented with py_trees.
"""

import copy
import math
import operator
import time

import numpy as np
from numpy import random
import py_trees
import networkx

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent
from agents.navigation.local_planner import RoadOption, LocalPlanner

from leaderboard.scenariomanager.simulation_data import SimulationData
from leaderboard.scenariomanager.timer import GameTime
from leaderboard.tools.scenario_helper import detect_lane_obstacle
from leaderboard.tools.scenario_helper import generate_target_waypoint_list_multilane


import leaderboard.tools as sr_tools
from leaderboard.utils.blackboard import get_blackboard, set_blackboard
from leaderboard.tools.actor_state import destroy_actor, get_ground_speed, has_actor_ticked
from leaderboard.tools.actor_factory import request_new_actor

EPSILON = 0.001


def calculate_distance(location, other_location, global_planner=None):
    """
    Method to calculate the distance between to locations

    Note: It uses the direct distance between the current location and the
          target location to estimate the time to arrival.
          To be accurate, it would have to use the distance along the
          (shortest) route between the two locations.
    """
    if global_planner:
        distance = 0

        # Get the route
        route = global_planner.trace_route(location, other_location)

        # Get the distance of the route
        for i in range(1, len(route)):
            curr_loc = route[i][0].transform.location
            prev_loc = route[i - 1][0].transform.location

            distance += curr_loc.distance(prev_loc)

        return distance

    return location.distance(other_location)


def get_actor_control(actor):
    """
    Method to return the type of control to the actor.
    """
    control = actor.get_control()
    actor_type = actor.type_id.split('.')[0]
    if not isinstance(actor, carla.Walker):
        control.steering = 0
    else:
        control.speed = 0

    return control, actor_type


class AtomicBehavior(py_trees.behaviour.Behaviour):

    """
    Base class for all atomic behaviors used to setup a scenario

    *All behaviors should use this class as parent*

    Important parameters:
    - name: Name of the atomic behavior
    """

    def __init__(self, name, actor=None):
        """
        Default init. Has to be called via super from derived class
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name
        self._actor = actor

    def setup(self, **kwargs):
        """
        Default setup
        """
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        """
        Initialise setup terminates WaypointFollowers
        Check whether WF for this actor is running and terminate all active WFs
        """
        if self._actor is not None:
            try:
                check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
                terminate_wf = copy.copy(check_attr(py_trees.blackboard.Blackboard()))
                set_blackboard(
                    "terminate_WF_actor_{}".format(self._actor.id), terminate_wf)
            except AttributeError:
                # It is ok to continue, if the Blackboard variable does not exist
                pass
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class UpdateAllActorControls(AtomicBehavior):

    """
    Atomic to update (run one control loop step) all actor controls.

    The behavior is always in RUNNING state.

    Args:
        name (string): Name of the behavior.
            Defaults to 'UpdateAllActorControls'.
    """

    def __init__(self, name="UpdateAllActorControls"):
        """
        Constructor
        """
        super().__init__(name)

    def update(self):
        """
        Execute one control loop step for all actor controls.

        returns:
            py_trees.common.Status.RUNNING
        """

        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        for actor_id in actor_dict:
            actor_dict[actor_id].run_step()

        return py_trees.common.Status.RUNNING


class ChangeActorWaypoints(AtomicBehavior):

    """
    Atomic to change the waypoints for an actor controller.

    The behavior is in RUNNING state until the last waypoint is reached, or if a
    second waypoint related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorLateralMotion
    - ChangeActorLaneOffset

    Args:
        actor (carla.Actor): Controlled actor.
        waypoints (List of (OSC position, OSC route option)): List of (Position, Route Option) as OpenScenario elements.
            position will be converted to Carla transforms, considering the corresponding
            route option (e.g. shortest, fastest)
        name (string): Name of the behavior.
            Defaults to 'ChangeActorWaypoints'.

    Attributes:
        _waypoints (List of (OSC position, OSC route option)): List of (Position, Route Option) as OpenScenario elements
        _start_time (float): Start time of the atomic [s].
            Defaults to None.

    '''Note: When using routing options such as fastest or shortest, it is advisable to run
             in synchronous mode
    """

    def __init__(self, actor, waypoints, times=None, name="ChangeActorWaypoints", is_osc1=True):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._waypoints = waypoints
        self._start_time = None
        self._times = times
        self._is_osc1 = is_osc1

        if len(self._waypoints) != len(self._times):
            raise ValueError("Both 'waypoints' and 'times' must have the same length")

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Set waypoint list for actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        if self._is_osc1:
            # Transforming OSC waypoints to Carla waypoints
            carla_route_elements = []
            for (osc_point, routing_option) in self._waypoints:
                carla_transforms = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(osc_point)
                carla_route_elements.append((carla_transforms, routing_option))
        else:
            carla_route_elements = []
            # mmap = SimulationData.get_map()
            for (point, routing_option) in self._waypoints:
                wp_transf = carla.Transform(location=carla.Location(point[0],point[1],point[2]))
                # carla_transforms = [mmap.get_waypoint(wp_transf.location)]
                carla_route_elements.append((wp_transf, routing_option))


        # Obtain final route, considering the routing option
        # At the moment everything besides "shortest" will use the CARLA GlobalPlanner
        grp = SimulationData.get_global_route_planner()
        route = []
        for i, element in enumerate(carla_route_elements):
            if element[1] == "shortest":
                route.append(element[0])
            else:
                if i == 0:
                    mmap = SimulationData.get_map()
                    ego_location = self._actor.get_location()
                    ego_waypoint = mmap.get_waypoint(ego_location)
                    try:
                        ego_next_wp = ego_waypoint.next(1)[0]
                    except IndexError:
                        ego_next_wp = ego_waypoint
                    waypoint = ego_next_wp.transform.location
                else:
                    waypoint = carla_route_elements[i - 1][0].location
                waypoint_next = element[0].location
                try:
                    interpolated_trace = grp.trace_route(waypoint, waypoint_next)
                except networkx.NetworkXNoPath:
                    print("WARNING: No route from {} to {} - Using direct path instead".format(waypoint, waypoint_next))
                    route.append(element[0])
                    continue
                for wp_tuple in interpolated_trace:
                    # The router sometimes produces points that go backward, or are almost identical
                    # We have to filter these, to avoid problems
                    if route and wp_tuple[0].transform.location.distance(route[-1].location) > 1.0:
                        new_heading_vec = wp_tuple[0].transform.location - route[-1].location
                        new_heading = np.arctan2(new_heading_vec.y, new_heading_vec.x)
                        if len(route) > 1:
                            last_heading_vec = route[-1].location - route[-2].location
                        else:
                            last_heading_vec = route[-1].location - ego_next_wp.transform.location
                        last_heading = np.arctan2(last_heading_vec.y, last_heading_vec.x)

                        heading_delta = math.fabs(new_heading - last_heading)
                        if math.fabs(heading_delta) < 0.5 or math.fabs(heading_delta) > 5.5:
                            route.append(wp_tuple[0].transform)
                    elif not route:
                        route.append(wp_tuple[0].transform)

        actor_dict[self._actor.id].update_waypoints(route, times=self._times, start_time=self._start_time)

        super().initialise()

    def update(self):
        """
        Check the actor's state along the waypoint route.

        returns:
            py_trees.common.Status.SUCCESS, if the final waypoint was reached, or
                                            if another ChangeActorWaypoints atomic for the same actor was triggered.
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.FAILURE, else.
        """
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        actor = actor_dict[self._actor.id]

        if actor.get_last_waypoint_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        if actor.check_reached_waypoint_goal():
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class ChangeActorLateralMotion(AtomicBehavior):

    """
    Atomic to change the waypoints for an actor controller.

    The behavior is in RUNNING state until the last waypoint is reached, or if a
    second waypoint related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorLateralMotion
    - ChangeActorLaneOffset

    If an impossible lane change is asked for (due to the lack of lateral lanes,
    next waypoints, continuous line, etc) the atomic will return a plan with the
    waypoints until such impossibility is found.

    Args:
        actor (carla.Actor): Controlled actor.
        direction (string): Lane change direction ('left' or 'right').
            Defaults to 'left'.
        distance_lane_change (float): Distance of the lance change [meters].
            Defaults to 25.
        distance_other_lane (float): Driven distance after the lange change [meters].
            Defaults to 100.
        name (string): Name of the behavior.
            Defaults to 'ChangeActorLateralMotion'.

    Attributes:
        _waypoints (List of carla.Transform): List of waypoints representing the lane change (CARLA transforms).
        _direction (string): Lane change direction ('left' or 'right').
        _distance_same_lane (float): Distance on the same lane before the lane change starts [meters]
            Constant to 5.
        _distance_other_lane (float): Max. distance on the target lane after the lance change [meters]
            Constant to 100.
        _distance_lane_change (float): Max. total distance of the lane change [meters].
        _pos_before_lane_change: carla.Location of the actor before the lane change.
            Defaults to None.
        _target_lane_id (int): Id of the target lane
            Defaults to None.
        _start_time (float): Start time of the atomic [s].
            Defaults to None.
    """

    def __init__(self, actor, direction='left', distance_lane_change=25, distance_other_lane=100,
                 lane_changes=1, name="ChangeActorLateralMotion"):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._waypoints = []
        self._direction = direction
        self._distance_same_lane = 5
        self._distance_other_lane = distance_other_lane
        self._distance_lane_change = distance_lane_change
        self._lane_changes = lane_changes
        self._pos_before_lane_change = None
        self._target_lane_id = None
        self._plan = None

        self._start_time = None

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Generate a waypoint list (route) which representes the lane change. Set
        this waypoint list for the actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        # get start position
        position_actor = SimulationData.get_map().get_waypoint(self._actor.get_location())

        # calculate plan with scenario_helper function
        self._plan, self._target_lane_id = generate_target_waypoint_list_multilane(
            position_actor, self._direction, self._distance_same_lane,
            self._distance_other_lane, self._distance_lane_change, check=False, lane_changes=self._lane_changes)

        if self._plan:
            for elem in self._plan:
                self._waypoints.append(elem[0].transform)

        actor_dict[self._actor.id].update_waypoints(self._waypoints, start_time=self._start_time)

        super().initialise()

    def update(self):
        """
        Check the actor's current state and if the lane change was completed

        returns:
            py_trees.common.Status.SUCCESS, if lane change was successful, or
                                            if another ChangeActorLateralMotion atomic for the same actor was triggered.
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.FAILURE, else.
        """

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if not self._plan:
            print("{} couldn't perform the expected lane change".format(self._actor))
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_waypoint_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        new_status = py_trees.common.Status.RUNNING

        current_position_actor = SimulationData.get_map().get_waypoint(self._actor.get_location())
        current_lane_id = current_position_actor.lane_id

        if current_lane_id == self._target_lane_id:
            # driving on new lane
            distance = current_position_actor.transform.location.distance(self._pos_before_lane_change)

            if distance > self._distance_other_lane:
                # long enough distance on new lane --> SUCCESS
                new_status = py_trees.common.Status.SUCCESS

                new_waypoints = []
                map_wp = current_position_actor
                while len(new_waypoints) < 200:
                    map_wps = map_wp.next(2.0)
                    if map_wps:
                        new_waypoints.append(map_wps[0].transform)
                        map_wp = map_wps[0]
                    else:
                        break

                actor_dict[self._actor.id].update_waypoints(new_waypoints, start_time=self._start_time)

        else:
            self._pos_before_lane_change = current_position_actor.transform.location

        return new_status


class ChangeActorLaneOffset(AtomicBehavior):

    """
    OpenSCENARIO atomic.
    Atomic to change the offset of the controller.

    The behavior is in RUNNING state until the offset os reached (if 'continuous' is set to False)
    or forever (if 'continuous' is True). This behavior will automatically stop if a second waypoint
    related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorLateralMotion
    - ChangeActorLaneOffset

    Args:
        actor (carla.Actor): Controlled actor.
        offset (float): Float determined the distance to the center of the lane. Positive distance imply a
            displacement to the right, while negative displacements are to the left.
        relative_actor (carla.Actor): The actor from which the offset is taken from. Defaults to None
        continuous (bool): If True, the behaviour never ends. If False, the behaviour ends when the lane
            offset is reached. Defaults to True.

    Attributes:
        _offset (float): lane offset.
        _relative_actor (carla.Actor): relative actor.
        _continuous (bool): stored the value of the 'continuous' argument.
        _start_time (float): Start time of the atomic [s].
            Defaults to None.
        _overwritten (bool): flag to check whether or not this behavior was overwritten by another. Helps
            to avoid the missinteraction between two ChangeActorLaneOffsets.
        _current_target_offset (float): stores the value of the offset when dealing with relative distances
        _map (carla.Map): instance of the CARLA map.
    """

    OFFSET_THRESHOLD = 0.1

    def __init__(self, actor, offset, relative_actor=None, continuous=True, name="ChangeActorWaypoints"):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._offset = offset
        self._relative_actor = relative_actor
        self._continuous = continuous
        self._start_time = None
        self._current_target_offset = 0

        self._overwritten = False
        self._map = SimulationData.get_map()

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Set offset for actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        actor_dict[self._actor.id].update_offset(self._offset, start_time=self._start_time)

        super().initialise()

    def update(self):
        """
        Check the actor's state along the waypoint route.

        returns:
            py_trees.common.Status.SUCCESS, if the lane offset was reached (and 'continuous' was False), or
                                            if another waypoint atomic for the same actor was triggered
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.RUNNING, else.
        """
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_lane_offset_command() != self._start_time:
            # Differentiate between lane offset and other lateral commands
            self._overwritten = True
            return py_trees.common.Status.SUCCESS

        if actor_dict[self._actor.id].get_last_waypoint_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        if self._relative_actor:
            # Calculate new offset
            relative_actor_loc = self._relative_actor.get_location()
            relative_center_wp = self._map.get_waypoint(relative_actor_loc)

            # Value
            relative_center_loc = relative_center_wp.transform.location
            relative_actor_offset = relative_actor_loc.distance(relative_center_loc)

            # Sign
            f_vec = relative_center_wp.transform.get_forward_vector()
            d_vec = relative_actor_loc - relative_center_loc
            cross = f_vec.x * d_vec.y - f_vec.y * d_vec.x

            if cross < 0:
                relative_actor_offset *= -1.0

            self._current_target_offset = relative_actor_offset + self._offset
            # Set the new offset
            actor_dict[self._actor.id].update_offset(self._current_target_offset)

        if not self._continuous:
            # Calculate new offset
            actor_loc = self._actor.get_location()
            center_wp = self._map.get_waypoint(actor_loc)

            # Value
            center_loc = center_wp.transform.location
            actor_offset = actor_loc.distance(center_loc)

            # Sign
            f_vec = center_wp.transform.get_forward_vector()
            d_vec = actor_loc - center_loc
            cross = f_vec.x * d_vec.y - f_vec.y * d_vec.x

            if cross < 0:
                actor_offset *= -1.0

            # Check if the offset has been reached
            if abs(actor_offset - self._current_target_offset) < self.OFFSET_THRESHOLD:
                return py_trees.common.Status.SUCCESS

        # TODO: As their is no way to check the distance to a specific lane, both checks will fail if the
        # actors are outside its 'route lane' or at an intersection

        new_status = py_trees.common.Status.RUNNING

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the offset is set back to zero
        """

        if not self._overwritten:
            try:
                check_actors = operator.attrgetter("ActorsWithController")
                actor_dict = check_actors(py_trees.blackboard.Blackboard())
            except AttributeError:
                pass

            if actor_dict and self._actor.id in actor_dict:
                actor_dict[self._actor.id].update_offset(0)

            self._overwritten = True

        super().terminate(new_status)


class KeepVelocity(AtomicBehavior):

    """
    This class contains an atomic behavior to keep the provided velocity.
    The controlled traffic participant will accelerate as fast as possible
    until reaching a given _target_velocity_, which is then maintained for
    as long as this behavior is active.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_velocity: The target velocity the actor should reach
    - forced_speed: Whether or not to forcefully set the actors speed. This will ony be active until a collision happens
    - duration[optional]: Duration in seconds of this behavior
    - distance[optional]: Maximum distance in meters covered by the actor during this behavior

    A termination can be enforced by providing distance or duration values.
    Alternatively, a parallel termination behavior has to be used.
    """

    def __init__(self, actor, target_velocity, force_speed=False,
                 duration=float("inf"), distance=float("inf"), name="KeepVelocity"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_velocity = target_velocity

        self._control, self._type = get_actor_control(actor)
        self._world = SimulationData.get_world()
        self._map = SimulationData.get_map()
        self._waypoint = self._map.get_waypoint(self._actor.get_location())

        self._forced_speed = force_speed
        self._duration = duration
        self._target_distance = distance
        self._distance = 0
        self._start_time = 0
        self._location = None

    def initialise(self):
        self._location = self._actor.get_location()
        self._start_time = GameTime.get_time()

        # In case of walkers, we have to extract the current heading
        if self._type == 'walker':
            self._control.speed = self._target_velocity
            self._control.direction = self._actor.get_transform().get_forward_vector()
        elif self._type == 'vehicle':
            self._control.hand_brake = False
        self._actor.apply_control(self._control)

        super().initialise()

    def update(self):
        """
        As long as the stop condition (duration or distance) is not violated, set a new vehicle control
        For vehicles: set throttle to throttle_value, as long as velocity is < target_velocity
        For walkers: simply apply the given self._control
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            if not self._forced_speed:
                if get_ground_speed(self._actor) < self._target_velocity:
                    self._control.throttle = 1.0
                else:
                    self._control.throttle = 0.0
                self._actor.apply_control(self._control)
            else:
                yaw = self._actor.get_transform().rotation.yaw * (math.pi / 180)
                self._actor.set_target_velocity(carla.Vector3D(
                    math.cos(yaw) * self._target_velocity, math.sin(yaw) * self._target_velocity, 0))

                # Add a throttle. Useless speed-wise, but makes the bicycle riders pedal.
                self._actor.apply_control(carla.VehicleControl(throttle=1.0))

        new_location = self._actor.get_location()
        self._distance += calculate_distance(self._location, new_location)
        self._location = new_location

        if self._distance > self._target_distance:
            new_status = py_trees.common.Status.SUCCESS

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        try:
            if self._type == 'vehicle':
                self._control.throttle = 0.0
            elif self._type == 'walker':
                self._control.speed = 0.0
            if self._actor is not None and self._actor.is_alive:
                self._actor.apply_control(self._control)
        except RuntimeError:
            pass
        super().terminate(new_status)


class ChangeAutoPilot(AtomicBehavior):

    """
    This class contains an atomic behavior to disable/enable the use of the autopilot.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - activate: True (=enable autopilot) or False (=disable autopilot)
    - lane_change: Traffic Manager parameter. True (=enable lane changes) or False (=disable lane changes)
    - distance_between_vehicles: Traffic Manager parameter
    - max_speed: Traffic Manager parameter. Max speed of the actor. This will only work for road segments
                 with the same speed limit as the first one

    The behavior terminates after changing the autopilot state
    """

    def __init__(self, actor, activate, parameters=None, name="ChangeAutoPilot"):
        """
        Setup parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._activate = activate
        self._tm = SimulationData.get_client().get_trafficmanager(
            SimulationData.get_traffic_manager_port())
        self._parameters = parameters

    def update(self):
        """
        De/activate autopilot
        """
        self._actor.set_autopilot(self._activate, SimulationData.get_traffic_manager_port())

        if self._parameters is not None:
            if "auto_lane_change" in self._parameters:
                lane_change = self._parameters["auto_lane_change"]
                self._tm.auto_lane_change(self._actor, lane_change)

            if "max_speed" in self._parameters:
                max_speed = self._parameters["max_speed"]
                max_road_speed = self._actor.get_speed_limit()
                if max_road_speed is not None:
                    percentage = (max_road_speed - max_speed) / max_road_speed * 100.0
                    self._tm.vehicle_percentage_speed_difference(self._actor, percentage)
                else:
                    print("ChangeAutopilot: Unable to find the vehicle's speed limit")

            if "distance_between_vehicles" in self._parameters:
                dist_vehicles = self._parameters["distance_between_vehicles"]
                self._tm.distance_to_leading_vehicle(self._actor, dist_vehicles)

            if "force_lane_change" in self._parameters:
                force_lane_change = self._parameters["force_lane_change"]
                self._tm.force_lane_change(self._actor, force_lane_change)

            if "ignore_vehicles_percentage" in self._parameters:
                ignore_vehicles = self._parameters["ignore_vehicles_percentage"]
                self._tm.ignore_vehicles_percentage(self._actor, ignore_vehicles)

        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class SyncArrivalWithAgent(AtomicBehavior):

    """
    Atomic to make two actors arrive at their corresponding places at the same time.
    This uses a controller and presuposes that the actor can reach its destination by following the lane.

    The behavior is in RUNNING state until the "main" actor has reached its destination.

    Args:
        actor (carla.Actor): Controlled actor.
        reference_actor (carla.Actor): Reference actor to sync up to.
        actor_target (carla.Transform): Endpoint of the actor after the behavior finishes.
        reference_target (carla.Transform): Endpoint of the reference_actor after the behavior finishes.
        delay (float): Time difference between the actors synchronization.
        end_dist (float): Minimum distance from the target to finish the behavior.
        name (string): Name of the behavior.
            Defaults to 'SyncArrivalWithAgent'.
    """

    def __init__(self, actor, reference_actor, actor_target, reference_target, end_dist=1,
                 name="SyncArrivalWithAgent"):
        """
        Setup required parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._actor = actor
        self._actor_target = actor_target
        self._reference_actor = reference_actor
        self._reference_target = reference_target
        self._end_dist = end_dist
        self._agent = None

    def initialise(self):
        """Initialises the agent"""
        self._agent = ConstantVelocityAgent(
            self._actor,
            map_inst=SimulationData.get_map(),
            grp_inst=SimulationData.get_global_route_planner())

    def update(self):
        """
        Dynamic control update for actor velocity to ensure that both actors reach their target
        positions at the same time.
        """
        new_status = py_trees.common.Status.RUNNING

        # Get the distance of the actor to its endpoint
        distance = calculate_distance(
            self._actor.get_location(), self._actor_target.location)

        # Check if the reference actor has passed its target
        if distance < self._end_dist:
            ref_dir = self._reference_target.get_forward_vector()
            ref_veh = self._reference_target.location - self._reference_actor.get_location()
            if ref_veh.dot(ref_dir) > 0:
                return py_trees.common.Status.SUCCESS

        # Get the time to arrival of the reference to its endpoint
        distance_reference = calculate_distance(
            self._reference_actor.get_location(), self._reference_target.location)

        velocity_reference = get_ground_speed(self._reference_actor)
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference
        else:
            time_reference = float('inf')

        # Get the required velocity of the actor
        desired_velocity = distance / time_reference

        self._agent.set_target_speed(3.6 * desired_velocity)
        self._actor.apply_control(self._agent.run_step())

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """Destroy the collision sensor of the agent"""
        if self._agent:
            self._agent.destroy_sensor()
        return super().terminate(new_status)


class CutIn(AtomicBehavior):

    """
    Atomic to make an actor lane change using a Python API agent, cutting in front of another one

    The behavior creates a lane change path and is in RUNNING state until the "main" actor has finsihes it.

    Args:
        actor (carla.Actor): Controlled actor.
        reference_actor (carla.Actor): Reference actor to cut in.
        direction (string): Side from which the cut in happens. Either 'left' or 'right'.
        speed_perc (float): Percentage of the reference actor speed on which the cut in is performed.
        same_lane_time (float): Amount of time spent at the same lane before cutting in.
        other_lane_time (float): Amount of time spent at the other lane after cutting in.
        change_time (float): Amount of time spent changing into the other
        name (string): Name of the behavior.
            Defaults to 'CutIn'.
    """

    def __init__(self, actor, reference_actor, direction, speed_perc=100,
                 same_lane_time=0, other_lane_time=0, change_time=2,
                 name="CutIn"):
        """
        Setup required parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._reference_actor = reference_actor
        self._direction = direction
        self._speed_perc = speed_perc
        self._same_lane_time = same_lane_time
        self._other_lane_time = other_lane_time
        self._change_time = change_time

        self._map = SimulationData.get_map()
        self._grp = SimulationData.get_global_route_planner()
        self._agent = None

    def initialise(self):
        """Initialises the agent"""
        speed = get_ground_speed(self._reference_actor)
        self._agent = BasicAgent(
            self._actor,
            3.6 * speed * self._speed_perc / 100,
            map_inst=SimulationData.get_map(),
            grp_inst=SimulationData.get_global_route_planner())
        self._agent.lane_change(self._direction, self._same_lane_time, self._other_lane_time, self._change_time)

    def update(self):
        """
        Dynamic control update for actor velocity to ensure that both actors reach their target
        positions at the same time.
        """
        new_status = py_trees.common.Status.RUNNING
        if self._agent.done():
            return py_trees.common.Status.SUCCESS

        self._actor.apply_control(self._agent.run_step())

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class AddNoiseToVehicle(AtomicBehavior):

    """
    This class contains an atomic jitter behavior.
    To add noise to steer as well as throttle of the vehicle.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - steer_value: Applied steering noise in [0,1]
    - throttle_value: Applied throttle noise in [0,1]

    The behavior terminates after setting the new actor controls
    """

    def __init__(self, actor, steer_value, throttle_value, name="Jittering"):
        """
        Setup actor , maximum steer value and throttle value
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._steer_value = steer_value
        self._throttle_value = throttle_value

    def update(self):
        """
        Set steer to steer_value and throttle to throttle_value until reaching full stop
        """
        self._control = self._actor.get_control()
        self._control.steer = self._steer_value
        self._control.throttle = self._throttle_value
        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)

        return new_status


class AddNoiseToRouteEgo(AtomicBehavior):

    """
    This class contains an atomic jitter behavior.
    To add noise to steer as well as throttle of the vehicle.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - steer_value: Applied steering noise in [0,1]
    - throttle_value: Applied throttle noise in [0,1]

    The behavior terminates after setting the new actor controls
    """

    def __init__(self, actor, throttle_mean, throttle_std, steer_mean, steer_std, name="AddNoiseToVehicle"):
        """
        Setup actor , maximum steer value and throttle value
        """
        super().__init__(name, actor)
        self._throttle_mean = throttle_mean
        self._throttle_std = throttle_std
        self._steer_mean = steer_mean
        self._steer_std = steer_std

        self._rng = SimulationData.get_rng()

    def update(self):
        """
        Set steer to steer_value and throttle to throttle_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING

        control = get_blackboard("AV_control")
        if not control:
            print("WARNING: Couldn't add noise to the ego because the control couldn't be found")
            return new_status

        throttle_noise = random.normal(self._throttle_mean, self._throttle_std)
        control.throttle = max(-1, min(1, control.throttle + throttle_noise))

        steer_noise = random.normal(self._steer_mean, self._steer_std)
        control.steer = max(0, min(1, control.steer + steer_noise))

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(control)

        return new_status


class BasicAgentBehavior(AtomicBehavior):
    """
    This class contains an atomic behavior, which uses the
    basic_agent from CARLA to control the actor until
    reaching a target location.
    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_location: Is the desired target location (carla.location),
                       the actor should move to
    The behavior terminates after reaching the target_location (within 2 meters)
    """

    def __init__(
        self,
        actor,
        target_location=None,
        plan=None,
        target_speed=20,
        opt_dict=None,
        name="BasicAgentBehavior",
    ):
        """
        Setup actor and maximum steer value
        """
        super().__init__(name, actor)
        self._map = SimulationData.get_map()
        self._target_location = target_location
        self._target_speed = target_speed
        self._plan = plan

        self._opt_dict = opt_dict if opt_dict else {}
        self._control = carla.VehicleControl()
        self._agent = None

        if self._target_location and self._plan:
            raise ValueError("Choose either a destination or a plan, but not both")

    def initialise(self):
        """Initialises the agent"""
        self._agent = BasicAgent(self._actor, self._target_speed, opt_dict=self._opt_dict,
            map_inst=SimulationData.get_map(), grp_inst=SimulationData.get_global_route_planner())
        if self._plan:
            self._agent.set_global_plan(self._plan)
        elif self._target_location:
            init_wp = self._map.get_waypoint(self._actor.get_location())
            end_wp = self._map.get_waypoint(self._target_location)
            self._plan = self._agent.trace_route(init_wp, end_wp)
            self._agent.set_global_plan(self._plan)

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self._agent.done():
            new_status = py_trees.common.Status.SUCCESS
        self._control = self._agent.run_step()
        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """Resets the control"""
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        super().terminate(new_status)


class ConstantVelocityAgentBehavior(AtomicBehavior):

    """
    This class contains an atomic behavior, which uses the
    constant_velocity_agent from CARLA to control the actor until
    reaching a target location.
    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_location: Is the desired target location (carla.location),
                       the actor should move to
    - plan: List of [carla.Waypoint, RoadOption] to pass to the controller
    - target_speed: Desired speed of the actor
    The behavior terminates after reaching the target_location (within 2 meters)
    """

    def __init__(self, actor, target_location, target_speed=None,
                 opt_dict=None, name="ConstantVelocityAgentBehavior"):
        """
        Set up actor and local planner
        """
        super().__init__(name, actor)
        self._target_speed = target_speed
        self._map = SimulationData.get_map()
        self._target_location = target_location
        self._opt_dict = opt_dict if opt_dict else {}
        self._control = carla.VehicleControl()
        self._agent = None
        self._plan = None

        self._map = SimulationData.get_map()
        self._grp = SimulationData.get_global_route_planner()

    def initialise(self):
        """Initialises the agent"""
        self._agent = ConstantVelocityAgent(
            self._actor, self._target_speed * 3.6, opt_dict=self._opt_dict,
            map_inst=SimulationData.get_map(), grp_inst=SimulationData.get_global_route_planner())
        self._plan = self._agent.trace_route(
            self._map.get_waypoint(self._actor.get_location()),
            self._map.get_waypoint(self._target_location))
        self._agent.set_global_plan(self._plan)

    def update(self):
        """Moves the actor and waits for it to finish the plan"""
        new_status = py_trees.common.Status.RUNNING

        if self._agent.done():
            new_status = py_trees.common.Status.SUCCESS

        self._control = self._agent.run_step()
        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """Resets the control"""
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        if self._agent:
            self._agent.destroy_sensor()
        super().terminate(new_status)

class AdaptiveConstantVelocityAgentBehavior(AtomicBehavior):

    """
    This class contains an atomic behavior, which uses the
    constant_velocity_agent from CARLA to control the actor until
    reaching a target location.
    Important parameters:
    - actor: CARLA actor to execute the behavior.
    - reference_actor: Reference CARLA actor to get target speed.
    - speed_increment: Float value (m/s). 
                       How much the actor will be faster then the reference_actor.
    - target_location: Is the desired target location (carla.location),
                       the actor should move to. 
                       If it's None, the actor will follow the lane and never stop.
    - plan: List of [carla.Waypoint, RoadOption] to pass to the controller.
    The behavior terminates after reaching the target_location (within 2 meters)
    """

    def __init__(self, actor, reference_actor, target_location=None, speed_increment=10,
                 opt_dict=None, name="AdaptiveConstantVelocityAgentBehavior"):
        """
        Set up actor and local planner
        """
        super().__init__(name, actor)
        self._speed_increment = speed_increment
        self._reference_actor = reference_actor
        self._target_location = target_location
        self._opt_dict = opt_dict if opt_dict else {}
        self._control = carla.VehicleControl()
        self._agent = None
        self._plan = None

        self._map = SimulationData.get_map()
        self._grp = SimulationData.get_global_route_planner()

    def initialise(self):
        """Initialises the agent"""
        # Get target speed
        target_speed = get_ground_speed(self._reference_actor) + self._speed_increment * 3.6

        self._agent = ConstantVelocityAgent(self._actor, target_speed, opt_dict=self._opt_dict,
                                            map_inst=self._map, grp_inst=self._grp)

        if self._target_location is not None:
            self._plan = self._agent.trace_route(
                self._map.get_waypoint(self._actor.get_location()),
                self._map.get_waypoint(self._target_location))
            self._agent.set_global_plan(self._plan)

    def update(self):
        """Moves the actor and waits for it to finish the plan"""
        new_status = py_trees.common.Status.RUNNING
        target_speed = get_ground_speed(self._reference_actor) + self._speed_increment * 3.6
        self._agent.set_target_speed(target_speed)

        if self._agent.done():
            new_status = py_trees.common.Status.SUCCESS

        self._control = self._agent.run_step()
        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """Resets the control"""
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        if self._agent:
            self._agent.destroy_sensor()
        super().terminate(new_status)

class Idle(AtomicBehavior):

    """
    This class contains an idle behavior scenario

    Important parameters:
    - duration[optional]: Duration in seconds of this behavior

    A termination can be enforced by providing a duration value.
    Alternatively, a parallel termination behavior has to be used.
    """

    def __init__(self, duration=float("inf"), name="Idle"):
        """
        Setup actor
        """
        super().__init__(name)
        self._duration = duration
        self._start_time = 0
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._start_time = GameTime.get_time()
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        return new_status

class WaitForever(AtomicBehavior):

    """
    This class contains a behavior that just waits forever.
    Useful to stop some behavior sequences from stopping unwated parts of the behavior tree

    Alternatively, a parallel termination behavior has to be used to stop it.
    """

    def __init__(self, name="WaitForever"):
        """
        Setup actor
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        wait forever
        """
        return py_trees.common.Status.RUNNING


class WaypointFollower(AtomicBehavior):

    """
    This is an atomic behavior to follow waypoints while maintaining a given speed.
    If no plan is provided, the actor will follow its foward waypoints indefinetely.
    Otherwise, the behavior will end with SUCCESS upon reaching the end of the plan.
    If no target velocity is provided, the actor continues with its current velocity.

    Args:
        actor (carla.Actor):  CARLA actor to execute the behavior.
        target_speed (float, optional): Desired speed of the actor in m/s. Defaults to None.
        plan ([carla.Location] or [(carla.Waypoint, carla.agent.navigation.local_planner)], optional):
            Waypoint plan the actor should follow. Defaults to None.
        blackboard_queue_name (str, optional):
            Blackboard variable name, if additional actors should be created on-the-fly. Defaults to None.
        avoid_collision (bool, optional):
            Enable/Disable(=default) collision avoidance for vehicles/bikes. Defaults to False.
        name (str, optional): Name of the behavior. Defaults to "FollowWaypoints".

    Attributes:
        actor (carla.Actor):  CARLA actor to execute the behavior.
        name (str, optional): Name of the behavior.
        _target_speed (float, optional): Desired speed of the actor in m/s. Defaults to None.
        _plan ([carla.Location] or [(carla.Waypoint, carla.agent.navigation.local_planner)]):
            Waypoint plan the actor should follow. Defaults to None.
        _blackboard_queue_name (str):
            Blackboard variable name, if additional actors should be created on-the-fly. Defaults to None.
        _avoid_collision (bool): Enable/Disable(=default) collision avoidance for vehicles/bikes. Defaults to False.
        _actor_dict: Dictonary of all actors, and their corresponding plans (e.g. {actor: plan}).
        _local_planner_dict: Dictonary of all actors, and their corresponding local planners.
            Either "Walker" for pedestrians, or a carla.agent.navigation.LocalPlanner for other actors.
        _args_lateral_dict: Parameters for the PID of the used carla.agent.navigation.LocalPlanner.
        _unique_id: Unique ID of the behavior based on timestamp in nanoseconds.

    Note:
        OpenScenario:
        The WaypointFollower atomic must be called with an individual name if multiple consecutive WFs.
        Blackboard variables with lists are used for consecutive WaypointFollower behaviors.
        Termination of active WaypointFollowers in initialise of AtomicBehavior because any
        following behavior must terminate the WaypointFollower.
    """

    def __init__(self, actor, target_speed=None, plan=None, blackboard_queue_name=None,
                 avoid_collision=False, name="FollowWaypoints"):
        """
        Set up actor and local planner
        """
        super().__init__(name, actor)
        self._actor_dict = {}
        self._actor_dict[actor] = None
        self._target_speed = target_speed
        self._local_planner_dict = {}
        self._local_planner_dict[actor] = None
        self._plan = plan
        self._blackboard_queue_name = blackboard_queue_name
        if blackboard_queue_name is not None:
            self._queue = get_blackboard(blackboard_queue_name)
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}
        self._avoid_collision = avoid_collision
        self._start_time = None
        self._unique_id = 0

    def initialise(self):
        """
        Delayed one-time initialization

        Checks if another WaypointFollower behavior is already running for this actor.
        If this is the case, a termination signal is sent to the running behavior.
        """
        super().initialise()
        self._start_time = GameTime.get_time()
        self._unique_id = int(round(time.time() * 1e9))
        try:
            # check whether WF for this actor is already running and add new WF to running_WF list
            check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
            running = check_attr(py_trees.blackboard.Blackboard())
            active_wf = copy.copy(running)
            active_wf.append(self._unique_id)
            set_blackboard(
                "running_WF_actor_{}".format(self._actor.id), active_wf)
        except AttributeError:
            # no WF is active for this actor
            set_blackboard("terminate_WF_actor_{}".format(self._actor.id), [])
            set_blackboard(
                "running_WF_actor_{}".format(self._actor.id), [self._unique_id])

        for actor in self._actor_dict:
            self._apply_local_planner(actor)
        return True

    def _apply_local_planner(self, actor):
        """
        Convert the plan into locations for walkers (pedestrians), or to a waypoint list for other actors.
        For non-walkers, activate the carla.agent.navigation.LocalPlanner module.
        """
        if self._target_speed is None:
            self._target_speed = get_ground_speed(actor)
        else:
            self._target_speed = self._target_speed

        if isinstance(actor, carla.Walker):
            self._local_planner_dict[actor] = "Walker"
            if self._plan is not None:
                if isinstance(self._plan[0], carla.Location):
                    self._actor_dict[actor] = self._plan
                else:
                    self._actor_dict[actor] = [element[0].transform.location for element in self._plan]
        else:
            local_planner = LocalPlanner(  # pylint: disable=undefined-variable
                actor, opt_dict={
                    'target_speed': self._target_speed * 3.6,
                    'lateral_control_dict': self._args_lateral_dict,
                    'max_throttle': 1.0})

            if self._plan is not None:
                if isinstance(self._plan[0], carla.Location):
                    plan = []
                    for location in self._plan:
                        waypoint = SimulationData.get_map().get_waypoint(location,
                                                                            project_to_road=True,
                                                                            lane_type=carla.LaneType.Any)
                        plan.append((waypoint, RoadOption.LANEFOLLOW))
                    local_planner.set_global_plan(plan)
                else:
                    local_planner.set_global_plan(self._plan)

            self._local_planner_dict[actor] = local_planner
            self._actor_dict[actor] = self._plan

    def update(self):
        """
        Compute next control step for the given waypoint plan, obtain and apply control to actor
        """
        new_status = py_trees.common.Status.RUNNING

        check_term = operator.attrgetter("terminate_WF_actor_{}".format(self._actor.id))
        terminate_wf = check_term(py_trees.blackboard.Blackboard())

        check_run = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
        active_wf = check_run(py_trees.blackboard.Blackboard())

        # Termination of WF if the WFs unique_id is listed in terminate_wf
        # only one WF should be active, therefore all previous WF have to be terminated
        if self._unique_id in terminate_wf:
            terminate_wf.remove(self._unique_id)
            if self._unique_id in active_wf:
                active_wf.remove(self._unique_id)

            set_blackboard(
                "terminate_WF_actor_{}".format(self._actor.id), terminate_wf)
            set_blackboard(
                "running_WF_actor_{}".format(self._actor.id), active_wf)
            new_status = py_trees.common.Status.SUCCESS
            return new_status

        if self._blackboard_queue_name is not None:
            while not self._queue.empty():
                actor = self._queue.get()
                if actor is not None and actor not in self._actor_dict:
                    self._apply_local_planner(actor)

        success = True
        for actor in self._local_planner_dict:
            local_planner = self._local_planner_dict[actor] if actor else None
            if actor is not None and actor.is_alive and local_planner is not None:
                # Check if the actor is a vehicle/bike
                if not isinstance(actor, carla.Walker):
                    control = local_planner.run_step(debug=False)
                    if self._avoid_collision and detect_lane_obstacle(actor):
                        control.throttle = 0.0
                        control.brake = 1.0
                    actor.apply_control(control)
                    # Check if the actor reached the end of the plan
                    # @TODO replace access to private _waypoints_queue with public getter
                    if local_planner._waypoints_queue:  # pylint: disable=protected-access
                        success = False
                # If the actor is a pedestrian, we have to use the WalkerAIController
                # The walker is sent to the next waypoint in its plan
                else:
                    actor_location = actor.get_location()
                    success = False
                    if self._actor_dict[actor]:
                        location = self._actor_dict[actor][0]
                        direction = location - actor_location
                        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
                        control = actor.get_control()
                        control.speed = self._target_speed
                        control.direction = direction / direction_norm
                        actor.apply_control(control)
                        if direction_norm < 1.0:
                            self._actor_dict[actor] = self._actor_dict[actor][1:]
                            if self._actor_dict[actor] is None:
                                success = True
                    else:
                        control = actor.get_control()
                        control.speed = self._target_speed
                        control.direction = actor.get_transform().rotation.get_forward_vector()
                        actor.apply_control(control)

        if success:
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior,
        the controls should be set back to 0.
        """
        for actor in self._local_planner_dict:
            if actor is not None and actor.is_alive:
                control, _ = get_actor_control(actor)
                actor.apply_control(control)
                local_planner = self._local_planner_dict[actor]
                if local_planner is not None and local_planner != "Walker":
                    local_planner.reset_vehicle()
                    local_planner = None

        self._local_planner_dict = {}
        self._actor_dict = {}
        super().terminate(new_status)


class HandBrakeVehicle(AtomicBehavior):

    """
    This class contains an atomic hand brake behavior.
    To set the hand brake value of the vehicle.

    Important parameters:
    - vehicle: CARLA actor to execute the behavior
    - hand_brake_value to be applied in [0,1]

    The behavior terminates after setting the hand brake value
    """

    def __init__(self, vehicle, hand_brake_value, name="Braking"):
        """
        Setup vehicle control and brake value
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle
        self._control, self._type = get_actor_control(vehicle)
        self._hand_brake_value = hand_brake_value

    def update(self):
        """
        Set handbrake
        """
        new_status = py_trees.common.Status.SUCCESS
        if self._type == 'vehicle':
            self._control.hand_brake = self._hand_brake_value
            self._vehicle.apply_control(self._control)
        else:
            self._hand_brake_value = None
            self.logger.debug("%s.update()[%s->%s]" %
                              (self.__class__.__name__, self.status, new_status))
            self._vehicle.apply_control(self._control)

        return new_status


class ActorDestroy(AtomicBehavior):

    """
    This class contains an actor destroy behavior.
    Given an actor this behavior will delete it.

    Important parameters:
    - actor: CARLA actor to be deleted

    The behavior terminates after removing the actor
    """

    def __init__(self, actor, name="ActorDestroy"):
        """
        Setup actor
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self._actor:
            destroy_actor(self._actor)
            self._actor = None
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class ActorTransformSetter(AtomicBehavior):

    """
    This class contains an atomic behavior to set the transform
    of an actor.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - transform: New target transform (position + orientation) of the actor
    - physics [optional]: Change the physics of the actors to true / false. To not change the physics, use None.

    The behavior terminates when actor is set to the new actor transform (closer than 1 meter)

    NOTE:
    It is very important to ensure that the actor location is spawned to the new transform because of the
    appearence of a rare runtime processing error. WaypointFollower with LocalPlanner,
    might fail if new_status is set to success before the actor is really positioned at the new transform.
    Therefore: calculate_distance(actor, transform) < 1 meter
    """

    def __init__(self, actor, transform, physics=True, name="ActorTransformSetter"):
        """
        Init
        """
        super().__init__(name, actor)
        self._transform = transform
        self._physics = physics
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        if self._actor.is_alive:
            self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_transform(self._transform)
        super().initialise()

    def update(self):
        """
        Transform actor
        """
        new_status = py_trees.common.Status.RUNNING

        if not self._actor.is_alive:
            new_status = py_trees.common.Status.FAILURE

        if calculate_distance(self._actor.get_location(), self._transform.location) < 1.0:
            if self._physics is not None:
                self._actor.set_simulate_physics(self._physics)
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class BatchActorTransformSetter(AtomicBehavior):

    """
    This class contains an atomic behavior to set the transform
    of an actor.

    Important parameters:
    - actor_transform_list: list [carla.Actor, carla.Transform]
    - physics [optional]: Change the physics of the actors to true / false. To not change the physics, use None.

    The behavior terminates immediately
    """

    def __init__(self, actor_transform_list, physics=True, name="BatchActorTransformSetter"):
        """
        Init
        """
        super().__init__(name)
        self._actor_transform_list = actor_transform_list
        self._physics = physics
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        Transform actor
        """

        for actor, transform in self._actor_transform_list:
            actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            actor.set_transform(transform)
            if self._physics is not None:
                actor.set_simulate_physics(self._physics)

        return py_trees.common.Status.SUCCESS


class ActorFlow(AtomicBehavior):
    """
    Behavior that indefinitely creates actors at a location,
    controls them until another location, and then destroys them.
    Therefore, a parallel termination behavior has to be used.

    Important parameters:
    - source_transform (carla.Transform): Transform at which actors will be spawned
    - sink_location (carla.Location): Location at which actors will be deleted
    - spawn_distance: Distance between spawned actors
    - sink_distance: Actors closer to the sink than this distance will be deleted
    - actors_speed: Speed of the actors part of the flow [m/s]
    - initial_actors: Populates all the flow trajectory at the start
    """

    def __init__(self, source_wp, sink_wp, spawn_dist_interval, sink_dist=2,
                 actor_speed=20 / 3.6, initial_actors=False, initial_junction=False, name="ActorFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._rng = SimulationData.get_rng()
        self._world = SimulationData.get_world()
        self._tm = SimulationData.get_client().get_trafficmanager(SimulationData.get_traffic_manager_port())

        self._collision_bp = self._world.get_blueprint_library().find('sensor.other.collision')
        self._is_constant_velocity_active = True

        self._source_wp = source_wp
        self._sink_wp = sink_wp

        self._sink_location = self._sink_wp.transform.location
        self._source_transform = self._source_wp.transform
        self._source_location = self._source_transform.location

        self._sink_dist = sink_dist
        self._speed = actor_speed
        self._initial_actors = initial_actors
        self._initial_junction = initial_junction

        self._min_spawn_dist = spawn_dist_interval[0]
        self._max_spawn_dist = spawn_dist_interval[1]
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        self._attribute_filter = {'base_type': 'car', 'has_lights': True, 'special_type': ''}

        self._actor_list = []
        self._collision_sensor_list = []

        self._terminated = False

    def initialise(self):
        if self._initial_actors:
            grp = SimulationData.get_global_route_planner()
            plan = grp.trace_route(self._source_location, self._sink_location)

            ref_loc = plan[0][0].transform.location
            for wp, _ in plan:
                if wp.is_junction and not self._initial_junction:
                    continue  # Spawning at junctions might break the path, so don't
                if wp.transform.location.distance(ref_loc) < self._spawn_dist:
                    continue
                self._spawn_actor(wp.transform)
                ref_loc = wp.transform.location
                self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

    def _spawn_actor(self, transform):
        actor = request_new_actor('vehicle.*', transform, rolename='scenario',
            attribute_filter=self._attribute_filter, tick=False
        )
        if actor is None:
            return py_trees.common.Status.RUNNING

        actor.set_autopilot(True, SimulationData.get_traffic_manager_port())
        self._tm.set_path(actor, [self._sink_location])
        self._tm.auto_lane_change(actor, False)
        self._tm.set_desired_speed(actor, 3.6 * self._speed)
        self._tm.update_vehicle_lights(actor, True)

        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        sensor = None
        if self._is_constant_velocity_active:
            self._tm.ignore_vehicles_percentage(actor, 100)
            actor.enable_constant_velocity(carla.Vector3D(self._speed, 0, 0))  # For when physics are active

            sensor = self._world.spawn_actor(self._collision_bp, carla.Transform(), attach_to=actor)
            sensor.listen(lambda _: self.stop_constant_velocity())

        self._tm.ignore_lights_percentage(actor, 100)
        self._tm.ignore_signs_percentage(actor, 100)
        self._collision_sensor_list.append(sensor)
        self._actor_list.append(actor)
        return None

    def update(self):
        """Controls the created actors and creaes / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor, sensor in zip(list(self._actor_list), list(self._collision_sensor_list)):
            if not has_actor_ticked(actor):
                continue
            location = actor.get_location()
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
                self._collision_sensor_list.remove(sensor)
                destroy_actor(actor)
                self._actor_list.remove(actor)

        # Spawn new actors if needed
        if len(self._actor_list) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = self._actor_list[-1].get_location()
            distance = self._source_location.distance(actor_location) if actor_location else 0

        if distance > self._spawn_dist:
            self._spawn_actor(self._source_transform)

        return py_trees.common.Status.RUNNING

    def stop_constant_velocity(self):
        """Stops the constant velocity behavior"""
        self._is_constant_velocity_active = False
        for actor in self._actor_list:
            actor.disable_constant_velocity()
            self._tm.ignore_vehicles_percentage(actor, 0)

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for sensor in self._collision_sensor_list:
            if sensor is None:
                continue
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass  # Actor was already destroyed

        for actor in self._actor_list:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, SimulationData.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                destroy_actor(actor)
            except RuntimeError:
                pass  # Actor was already destroyed


class OppositeActorFlow(AtomicBehavior):
    """
    Similar to ActorFlow, but this is meant as an actor flow in the opposite direction.
    As such, some configurations are different and for clarity, another behavior has been created

    Important parameters:
    - source_wp (carla.Waypoint): Waypoint at which actors will be spawned
    - sink_wp (carla.Waypoint): Waypoint at which actors will be deleted
    - spawn_dist_interval: Distance interval between spawned actors
    - sink_dist: Actors closer to the sink than this distance will be deleted
    - actors_speed: Speed of the actors part of the flow [m/s]
    - offset: offset from the center lane of the actors
    """

    def __init__(self, reference_wp, reference_actor, spawn_dist_interval,
                 time_distance=1.5, base_distance=30, sink_dist=2, name="OppositeActorFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._rng = SimulationData.get_rng()
        self._world = SimulationData.get_world()
        self._tm = SimulationData.get_client().get_trafficmanager(SimulationData.get_traffic_manager_port())

        self._reference_wp = reference_wp
        self._reference_actor = reference_actor
        self._time_distance = time_distance
        self._base_distance = base_distance
        self._min_spawn_dist = spawn_dist_interval[0]
        self._max_spawn_dist = spawn_dist_interval[1]
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        self._sink_dist = sink_dist

        self._attribute_filter = {'base_type': 'car', 'has_lights': True, 'special_type': ''}

        # Opposite direction needs earlier vehicle detection
        self._opt_dict = {'base_vehicle_threshold': 10, 'detection_speed_ratio': 1.6}

        self._actor_list = []
        self._grp = SimulationData.get_global_route_planner()
        self._map = SimulationData.get_map()

        self._terminated = False

        # initialise attributes
        self._speed = None # Km / h
        self._flow_distance = -1.0

        self._sink_wp = None
        self._source_wp = None

        self._source_transform = None
        self._source_location = None
        self._sink_location = None

        self._route = None

    def _move_waypoint_forward(self, wp, distance):
        """Moves forward a certain distance, stopping at junctions"""
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1

        return next_wp

    def _move_waypoint_backwards(self, wp, distance):
        """Moves backwards a certain distance, stopping at junctions"""
        dist = 0
        prev_wp = wp
        while dist < distance:
            prev_wps = prev_wp.previous(1)
            if prev_wps[0].is_junction:
                break
            prev_wp = prev_wps[0]
            dist += 1

        return prev_wp

    def initialise(self):
        """Get the actor flow source and sink, depending on the reference actor speed"""
        self._speed = self._reference_actor.get_speed_limit() # Km / h
        self._flow_distance = self._time_distance * self._speed + self._base_distance

        self._sink_wp = self._move_waypoint_forward(self._reference_wp, self._flow_distance)
        self._source_wp = self._move_waypoint_backwards(self._reference_wp, self._flow_distance)

        self._source_transform = self._source_wp.transform
        self._source_location = self._source_transform.location
        self._sink_location = self._sink_wp.transform.location

        self._route = self._grp.trace_route(self._source_location, self._sink_location)

        return super().initialise()

    def _spawn_actor(self):
        actor = request_new_actor('vehicle.*', self._source_transform, rolename='scenario',
            attribute_filter=self._attribute_filter, tick=False
        )
        if actor is None:
            return py_trees.common.Status.RUNNING

        controller = BasicAgent(actor, self._speed, self._opt_dict, self._map, self._grp)
        controller.set_global_plan(self._route)
        self._actor_list.append([actor, controller])

        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)
        return None

    def update(self):
        """Controls the created actors and creates / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor_data in list(self._actor_list):
            actor, controller = actor_data
            if not has_actor_ticked(actor):
                continue
            location = actor.get_location()
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                destroy_actor(actor)
                self._actor_list.remove(actor_data)
            else:
                actor.apply_control(controller.run_step())

        # Spawn new actors if needed
        if len(self._actor_list) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = self._actor_list[-1][0].get_location()
            distance = self._source_location.distance(actor_location) if actor_location else 0
        if distance > self._spawn_dist:
            self._spawn_actor()

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for actor, _ in self._actor_list:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, SimulationData.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                destroy_actor(actor)
            except RuntimeError:
                pass  # Actor was already destroyed


class InvadingActorFlow(AtomicBehavior):
    """
    Similar to ActorFlow, but this is meant as an actor flow in the opposite direction that invades the lane.
    As such, some configurations are different and for clarity, another behavior has been created

    Important parameters:
    - source_wp (carla.Waypoint): Waypoint at which actors will be spawned
    - sink_wp (carla.Waypoint): Waypoint at which actors will be deleted
    - spawn_dist_interval: Distance interval between spawned actors
    - sink_dist: Actors closer to the sink than this distance will be deleted
    - actors_speed: Speed of the actors part of the flow [m/s]
    - offset: offset from the center lane of the actors
    """

    def __init__(self, source_wp, sink_wp, reference_actor, spawn_dist,
                 sink_dist=2, offset=0, name="OppositeActorFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._world = SimulationData.get_world()
        self._tm = SimulationData.get_client().get_trafficmanager(SimulationData.get_traffic_manager_port())

        self._reference_actor = reference_actor

        self._source_wp  = source_wp
        self._source_transform = self._source_wp.transform
        self._source_location = self._source_transform.location

        self._sink_wp = sink_wp
        self._sink_location = self._sink_wp.transform.location

        self._spawn_dist = spawn_dist

        self._sink_dist = sink_dist

        self._attribute_filter = {'base_type': 'car', 'has_lights': True, 'special_type': ''}

        self._actor_list = []

        # Opposite direction needs earlier vehicle detection
        self._opt_dict = {'base_vehicle_threshold': 10, 'detection_speed_ratio': 2, 'distance_ratio': 0.2}
        self._opt_dict['offset'] = offset

        self._grp = SimulationData.get_global_route_planner()
        self._map = SimulationData.get_map()

        self._terminated = False
        # Initialise attributes
        self._speed = None
        self._route = None

    def initialise(self):
        """Get the actor flow source and sink, depending on the reference actor speed"""
        self._speed = self._reference_actor.get_speed_limit()  # Km / h
        self._route = self._grp.trace_route(self._source_location, self._sink_location)
        return super().initialise()

    def _spawn_actor(self):
        actor = request_new_actor('vehicle.*', self._source_transform, rolename='scenario',
            attribute_filter=self._attribute_filter, tick=False
        )
        if actor is None:
            return py_trees.common.Status.RUNNING

        controller = BasicAgent(actor, self._speed, self._opt_dict, self._map, self._grp)
        controller.set_global_plan(self._route)
        self._actor_list.append([actor, controller])
        return None

    def update(self):
        """Controls the created actors and creates / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor_data in list(self._actor_list):
            actor, controller = actor_data
            if not has_actor_ticked(actor):
                continue
            location = actor.get_location()
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                destroy_actor(actor)
                self._actor_list.remove(actor_data)
            else:
                actor.apply_control(controller.run_step())

        # Spawn new actors if needed
        if len(self._actor_list) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = self._actor_list[-1][0].get_location()
            distance = self._source_location.distance(actor_location) if actor_location else 0
        if distance > self._spawn_dist:
            self._spawn_actor()

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for actor, _ in self._actor_list:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, SimulationData.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                destroy_actor(actor)
            except RuntimeError:
                pass  # Actor was already destroyed


class BicycleFlow(AtomicBehavior):
    """
    Behavior that indefinitely creates bicycles at a location,
    controls them until another location, and then destroys them.
    Therefore, a parallel termination behavior has to be used.

    Important parameters:
    - plan (list(carla.Waypoint)): plan used by the bicycles.
    - spawn_distance_interval (list(float, float)): Distance between spawned actors
    - sink_distance (float): Actors at this distance from the sink will be deleted
    - actors_speed (float): Speed of the actors part of the flow [m/s]
    - initial_actors (bool): Boolean to initialy populate all the flow with bicycles
    """

    def __init__(self, plan, spawn_dist_interval, sink_dist=2,
                 actor_speed=20 / 3.6, initial_actors=False, name="BicycleFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._rng = SimulationData.get_rng()

        self._plan = plan
        self._sink_dist = sink_dist
        self._speed = actor_speed

        self._source_transform = self._plan[0][0].transform
        self._source_location = self._source_transform.location
        self._sink_location = self._plan[-1][0].transform.location

        self._min_spawn_dist = spawn_dist_interval[0]
        self._max_spawn_dist = spawn_dist_interval[1]
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        self._initial_actors = initial_actors

        self._opt_dict = {"ignore_traffic_lights": True, "ignore_vehicles": True}

        self._actor_data = []
        self._grp = SimulationData.get_global_route_planner()

        self._terminated = False

    def initialise(self):
        if self._initial_actors:
            ref_loc = self._plan[0][0].transform.location
            for wp, _ in self._plan:
                if wp.is_junction:
                    continue  # Spawning at junctions might break the path, so don't
                if wp.transform.location.distance(ref_loc) < self._spawn_dist:
                    continue
                self._spawn_actor(wp.transform)
                ref_loc = wp.transform.location
                self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

    def _spawn_actor(self, transform):
        """Spawn the actor"""
        # Initial actors don't want all the plan. Remove the points behind them
        plan = self._plan
        actor_loc = transform.location
        while len(plan) > 0:
            wp, _ = plan[0]
            loc = wp.transform.location
            actor_heading = transform.get_forward_vector()
            actor_wp_vec = loc - actor_loc
            if actor_heading.dot(actor_wp_vec) < 0 or loc.distance(actor_loc) < 10:
                plan.pop(0)
            else:
                break

        if not plan:
            return

        actor = request_new_actor('vehicle.*', transform, rolename='scenario no lights',
            attribute_filter={'base_type': 'bicycle'}, tick=False
        )
        if actor is None:
            return

        controller = BasicAgent(actor, 3.6 * self._speed, opt_dict=self._opt_dict,
            map_inst=SimulationData.get_map(), grp_inst=SimulationData.get_global_route_planner())
        controller.set_global_plan(plan)

        initial_vec = plan[0][0].transform.get_forward_vector()
        actor.set_target_velocity(self._speed * initial_vec)
        actor.apply_control(carla.VehicleControl(throttle=1, gear=1, manual_gear_shift=True))

        self._actor_data.append([actor, controller])
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

    def update(self):
        """Controls the created actors and creaes / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor_data in list(self._actor_data):
            actor, controller = actor_data
            if not has_actor_ticked(actor):
                continue
            location = actor.get_location()
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                destroy_actor(actor)
                self._actor_data.remove(actor_data)
            else:
                actor.apply_control(controller.run_step())

        # Spawn new actors if needed
        if len(self._actor_data) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = self._actor_data[-1][0].get_location()
            distance = self._source_location.distance(actor_location)

        if distance > self._spawn_dist:
            self._spawn_actor(self._source_transform)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for actor, _ in self._actor_data:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, SimulationData.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                destroy_actor(actor)
            except RuntimeError:
                pass  # Actor was already destroyed


class OpenVehicleDoor(AtomicBehavior):

    """
    Implementation for a behavior that will open the door of a vehicle,
    then close it after a while.

    Important parameters:
    - actor: Type of CARLA actors to be spawned
    - vehicle_door: The specific door that will be opened
    - duration: Duration of the open door
    """

    def __init__(self, actor, vehicle_door, name="OpenVehicleDoor"):
        """
        Setup class members
        """
        super().__init__(name, actor)
        self._vehicle_door = vehicle_door
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._actor.open_door(self._vehicle_door)
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.SUCCESS
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class TrafficLightFreezer(AtomicBehavior):
    """
    Behavior that freezes a group of traffic lights for a specific amount of time,
    returning them to their original state after ending it

    Important parameters:
    - traffic_lights_dict dict{carla.TrafficLight: carla.TrafficLightState}
    - timeout: Amount of time the traffic lights are frozen
    """

    def __init__(self, traffic_lights_dict, duration=10000, name="TrafficLightFreezer"):
        """Setup class members"""
        super().__init__(name)
        self._traffic_lights_dict = traffic_lights_dict
        self._duration = duration
        self._previous_traffic_light_info = {}
        self._start_time = None

    def initialise(self):
        """
        Sets the traffic lights to the desired states and remembers their previous one.
        These should technically be frozen, but that freezes all tls in the town
        """
        self._start_time = GameTime.get_time()
        for tl in self._traffic_lights_dict:
            elapsed_time = tl.get_elapsed_time()
            self._previous_traffic_light_info[tl] = {
                'state': tl.get_state(),
                'green_time': tl.get_green_time(),
                'red_time': tl.get_red_time(),
                'yellow_time': tl.get_yellow_time()
            }
            tl.set_state(self._traffic_lights_dict[tl])
            tl.set_green_time(self._duration + elapsed_time)
            tl.set_red_time(self._duration + elapsed_time)
            tl.set_yellow_time(self._duration + elapsed_time)

    def update(self):
        """Waits until the adequate time has passed"""
        if GameTime.get_time() - self._start_time > self._duration:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Reset all traffic lights back to their previous states"""
        if self._previous_traffic_light_info:
            for tl in self._traffic_lights_dict:
                tl.set_state(self._previous_traffic_light_info[tl]['state'])
                tl.set_green_time(self._previous_traffic_light_info[tl]['green_time'])
                tl.set_red_time(self._previous_traffic_light_info[tl]['red_time'])
                tl.set_yellow_time(self._previous_traffic_light_info[tl]['yellow_time'])


class ScenarioTriggerer(AtomicBehavior):

    """
    Handles the triggering of the scenarios that are part of a route.

    Initializes a list of blackboard variables to False, and only sets them to True when
    the ego vehicle is very close to the scenarios
    """

    WINDOWS_SIZE = 5

    def __init__(self, actor, route, blackboard_list, distance, debug=False, name="ScenarioTriggerer"):
        """
        Setup class members
        """
        super().__init__(name)
        self._world = SimulationData.get_world()
        self._map = SimulationData.get_map()
        self._debug = debug

        self._actor = actor
        self._route = route
        self._distance = distance
        self._blackboard_list = blackboard_list
        self._triggered_scenarios = []  # List of already done scenarios

        self._current_index = 0
        self._route_length = len(self._route)
        self._waypoints, _ = zip(*self._route)

    def add_blackboard(self, blackboard):
        """
        Adds new blackboards to the list. Used by the runtime initialization of scenarios
        """
        self._blackboard_list.append(blackboard)

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not has_actor_ticked(self._actor):
            return new_status
        location = self._actor.get_location()

        lower_bound = self._current_index
        upper_bound = min(self._current_index + self.WINDOWS_SIZE + 1, self._route_length)

        shortest_distance = float('inf')
        closest_index = -1

        for index in range(lower_bound, upper_bound):
            ref_waypoint = self._waypoints[index]
            ref_location = ref_waypoint.location

            dist_to_route = ref_location.distance(location)
            if dist_to_route <= shortest_distance:
                closest_index = index
                shortest_distance = dist_to_route

        if closest_index == -1 or shortest_distance == float('inf'):
            return new_status

        # Update the ego position at the route
        self._current_index = closest_index

        route_location = self._waypoints[closest_index].location

        # Check which scenarios can be triggered
        blackboard = py_trees.blackboard.Blackboard()
        for black_var_name, scen_location, scen_name in self._blackboard_list:

            # Close enough
            scen_distance = route_location.distance(scen_location)
            condition1 = bool(scen_distance < self._distance)

            # Not being currently done
            value = blackboard.get(black_var_name)
            condition2 = bool(not value)

            # Already done, if needed
            condition3 = bool(black_var_name not in self._triggered_scenarios)

            if condition1 and condition2 and condition3:
                _ = blackboard.set(black_var_name, True)
                self._triggered_scenarios.append(black_var_name)

                set_blackboard("latest_scenario", scen_name)

                if self._debug:
                    self._world.debug.draw_point(
                        scen_location + carla.Location(z=4),
                        size=0.5,
                        life_time=0.5,
                        color=carla.Color(255, 255, 0)
                    )
                    self._world.debug.draw_string(
                        scen_location + carla.Location(z=5),
                        str(black_var_name),
                        False,
                        color=carla.Color(0, 0, 0),
                        life_time=1000
                    )

        return new_status


class SwitchWrongDirectionTest(AtomicBehavior):

    """
    Atomic that switch the OutsideRouteLanesTest criterion.

    Args:
        active (bool): True: activated; False: deactivated
        name (str): name of the behavior
    """

    def __init__(self, active, name="SwitchWrongDirectionTest"):
        """
        Setup class members
        """
        self._active = active
        super().__init__(name)

    def update(self):
        set_blackboard("AC_SwitchWrongDirectionTest", self._active)
        return py_trees.common.Status.SUCCESS


class ScenarioTimeout(AtomicBehavior):

    """
    This class is an idle behavior that waits for a set amount of time
    before stopping.

    It is meant to be used with the `ScenarioTimeoutTest` to be used at scenarios
    that block the ego's route (such as adding obstacles) so that if the ego is
    incapable of surpassing them, it isn't blocked forever. Instead,
    the scenario will timeout, but it will be penalized by the `ScenarioTimeoutTest`

    Parameters:
    - duration: Duration in seconds of this behavior
    """

    def __init__(self, duration, scenario_name, name="ScenarioTimeout"):
        """
        Setup actor
        """
        super().__init__(name)
        self._duration = duration
        self._scenario_name = scenario_name
        self._start_time = 0
        self._scenario_timeout = False
        self._terminated = False
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._start_time = GameTime.get_time()
        set_blackboard("AC_SwitchActorBlockedTest", False)
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._duration:
            self._scenario_timeout = True
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def terminate(self, new_status):
        """
        Modifies the blackboard to tell the `ScenarioTimeoutTest` if the timeout was triggered
        """
        if not self._terminated:  # py_trees calls the terminate several times for some reason.
            set_blackboard(
                f"ScenarioTimeout_{self._scenario_name}",
                self._scenario_timeout,
            )
            set_blackboard(
                "AC_SwitchActorBlockedTest", True
            )
            self._terminated = True
        super().terminate(new_status)


class MovePedestrianWithEgo(AtomicBehavior):
    """This class is an atomic behavior that moves a pedestrian with the ego vehicle."""

    def __init__(self, reference_actor, actor, distance, displacement=0, name="TrackActor"):
        """
        Setup actor
        """
        super().__init__(name)
        self._actor = actor
        self._reference_actor = reference_actor
        self._distance = distance
        self._displacement = displacement

        added_location = carla.Location(x=self._displacement, z=-self._distance)
        self._actor.set_location(self._reference_actor.get_location() + added_location)

        self._start_time = 0
        self._teleport_time = 5

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._start_time = GameTime.get_time()
        added_location = carla.Location(x=self._displacement, z=-self._distance)
        self._actor.set_location(self._reference_actor.get_location() + added_location)
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._teleport_time:
            added_location = carla.Location(x=self._displacement, z=-self._distance)
            self._actor.set_location(self._reference_actor.get_location() + added_location)
            self._start_time = GameTime.get_time()
        return new_status
