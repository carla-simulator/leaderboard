#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic scenario behaviors that reflect
trigger conditions to either activate another behavior, or to stop
another behavior.

For example, such a condition could be "InTriggerRegion", which checks
that a given actor reached a certain region on the map, and then starts/stops
a behavior of this actor.

The atomics are implemented with py_trees and make use of the AtomicCondition
base class
"""

import operator
import py_trees


from leaderboard.scenariomanager.scenarioatomics.atomic_behaviors import calculate_distance
from leaderboard.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.tools.scenario_helper import get_distance_between_actors


EPSILON = 0.001


class AtomicCondition(py_trees.behaviour.Behaviour):

    """
    Base class for all atomic conditions used to setup a scenario

    *All behaviors should use this class as parent*

    Important parameters:
    - name: Name of the atomic condition
    """

    def __init__(self, name):
        """
        Default init. Has to be called via super from derived class
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name

    def setup(self, **kwargs):
        """
        Default setup
        """
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        """
        Initialise setup
        """
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class InTriggerRegion(AtomicCondition):

    """
    This class contains the trigger region (condition) of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - min_x, max_x, min_y, max_y: bounding box of the trigger region

    The condition terminates with SUCCESS, when the actor reached the target region
    """

    def __init__(self, actor, min_x, max_x, min_y, max_y, name="TriggerRegion"):
        """
        Setup trigger region (rectangle provided by
        [min_x,min_y] and [max_x,max_y]
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def update(self):
        """
        Check if the _actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)

        if location is None:
            return new_status

        not_in_region = (location.x < self._min_x or location.x > self._max_x) or \
                        (location.y < self._min_y or location.y > self._max_y)
        if not not_in_region:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToVehicle(AtomicCondition):

    """
    This class contains the trigger distance (condition) between to actors
    of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - reference_actor: Reference CARLA actor
    - name: Name of the condition
    - distance: Trigger distance between the two actors in meters
    - distance_type: Specifies how distance should be calculated between the two actors
    - freespace: if True distance is calculated between closest boundary points else it will be from center-center
    - dx, dy, dz: distance to reference_location (location of reference_actor)

    The condition terminates with SUCCESS, when the actor reached the target distance to the other actor
    """

    def __init__(self, reference_actor, actor, distance, comparison_operator=operator.lt,
                 distance_type="cartesianDistance", freespace=False, name="TriggerDistanceToVehicle"):
        """
        Setup trigger distance
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._reference_actor = reference_actor
        self._actor = actor
        self._distance = distance
        self._distance_type = distance_type
        self._freespace = freespace
        self._comparison_operator = comparison_operator

        if distance_type == "longitudinal":
            self._global_rp = CarlaDataProvider.get_global_route_planner()
        else:
            self._global_rp = None

    def update(self):
        """
        Check if the ego vehicle is within trigger distance to other actor
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        reference_location = CarlaDataProvider.get_location(self._reference_actor)

        if location is None or reference_location is None:
            return new_status

        distance = get_distance_between_actors(
            self._actor, self._reference_actor, self._distance_type, self._freespace, self._global_rp)

        if self._comparison_operator(distance, self._distance):
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToLocation(AtomicCondition):

    """
    This class contains the trigger (condition) for a distance to a fixed
    location of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_location: Reference location (carla.location)
    - name: Name of the condition
    - distance: Trigger distance between the actor and the target location in meters

    The condition terminates with SUCCESS, when the actor reached the target distance to the given location
    """

    def __init__(self,
                 actor,
                 target_location,
                 distance,
                 comparison_operator=operator.lt,
                 name="InTriggerDistanceToLocation"):
        """
        Setup trigger distance
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_location = target_location
        self._actor = actor
        self._distance = distance
        self._comparison_operator = comparison_operator

    def update(self):
        """
        Check if the actor is within trigger distance to the target location
        """

        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)

        if location is None:
            return new_status

        if self._comparison_operator(calculate_distance(
                location, self._target_location), self._distance):
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTimeToArrivalToLocation(AtomicCondition):

    """
    This class contains a check if a actor arrives within a given time
    at a given location.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - time: The behavior is successful, if TTA is less than _time_ in seconds
    - location: Location to be checked in this behavior

    The condition terminates with SUCCESS, when the actor can reach the target location within the given time
    """

    _max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, actor, time, location, comparison_operator=operator.lt, name="TimeToArrival"):
        """
        Setup parameters
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._time = time
        self._target_location = location
        self._comparison_operator = comparison_operator

    def update(self):
        """
        Check if the actor can arrive at target_location within time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self._actor)

        if current_location is None:
            return new_status

        distance = calculate_distance(current_location, self._target_location)
        velocity = CarlaDataProvider.get_velocity(self._actor)

        # if velocity is too small, simply use a large time to arrival
        time_to_arrival = self._max_time_to_arrival
        if velocity > EPSILON:
            time_to_arrival = distance / velocity

        if self._comparison_operator(time_to_arrival, self._time):
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class WaitUntilInFront(AtomicCondition):

    """
    Behavior that support the creation of cut ins. It waits until the actor has passed another actor

    Parameters:
    - actor: the one getting in front of the other actor
    - other_actor: the reference vehicle that the actor will have to get in front of
    - factor: How much in front the actor will have to get (from 0 to infinity):
        0: They are right next to each other
        1: The front of the other_actor and the back of the actor are right next to each other
    """

    def __init__(self, actor, other_actor, factor=1, check_distance=True, name="WaitUntilInFront"):
        """
        Init
        """
        super().__init__(name)
        self._actor = actor
        self._other_actor = other_actor
        self._distance = 10
        self._factor = max(EPSILON, factor)  # Must be > 0
        self._check_distance = check_distance

        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map(self._world)

        actor_extent = self._actor.bounding_box.extent.x
        other_extent = self._other_actor.bounding_box.extent.x
        self._length = self._factor * (actor_extent + other_extent)

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        Checks if the two actors meet the requirements
        """
        new_status = py_trees.common.Status.RUNNING

        in_front = False
        close_by = False

        # Location of our actor
        actor_location = CarlaDataProvider.get_location(self._actor)
        if actor_location is None:
            return new_status

        # Waypoint in front of the other actor
        other_location = CarlaDataProvider.get_location(self._other_actor)
        other_waypoint = self._map.get_waypoint(other_location)
        if other_waypoint is None:
            return new_status
        other_next_waypoints = other_waypoint.next(self._length)
        if other_next_waypoints is None:
            return new_status
        other_next_waypoint = other_next_waypoints[0]

        # Wait for the vehicle to be in front
        other_dir = other_next_waypoint.transform.get_forward_vector()
        act_other_dir = actor_location - other_next_waypoint.transform.location

        if other_dir.dot(act_other_dir) > 0.0:
            in_front = True

        # Wait for it to be close-by
        if not self._check_distance:
            close_by = True
        elif actor_location.distance(other_next_waypoint.transform.location) < self._distance:
            close_by = True

        if in_front and close_by:
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class WaitUntilInFrontPosition(AtomicCondition):

    """
    Behavior that support the creation of cut ins. It waits until the actor has passed another actor

    Parameters:
    - actor: the one getting in front of the other actor
    - transform: the reference transform that the actor will have to get in front of
    """

    def __init__(self, actor, transform, check_distance=True, distance=10, name="WaitUntilInFrontPosition"):
        """
        Init
        """
        super().__init__(name)

        self._actor = actor
        self._ref_transform = transform
        self._ref_location = transform.location
        self._ref_vector = transform.get_forward_vector()
        self._check_distance = check_distance
        self._distance = distance

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        Checks if the two actors meet the requirements
        """
        new_status = py_trees.common.Status.RUNNING

        # Is the actor in front?
        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        actor_dir = location - self._ref_location
        in_front = actor_dir.dot(self._ref_vector) > 0.0

        # Is the actor close-by?
        close_by = (
            not self._check_distance
            or location.distance(self._ref_location) < self._distance
        )

        if in_front and close_by:
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class DriveDistance(AtomicCondition):

    """
    This class contains an atomic behavior to drive a certain distance.

    Important parameters:
    - actor: CARLA actor to execute the condition
    - distance: Distance for this condition in meters

    The condition terminates with SUCCESS, when the actor drove at least the given distance
    """

    def __init__(self, actor, distance, name="DriveDistance"):
        """
        Setup parameters
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_distance = distance
        self._distance = 0
        self._location = None
        self._actor = actor

    def initialise(self):
        self._location = CarlaDataProvider.get_location(self._actor)
        super().initialise()

    def update(self):
        """
        Check driven distance
        """
        new_status = py_trees.common.Status.RUNNING

        if self._location is None:
            return new_status

        new_location = CarlaDataProvider.get_location(self._actor)
        self._distance += calculate_distance(self._location, new_location)
        self._location = new_location

        if self._distance > self._target_distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class WaitEndIntersection(AtomicCondition):

    """
    Atomic behavior that waits until the vehicles has gone outside the junction.
    If currently inside no intersection, it will wait until one is found.
    If 'junction_id' is given, it will wait until that specific junction has finished
    """

    def __init__(self, actor, junction_id=None, debug=False, name="WaitEndIntersection"):
        super().__init__(name)
        self.actor = actor
        self.debug = debug
        self._junction_id = junction_id
        self._inside_junction = False
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):

        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self.actor)
        waypoint = CarlaDataProvider.get_map().get_waypoint(location)

        # Wait for the actor to enter a / the junction
        if not self._inside_junction:
            junction = waypoint.get_junction()
            if not junction:
                return new_status
            if not self._junction_id or junction.id == self._junction_id:
                self._inside_junction = True

        # And to leave it
        elif self._inside_junction and not waypoint.is_junction:
            if self.debug:
                print("--- Leaving the junction")
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class WaitForBlackboardVariable(AtomicCondition):

    """
    Atomic behavior that keeps running until the blackboard variable is set to the corresponding value.
    Used to avoid returning FAILURE if the blackboard comparison fails.

    It also initially sets the variable to a given value, if given
    """

    def __init__(self, variable_name, variable_value, var_init_value=None,
                 debug=False, name="WaitForBlackboardVariable"):
        super().__init__(name)
        self._debug = debug
        self._variable_name = variable_name
        self._variable_value = variable_value
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        if var_init_value is not None:
            blackboard = py_trees.blackboard.Blackboard()
            _ = blackboard.set(variable_name, var_init_value)

    def update(self):

        new_status = py_trees.common.Status.RUNNING

        blackv = py_trees.blackboard.Blackboard()
        value = blackv.get(self._variable_name)
        if value == self._variable_value:
            if self._debug:
                print("Blackboard variable {} set to True".format(self._variable_name))
            new_status = py_trees.common.Status.SUCCESS

        return new_status
