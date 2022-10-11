#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import glob
import os
import sys
import importlib
import inspect
import py_trees
import traceback

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ScenarioTriggerer, Idle
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitForBlackboardVariable
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorBlockedTest,
                                                                     MinimumSpeedRouteTest)

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.background_activity import BackgroundBehavior
from srunner.scenariomanager.weather_sim import RouteWeatherBehavior
from srunner.scenariomanager.lights_sim import RouteLightsBehavior
from srunner.scenariomanager.timer import RouteTimeoutBehavior

from leaderboard.utils.route_parser import RouteParser, DIST_THRESHOLD
from leaderboard.utils.route_manipulation import interpolate_trajectory

import leaderboard.utils.parked_vehicles as parked_vehicles


class RouteScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"

    def __init__(self, world, config, debug_mode=0, criteria_enable=True):
        """
        Setup all relevant parameters and create scenarios along route
        """
        self.config = config
        self.route = self._get_route(config)
        self.list_scenarios = []
        sampled_scenario_definitions = self._filter_scenarios(config.scenario_configs)

        ego_vehicle = self._spawn_ego_vehicle(world)
        if ego_vehicle is None:
            raise ValueError("Shutting down, couldn't spawn the ego vehicle")

        if debug_mode>0:
            self._draw_waypoints(world, self.route, vertical_shift=0.1, size=0.05, persistency=10000)

        self._build_scenarios(
            world, ego_vehicle, sampled_scenario_definitions, timeout=10000, debug=debug_mode > 0
        )

        self._spawn_parked_vehicles()

        super(RouteScenario, self).__init__(
            config.name, [ego_vehicle], config, world, debug_mode > 3, False, criteria_enable
        )

    def _get_route(self, config):
        """
        Gets the route from the configuration, interpolating it to the desired density,
        saving it to the CarlaDataProvider and sending it to the agent

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        - debug_mode: boolean to decide whether or not the route poitns are printed
        """

        # prepare route's trajectory (interpolate and add the GPS route)
        self.gps_route, self.route = interpolate_trajectory(config.keypoints)
        return self.route

    def _filter_scenarios(self, scenario_configs):
        """
        Given a list of scenarios, filters out does that don't make sense to be triggered,
        as they are either too far from the route or don't fit with the route shape

        Parameters:
        - scenario_configs: list of ScenarioConfiguration
        """
        new_scenarios_config = []
        for scenario_config in scenario_configs:
            trigger_point = scenario_config.trigger_points[0]
            if not RouteParser.is_scenario_at_route(trigger_point, self.route):
                print("WARNING: Ignoring scenario '{}' as it is too far from the route".format(scenario_config.name))
                continue

            new_scenarios_config.append(scenario_config)

        return new_scenarios_config

    def _spawn_ego_vehicle(self, world):
        """Spawn the ego vehicle at the first waypoint of the route"""
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz_2020',
                                                          elevate_transform,
                                                          rolename='hero')
        if not ego_vehicle:
            return

        spectator = CarlaDataProvider.get_world().get_spectator()
        spectator.set_transform(carla.Transform(elevate_transform.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        world.tick()

        return ego_vehicle

    def _spawn_parked_vehicles(self):
        """Spawn parked vehicles."""
        SpawnActor = carla.command.SpawnActor

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for route_transform, _ in self.route:
            min_x = min(min_x, route_transform.location.x - 100)
            min_y = min(min_y, route_transform.location.y - 100)

            max_x = max(max_x, route_transform.location.x + 100)
            max_y = max(max_y, route_transform.location.y + 100)

        # Occupied parking locations
        all_occupied_parking_locations = []
        for scenario in self.list_scenarios:
            all_occupied_parking_locations.extend(scenario.get_parking_slots())

        all_available_parking_slots = []
        map_name = CarlaDataProvider.get_map().name.split('/')[-1]
        all_available_parking_slots = getattr(parked_vehicles, map_name, [])

        batch = []
        for slot in all_available_parking_slots:
            slot_transform = carla.Transform(
                location=carla.Location(slot["location"][0], slot["location"][1], slot["location"][2]),
                rotation=carla.Rotation(slot["rotation"][0], slot["rotation"][1], slot["rotation"][2])
            )
            mesh_path = slot["mesh"]

            if not (min_x < slot_transform.location.x < max_x) or not (min_y < slot_transform.location.y < max_y):
                continue

            is_free = True
            for occupied_slot in all_occupied_parking_locations:
                distance = slot_transform.location.distance(occupied_slot)
                if distance < 10:
                    is_free = False
                    break
            if is_free:
                mesh_bp = CarlaDataProvider.get_world().get_blueprint_library().filter("static.prop.mesh")[0]
                mesh_bp.set_attribute("mesh_path", mesh_path)
                mesh_bp.set_attribute("scale", "0.9")
                batch.append(SpawnActor(mesh_bp, slot_transform))
            else:
                pass

        CarlaDataProvider.get_client().apply_batch(batch)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, size, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)


        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2*size,
                               color=carla.Color(0, 0, 128), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=2*size,
                               color=carla.Color(128, 128, 128), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios):
        """
        Sample the scenarios that are going to happen for this route.
        """
        def select_scenario(scenario_list):
            """Higher number scenarios have higher priority."""
            higher_id = -1
            selected_scenario = None
            for scenario in scenario_list:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1
                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario
            return selected_scenario

        sampled_scenarios = []
        for trigger in list(potential_scenarios):
            scenario_list = potential_scenarios[trigger]
            sampled_scenarios.append(select_scenario(scenario_list))

        return sampled_scenarios

    def get_all_scenario_classes(self):

        # Path of all scenario at "srunner/scenarios" folder
        scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))

        all_scenario_classes = {}

        for scenario_file in scenarios_list:

            # Get their module
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # And their members of type class
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                # TODO: Filter out any class that isn't a child of BasicScenario
                all_scenario_classes[member[0]] = member[1]

        return all_scenario_classes

    def _build_scenarios(self, world, ego_vehicle, scenario_definitions, scenarios_per_tick=5, timeout=300, debug=False):
        """
        Initializes the class of all the scenarios that will be present in the route.
        If a class fails to be initialized, a warning is printed but the route execution isn't stopped
        """
        all_scenario_classes = self.get_all_scenario_classes()
        self.list_scenarios = []
        ego_data = ActorConfigurationData(ego_vehicle.type_id, ego_vehicle.get_transform(), 'hero')

        if debug:
            tmap = CarlaDataProvider.get_map()
            for scenario_config in scenario_definitions:
                scenario_loc = scenario_config.trigger_points[0].location
                debug_loc = tmap.get_waypoint(scenario_loc).transform.location + carla.Location(z=0.2)
                world.debug.draw_point(debug_loc, size=0.1, color=carla.Color(128, 0, 0), life_time=timeout)
                world.debug.draw_string(debug_loc, str(scenario_config.name), draw_shadow=False,
                                        color=carla.Color(0, 0, 128), life_time=timeout, persistent_lines=True)

        for scenario_number, scenario_config in enumerate(scenario_definitions):
            scenario_config.ego_vehicles = [ego_data]
            scenario_config.route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_config.route = self.route

            try:
                scenario_class = all_scenario_classes[scenario_config.type]
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_config, timeout=timeout)

                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    world.tick()

            except Exception as e:
                print(f"\033[93mSkipping scenario '{scenario_config.name}' due to setup error: {e}")
                if debug:
                    print(f"\n{traceback.format_exc()}")
                print("\033[0m", end="")
                continue

            self.list_scenarios.append(scenario_instance)

    # pylint: enable=no-self-use
    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Creates a parallel behavior that runs all of the scenarios part of the route.
        These subbehaviors have had a trigger condition added so that they wait until
        the agent is close to their trigger point before activating.

        It also adds the BackgroundActivity scenario, which will be active throughout the whole route.
        This behavior never ends and the end condition is given by the RouteCompletionTest criterion.
        """
        scenario_trigger_distance = DIST_THRESHOLD  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(name="Route Behavior",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        scenario_behaviors = []
        blackboard_list = []

        for scenario in self.list_scenarios:
            if scenario.behavior_tree is not None:
                scenario_behaviors.append(scenario.behavior_tree)
                blackboard_list.append([scenario.config.route_var_name,
                                        scenario.config.trigger_points[0].location])

        # Add the behavior that manages the scenario trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance)
        behavior.add_child(scenario_triggerer)  # Tick the ScenarioTriggerer before the scenarios

        # Add the Background Activity
        behavior.add_child(BackgroundBehavior(self.ego_vehicles[0], self.route, name="BackgroundActivity"))

        behavior.add_children(scenario_behaviors)
        return behavior

    def _create_test_criteria(self):
        """
        Create the criteria tree. It starts with some route criteria (which are always active),
        and adds the scenario specific ones, which will only be active during their scenario
        """
        criteria = py_trees.composites.Parallel(name="Criteria",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # End condition
        criteria.add_child(RouteCompletionTest(self.ego_vehicles[0], route=self.route))

        # 'Normal' criteria
        criteria.add_child(OutsideRouteLanesTest(self.ego_vehicles[0], route=self.route))
        criteria.add_child(CollisionTest(self.ego_vehicles[0], name="CollisionTest"))
        criteria.add_child(RunningRedLightTest(self.ego_vehicles[0]))
        criteria.add_child(RunningStopTest(self.ego_vehicles[0]))
        criteria.add_child(MinimumSpeedRouteTest(self.ego_vehicles[0], self.route, checkpoints=4, name="MinSpeedTest"))

        # These stop the route early to save computational time
        criteria.add_child(InRouteTest(
            self.ego_vehicles[0], route=self.route, offroad_max=30, terminate_on_failure=True))
        criteria.add_child(ActorBlockedTest(
            self.ego_vehicles[0], min_speed=0.1, max_time=180.0, terminate_on_failure=True, name="AgentBlockedTest")
        )

        for scenario in self.list_scenarios:
            scenario_criteria = scenario.get_criteria()
            if len(scenario_criteria) == 0:
                continue  # No need to create anything

            criteria.add_child(
                self._create_criterion_tree(scenario, scenario_criteria)
            )

        return criteria

    def _create_weather_behavior(self):
        """
        Create the weather behavior
        """
        if len(self.config.weather) == 1:
            return  # Just set the weather at the beginning and done
        return RouteWeatherBehavior(self.ego_vehicles[0], self.route, self.config.weather)

    def _create_lights_behavior(self):
        """
        Create the street lights behavior
        """
        return RouteLightsBehavior(self.ego_vehicles[0], 100)

    def _create_timeout_behavior(self):
        """
        Create the timeout behavior
        """
        return RouteTimeoutBehavior(self.ego_vehicles[0], self.route)

    def _initialize_environment(self, world):
        """
        Set the weather
        """
        # Set the appropriate weather conditions
        world.set_weather(self.config.weather[0][1])

    def _create_criterion_tree(self, scenario, criteria):
        """
        We can make use of the blackboard variables used by the behaviors themselves,
        as we already have an atomic that handles their (de)activation.
        The criteria will wait until that variable is active (the scenario has started),
        and will automatically stop when it deactivates (as the scenario has finished)
        """
        scenario_name = scenario.name
        var_name = scenario.config.route_var_name
        check_name = "WaitForBlackboardVariable: {}".format(var_name)

        criteria_tree = py_trees.composites.Sequence(name=scenario_name)
        criteria_tree.add_child(WaitForBlackboardVariable(var_name, True, False, name=check_name))

        scenario_criteria = py_trees.composites.Parallel(name=scenario_name,
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        for criterion in criteria:
            scenario_criteria.add_child(criterion)
        scenario_criteria.add_child(WaitForBlackboardVariable(var_name, False, None, name=check_name))

        criteria_tree.add_child(scenario_criteria)
        criteria_tree.add_child(Idle())  # Avoid the indivual criteria stopping the simulation
        return criteria_tree

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
