#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import inspect
import os
import time
import pkg_resources
import sys

import carla
from srunner.challenge.autoagents.agent_wrapper import SensorConfigurationInvalid
from srunner.challenge.challenge_statistics_manager import ChallengeStatisticsManager
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import *
#from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenarios.control_loss import *
from srunner.scenarios.follow_leading_vehicle import *
from srunner.scenarios.maneuver_opposite_direction import *
from srunner.scenarios.no_signal_junction_crossing import *
from srunner.scenarios.object_crash_intersection import *
from srunner.scenarios.object_crash_vehicle import *
from srunner.scenarios.opposite_vehicle_taking_priority import *
from srunner.scenarios.other_leading_vehicle import *
from srunner.scenarios.signalized_junction_left_turn import *
from srunner.scenarios.signalized_junction_right_turn import *
from srunner.scenarios.change_lane import *
from srunner.scenarios.cut_in import *
from srunner.tools.scenario_config_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser


from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.scenarios.scenario_manager import ScenarioManager



class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 30.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    # CARLA world and scenario handlers
    world = None
    manager = None

    additional_scenario_module = None

    agent_instance = None
    module_agent = None

    def __init__(self, args):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)

        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion('0.9.6'):
            raise ImportError("CARLA version 0.9.6 or newer required. CARLA version found: {}".format(dist))


        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.debug, True)

        self._start_wall_time = datetime.now()

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup(True)
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world

    def _within_available_time(self):
        """
        Check if the elapsed runtime is within the remaining user time budget
        Only relevant when running in challenge mode
        """
        current_time = datetime.now()
        elapsed_seconds = (current_time - self._start_wall_time).seconds

        return elapsed_seconds < int(os.getenv('CHALLENGE_TIME_AVAILABLE', '1080000'))

    def _get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """

        for scenarios in SCENARIOS.values():
            if scenario in scenarios:
                if scenario in globals():
                    return globals()[scenario]

        for member in inspect.getmembers(self.additional_scenario_module):
            if scenario in member and inspect.isclass(member[1]):
                return member[1]

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def _cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """

        self.client.stop_recorder()

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if ego:
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaActorPool.setup_actor(vehicle.model,
                                                                    vehicle.transform,
                                                                    vehicle.rolename,
                                                                    True,
                                                                    color=vehicle.color,
                                                                    vehicle_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _analyze_scenario(self, args, config):
        """
        Provide feedback about success/failure of a scenario
        """

        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        config_name = config.name


    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaActorPool and CarlaDataProvider
        """

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        CarlaActorPool.set_client(self.client)
        CarlaActorPool.set_world(self.world)
        CarlaDataProvider.set_world(self.world)

        # Wait for the world to be ready
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config
        """

        if not self._load_and_wait_for_world(args, config.town, config.ego_vehicles):
            self._cleanup()
            return

        agent_class_name = getattr(self.module_agent, 'get_entry_point')()
        try:
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
            config.agent = self.agent_instance
        except Exception as e:
            print("Could not setup required agent due to {}".format(e))
            self._cleanup()
            return

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)

        except Exception as exception:
            print("The scenario cannot be loaded")
            if args.debug:
                traceback.print_exc()
            print(exception)
            self._cleanup()
            return

        # Set the appropriate weather conditions
        weather = carla.WeatherParameters(
            cloudyness=config.weather.cloudyness,
            precipitation=config.weather.precipitation,
            precipitation_deposits=config.weather.precipitation_deposits,
            wind_intensity=config.weather.wind_intensity,
            sun_azimuth_angle=config.weather.sun_azimuth,
            sun_altitude_angle=config.weather.sun_altitude
        )

        self.world.set_weather(weather)

        # Set the appropriate road friction
        if config.friction is not None:
            friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            self.world.spawn_actor(friction_bp, transform)

        import pdb; pdb.set_trace()
        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}.log".format(os.getenv('ROOT_SCENARIO_RUNNER', "./"), config.name))
            self.manager.load_scenario(scenario, self.agent_instance)
            print('OK')
            self.manager.run_scenario()
            print('AK')

            # Stop scenario
            self.manager.stop_scenario()

            # Provide outputs if required
            self._analyze_scenario(args, config)

            # Remove all actors
            scenario.remove_all_actors()
        except SensorConfigurationInvalid as e:
            self._cleanup(True)
            ChallengeStatisticsManager.record_fatal_error(e)
            sys.exit(-1)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            ChallengeStatisticsManager.set_error_message(traceback.format_exc())
            print(e)

        self._cleanup()

    def _run_scenarios(self, args):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """

        # Setup and run the scenarios for repetition times
        for _ in range(int(args.repetitions)):

            # Load the scenario configurations provided in the config file
            scenario_configurations = None
            scenario_config_file = ScenarioConfigurationParser.find_scenario_config(args.scenario)
            if scenario_config_file is None:
                print("Configuration for scenario {} cannot be found!".format(args.scenario))
                continue

            scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(scenario_config_file,
                                                                                               args.scenario)

            # Execute each configuration
            for config in scenario_configurations:
                self._load_and_run_scenario(args, config)

            self._cleanup(ego=True)

            print("No more scenarios .... Exiting")

    def run(self, args):
        """
        Run the challenge mode
        """

        phase_codename = os.getenv('CHALLENGE_PHASE_CODENAME', 'dev_track_3')
        phase = phase_codename.split("_")[0]

        repetitions = args.repetitions
        routes = args.routes
        weather_profiles = CarlaDataProvider.find_weather_presets()


        # retrieve routes
        route_descriptions_list = RouteParser.parse_routes_file(routes, False)
        # find and filter potential scenarios for each of the evaluated routes
        # For each of the routes and corresponding possible scenarios to be evaluated.

        n_routes = len(route_descriptions_list) * repetitions
        ChallengeStatisticsManager.set_number_of_scenarios(n_routes)

        for _, route_description in enumerate(route_descriptions_list):
            for repetition in range(repetitions):

                if not self._within_available_time():
                    error_message = 'Not enough simulation time available to continue'
                    print(error_message)
                    ChallengeStatisticsManager.record_fatal_error(error_message)
                    self._cleanup(True)
                    sys.exit(-1)

                config = RouteScenarioConfiguration(route_description, args.scenarios)
                profile = weather_profiles[repetition % len(weather_profiles)]
                config.weather = profile[0]
                config.weather.sun_azimuth = -1
                config.weather.sun_altitude = -1

                self._load_and_run_scenario(args, config)
                self._cleanup(ego=True)

def main():
    description = ("CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n")

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--show-to-participant', type=bool, help='Show results to participant?', default=True)
    parser.add_argument('--spectator', type=bool, help='Switch spectator view on?', default=True)
    parser.add_argument('--record', action="store_true",
                        help='Use CARLA recording feature to create a recording of the scenario')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.')
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate")
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    arguments = parser.parse_args()

    CARLA_ROOT = os.environ.get('CARLA_ROOT')
    ROOT_SCENARIO_RUNNER = os.environ.get('ROOT_SCENARIO_RUNNER')

    if not CARLA_ROOT:
        print("Error. CARLA_ROOT not found. Please run setup_environment.sh first.")
        sys.exit(0)

    if not ROOT_SCENARIO_RUNNER:
        print("Error. ROOT_SCENARIO_RUNNER not found. Please run setup_environment.sh first.")
        sys.exit(0)

    if arguments.scenarios is None:
        print("Please specify a path to a scenario specification file  '--scenarios path-to-file'\n\n")
        parser.print_help(sys.stdout)
        sys.exit(0)

    arguments.carla_root = CARLA_ROOT
    challenge_evaluator = None

    phase_codename = os.getenv('CHALLENGE_PHASE_CODENAME', 'dev_track_3')
    if not phase_codename:
        raise ValueError('environment variable CHALLENGE_PHASE_CODENAME not defined')
    track = int(phase_codename.split("_")[2])
    phase_codename = phase_codename.split("_")[0]
    if phase_codename == 'test':
        arguments.show_to_participant = False

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments)
        leaderboard_evaluator.run(arguments)
    except Exception as e:
        traceback.print_exc()
        if leaderboard_evaluator:
            leaderboard_evaluator.report_challenge_statistics(arguments.filename, arguments.show_to_participant)
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
