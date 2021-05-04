#!/usr/bin/env python
# Copyright (c) 2021 Intel Corporation.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import argparse
import os
import traceback
from argparse import RawTextHelpFormatter

from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager


def serialize_header(f):
    s =  "| Town | Route | Index | # of Vehicles | Mean FPS | Mean ratio | Sensors |\n"
    s += "| ---- | ----- | ----- | ------------- | -------- | ---------- | ------- |\n"
    f.write(s)
    f.flush()


def serialize_records(f, town, route, repetition, background_amount, duration_game, duration_system, sensors):

    def serialize_sensor(sensor_spec):
        if sensor_spec['type'] == 'sensor.camera.rgb':
            return "RGB_{}x{}".format(sensor_spec['width'], sensor_spec['height'])
        elif sensor_spec['type'] == 'sensor.lidar.ray_cast':
            return "Lidar"
        elif sensor_spec['type'] == 'sensor.other.radar':
            return "Radar"
        elif sensor_spec['type'] == 'sensor.other.gnss':
            return "gnss"
        elif sensor_spec['type'] == 'sensor.other.imu':
            return "imu"
        elif sensor_spec['type'] == 'sensor.opendrive_map':
            return "map"
        elif sensor_spec['type'] == 'sensor.speedometer':
            return "speed"
        else:
            return ""

    ratio = 0.0
    if duration_system > 0:
        ratio = duration_game / duration_system

    s = "| {} | {} | {} | {} | {:03.2f} | {:03.2f} | {} |\n".format(
        town,
        route,
        repetition,
        background_amount,
        20 * ratio,
        ratio,
        " ".join([serialize_sensor(s) for s in sensors]),
    )
    
    f.write(s)
    f.flush()


class LeaderboardBenchmarkEvaluator(LeaderboardEvaluator):

    def __init__(self, args, statistics_manager):
        super(LeaderboardBenchmarkEvaluator, self).__init__(args, statistics_manager)

        if os.path.isfile(args.benchmark_filename) and args.resume:
            self.benchmark_file = open(args.benchmark_filename, "a")
        else:
            self.benchmark_file = open(args.benchmark_filename, "w")
            serialize_header(self.benchmark_file)

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        super(LeaderboardBenchmarkEvaluator, self)._register_statistics(
            config, checkpoint, entry_status, crash_message
        )

        serialize_records(
            self.benchmark_file,
            config.town,
            config.name,
            config.repetition_index,
            self.manager.scenario_class.background_amount,
            self.manager.scenario_duration_game,
            self.manager.scenario_duration_system,
            self.sensors
        )


def main():
    description = "CARLA AD Leaderboard Benchmark Evaluation: benchmark your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=0, type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default=60.0, type=float,
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    # benchmark options
    parser.add_argument("--benchmark-filename", type=str,
                        default='benchmark.md',
                        help="File used for saving benchmark results")

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    try:
        leaderboard_evaluator = LeaderboardBenchmarkEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
