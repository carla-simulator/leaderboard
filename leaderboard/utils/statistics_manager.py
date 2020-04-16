#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""

from __future__ import print_function

from dictor import dictor
import math
import sys

from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict, create_default_json_msg


PENALTY_COLLISION_STATIC = 0.65
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80


class RouteRecord():
    def __init__(self):
        self.route_id = None
        self.index = None
        self.status = 'Started'
        self.infractions = {
            'collisions_layout': [],
            'collisions_pedestrian': [],
            'collisions_vehicle': [],
            'outside_route_lanes': [],
            'red_light': [],
            'route_dev': [],
            'stop_infraction': [],
            'route_timeout': [],
            'vehicle_blocked': []
        }

        self.scores = {
            'score_route': 0,
            'score_penalty': 0,
            'score_composed': 0
        }

        self.meta = {}


def to_route_record(record_dict):
    record = RouteRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)

    return record


def compute_route_length(config):
    trajectory = config.route_description['trajectory']

    route_length = 0.0
    previous_location = None
    for location in trajectory:
        if previous_location:
            dist = math.sqrt((location.x-previous_location.x)*(location.x-previous_location.x) +
                             (location.y-previous_location.y)*(location.y-previous_location.y) +
                             (location.z - previous_location.z) * (location.z - previous_location.z))
            route_length += dist
        previous_location = location

    return route_length


class StatisticsManager(object):

    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self):
        self._master_scenario = None
        self._registry_route_records = []

    def resume(self, endpoint):
        data = fetch_dict(endpoint)

        if data and dictor(data, '_checkpoint.records'):
            records = data['_checkpoint']['records']

            for record in records:
                self._registry_route_records.append(to_route_record(record))

    def set_route(self, route_id, index, scenario):
        self._master_scenario = scenario

        route_record = RouteRecord()
        route_record.route_id = route_id
        route_record.index = index

        if index < len(self._registry_route_records):
            # the element already exists and therefore we update it
            self._registry_route_records[index] = route_record
        else:
            self._registry_route_records.append(route_record)

    def compute_route_statistics(self, config, duration_time_system=-1, duration_time_game=-1):
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """
        index = config.index

        if not self._registry_route_records or index >= len(self._registry_route_records):
            raise Exception('Critical error with the route registry.')

        # fetch latest record to fill in
        route_record = self._registry_route_records[index]

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0

        route_record.meta['duration_system'] = duration_time_system
        route_record.meta['duration_game'] = duration_time_game
        route_record.meta['route_length'] = compute_route_length(config)

        if self._master_scenario.timeout_node.timeout:
            route_record.infractions['route_timeout'].append('Route timeout.')

        for node in self._master_scenario.get_criteria():
            if node.list_traffic_events:
                # analyze all traffic events
                for event in node.list_traffic_events:
                    if event.get_type() == TrafficEventType.COLLISION_STATIC:
                        score_penalty *= PENALTY_COLLISION_STATIC
                        route_record.infractions['collisions_layout'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                        score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                        route_record.infractions['collisions_pedestrian'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                        score_penalty *= PENALTY_COLLISION_VEHICLE
                        route_record.infractions['collisions_vehicle'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                        score_penalty *= (1 - event.get_dict()['percentage'] / 100)
                        route_record.infractions['outside_route_lanes'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                        score_penalty *= PENALTY_TRAFFIC_LIGHT
                        route_record.infractions['red_light'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                        route_record.infractions['route_dev'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                        score_penalty *= PENALTY_STOP
                        route_record.infractions['stop_infraction'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                        route_record.infractions['vehicle_blocked'].append(event.get_message())

                    elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                        score_route = 100.0
                        target_reached = True
                    elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                        if not target_reached:
                            if event.get_dict():
                                score_route = event.get_dict()['route_completed']
                            else:
                                score_route = 0

        # update route scores
        route_record.scores['score_route'] = score_route
        route_record.scores['score_penalty'] = score_penalty
        route_record.scores['score_composed'] = max(score_route*score_penalty, 0.0)

        # update status
        if target_reached:
            route_record.status = 'Completed'
        else:
            route_record.status = 'Failed'

        return route_record

    def compute_global_statistics(self, total_routes):
        global_record = RouteRecord()
        global_record.route_id = -1
        global_record.index = -1
        global_record.status = 'Completed'

        if self._registry_route_records:
            for route_record in self._registry_route_records:
                global_record.scores['score_route'] += route_record.scores['score_route']
                global_record.scores['score_penalty'] += route_record.scores['score_penalty']
                global_record.scores['score_composed'] += route_record.scores['score_composed']

                for key in global_record.infractions.keys():
                    route_length_kms = route_record.scores['score_route'] * route_record.meta['route_length'] / 1000.0
                    if isinstance(global_record.infractions[key], list):
                        global_record.infractions[key] = len(route_record.infractions[key]) / route_length_kms
                    else:
                        global_record.infractions[key] += len(route_record.infractions[key]) / route_length_kms

                if route_record.status is not 'Completed':
                    global_record.status = 'Failed'
                    if 'exceptions' not in global_record.meta:
                        global_record.meta['exceptions'] = []
                    global_record.meta['exceptions'].append((route_record.route_id,
                                                             route_record.index,
                                                             route_record.status))

        global_record.scores['score_route'] /= float(total_routes)
        global_record.scores['score_penalty'] /= float(total_routes)
        global_record.scores['score_composed'] /= float(total_routes)

        return global_record

    @staticmethod
    def save_record(route_record, index, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        record_list = data['_checkpoint']['records']
        if index > len(record_list):
            print('Error! No enough entries in the list')
            sys.exit(-1)
        elif index == len(record_list):
            record_list.append(stats_dict)
        else:
            record_list[index] = stats_dict

        save_dict(endpoint, data)

    @staticmethod
    def save_global_record(route_record, sensors, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        data['_checkpoint']['global_record'] = stats_dict
        data['values'] = ['{:.3f}'.format(stats_dict['scores']['score_composed']),
                          '{:.3f}'.format(stats_dict['scores']['score_route']),
                          '{:.3f}'.format(stats_dict['scores']['score_penalty']),
                          # infractions
                          '{:.3f}'.format(stats_dict['infractions']['collisions_layout']),
                          '{:.3f}'.format(stats_dict['infractions']['collisions_pedestrian']),
                          '{:.3f}'.format(stats_dict['infractions']['collisions_vehicle']),
                          '{:.3f}'.format(stats_dict['infractions']['outside_route_lanes']),
                          '{:.3f}'.format(stats_dict['infractions']['red_light']),
                          '{:.3f}'.format(stats_dict['infractions']['route_dev']),
                          '{:.3f}'.format(stats_dict['infractions']['stop_infraction']),
                          '{:.3f}'.format(stats_dict['infractions']['route_timeout']),
                          '{:.3f}'.format(stats_dict['infractions']['vehicle_blocked'])
                          ]

        data['labels'] = ['avg. total score',
                          'avg. route score',
                          'avg. infraction penalty',
                          'collisions with layout',
                          'collisions with pedestrians',
                          'collisions with vehicles',
                          'off-road infractions',
                          'red lights infractions',
                          'route deviation infractions',
                          'stop sign infractions',
                          'timeouts',
                          'vehicle blocked'
                          ]

        data['sensors'] = sensors

        save_dict(endpoint, data)

    @staticmethod
    def clear_record(endpoint):
        if not endpoint.startswith(('http:', 'https:', 'ftp:')):
            with open(endpoint, 'w') as fd:
                fd.truncate(0)
