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
import py_trees
import sys

from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict, create_default_json_msg


PENALTY_COLLISION_STATIC = 0.8
PENALTY_COLLISION_VEHICLE = 0.8
PENALTY_COLLISION_PEDESTRIAN = 0.8
PENALTY_TRAFFIC_LIGHT = 0.95
PENALTY_WRONG_WAY = 0.95
PENALTY_SIDEWALK_INVASION = 0.85
PENALTY_STOP = 0.95

class RouteRecord():

    def __init__(self):
        self.route_id = None
        self.index = None
        self.status = 'Started'
        self.infractions = {
            'collisions_layout': [],
            'collision_vehicle': [],
            'collisions_pedestrian': [],
            'red_light': [],
            'wrong_way': [],
            'route_dev': [],
            'sidewalk_invasion': [],
            'stop_infraction': []
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

class StatisticsManager(object):

    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """
    logger = None

    def __init__(self):
        self._master_scenario = None
        self._registry_route_records = []

    def resume(self, endpoint):
        data = fetch_dict(endpoint)

        if data and dictor(data, 'value.results._checkpoint.records'):
            records = data['value']['results']['_checkpoint']['records']

            for record in records:
                self._registry_route_records.append(to_route_record(record))

    @staticmethod
    def set_logger(logger):
        StatisticsManager.logger = logger

    def set_route(self, route_id, index, scenario):
        self._master_scenario = scenario

        route_record = RouteRecord()
        route_record.route_id = route_id
        route_record.index = index
        self._registry_route_records.append(route_record)

    def compute_route_statistics(self):
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """

        if not self._registry_route_records:
            raise Exception('Critical error with the route registry.')

        # fetch latest record to fill in
        route_record = self._registry_route_records[-1]

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0

        for node in self._master_scenario.get_criteria():
            if node.list_traffic_events:
                # analyze all traffic events
                for event in node.list_traffic_events:
                    if event.get_type() == TrafficEventType.COLLISION_STATIC:
                        score_penalty *= PENALTY_COLLISION_STATIC
                        route_record['collisions_layout'].append(event)

                    elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                        score_penalty *= PENALTY_COLLISION_VEHICLE
                        route_record['collision_vehicle'].append(event)

                    elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                        score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                        route_record['collisions_pedestrian'].append(event)

                    elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                        score_penalty *= PENALTY_TRAFFIC_LIGHT
                        route_record['red_light'].append(event)

                    elif event.get_type() == TrafficEventType.WRONG_WAY_INFRACTION:
                        score_penalty *= PENALTY_WRONG_WAY
                        route_record['wrong_way'].append(event)

                    elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                        route_record['route_dev'].append(event)

                    elif event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
                        score_penalty *= PENALTY_SIDEWALK_INVASION
                        route_record['sidewalk_invasion'].append(event)

                    elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                        score_penalty *= PENALTY_STOP
                        route_record['stop_infraction'].append(event)

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

        if self._registry_route_records:
            for route_record in self._registry_route_records:
                global_record.scores['score_route'] += route_record.scores['score_route']
                global_record.scores['score_penalty'] += route_record.scores['score_penalty']
                global_record.scores['score_composed'] += route_record.scores['score_composed']

                for key in global_record.infractions.keys():
                    global_record.infractions[key] = len(route_record.infractions[key])

                if route_record.status is not 'Completed':
                    if not 'exceptions' in global_record.meta:
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
        record_list = data['value']['results']['_checkpoint']['records']
        if index > len(record_list):
            print('Error! No enough entries in the list')
            sys.exit(-1)
        elif index == len(record_list):
            record_list.append(stats_dict)
        else:
            record_list[index] = stats_dict

        save_dict(endpoint, data)

    @staticmethod
    def save_global_record(route_record, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        data['value']['results']['_checkpoint']['global_record'] = stats_dict

        save_dict(endpoint, data)