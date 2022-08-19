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

from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict

PENALTY_VALUE_DICT = {
    # Traffic events that substract a set amount of points.
    TrafficEventType.COLLISION_PEDESTRIAN: 0.5,
    TrafficEventType.COLLISION_VEHICLE: 0.6,
    TrafficEventType.COLLISION_STATIC: 0.65,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 0.7,
    TrafficEventType.STOP_INFRACTION: 0.8,
    TrafficEventType.SCENARIO_TIMEOUT: 0.7,
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 0.7
}
PENALTY_PERC_DICT = {
    # Traffic events that substract a varying amount of points. This is the per unit value.
    # 'increases' means that the higher the value, the higher the penalty.
    # 'decreases' means that the ideal value is 100 and the lower the value, the higher the penalty.
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: [0, 'increases'],  # All route traversed through outside lanes is ignored
    TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'decreases'],
}

PENALTY_NAME_DICT = {
    TrafficEventType.COLLISION_STATIC: 'Collisions with layout',
    TrafficEventType.COLLISION_PEDESTRIAN: 'Collisions with pedestrians',
    TrafficEventType.COLLISION_VEHICLE: 'Collisions with vehicles',
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 'Red lights infractions',
    TrafficEventType.STOP_INFRACTION: 'Stop sign infractions',
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: 'Off-road infractions',
    TrafficEventType.MIN_SPEED_INFRACTION: 'Min speed infractions',
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 'Yield to emergency vehicle infractions',
    TrafficEventType.SCENARIO_TIMEOUT: 'Scenario timeouts',
    TrafficEventType.ROUTE_DEVIATION: 'Route deviations',
    TrafficEventType.VEHICLE_BLOCKED: 'Agent blocked',
}

# Limit the entry status to some values. Eligible should always be gotten from this table
ENTRY_STATUS_VALUES = ['Started', 'Finished', 'Rejected', 'Crashed', 'Invalid']
ELIGIBLE_VALUES = {'Started': False, 'Finished': True, 'Rejected': False, 'Crashed': False, 'Invalid': False}

ROUND_DIGITS = 3
ROUND_DIGITS_SCORE = 6


class RouteRecord():
    def __init__(self):
        self.route_id = None
        self.result = 'Started'
        self.num_infractions = 0
        self.infractions = {'Route timeouts': []}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions[event_name] = []

        self.scores = {
            'Driving score': 0,
            'Route completion': 0,
            'Infraction penalty': 0
        }

        self.meta = {
            'Route length': 0,
            'Game duration': 0,
            'System duration': 0,
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)


class GlobalRecord():
    def __init__(self):
        self.result = 'Perfect'
        self.infractions_per_km = {'Route timeouts': 0}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions_per_km[event_name] = 0

        self.scores_mean = {
            'Driving score': 0,
            'Route completion': 0,
            'Infraction penalty': 0
        }
        self.scores_std_dev = self.scores_mean.copy()

        self.meta = {
            'exceptions': {}
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)

class Checkpoint():

    def __init__(self):
        self.global_record = {}
        self.progress = []
        self.records = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['global_record'] = self.global_record.to_json() if self.global_record else {}
        d['progress'] = self.progress
        d['records'] = [x.to_json() for x in self.records]

        return d


class Results():

    def __init__(self):
        self.checkpoint = Checkpoint()
        self.entry_status = "Started"
        self.eligible = ELIGIBLE_VALUES[self.entry_status]
        self.sensors = []
        self.values = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['_checkpoint'] = self.checkpoint.to_json()
        d['entry_status'] = self.entry_status
        d['eligible'] = self.eligible
        d['sensors'] = self.sensors
        d['values'] = self.values

        return d


def to_route_record(record_dict):
    record = RouteRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)

    return record


def compute_route_length(config):
    route_length = 0.0
    previous_location = None

    for transform, _ in config.route:
        location = transform.location
        if previous_location:
            dist_vec = location - previous_location
            route_length += dist_vec.length()
        previous_location = location

    return route_length



class StatisticsManager(object):

    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self, endpoint, debug_endpoint):
        self._scenario = None
        self._route_length = 0
        self._total_routes = 0
        self._results = Results()
        self._endpoint = endpoint
        self._debug_endpoint = debug_endpoint

    def add_file_records(self, endpoint):
        """Reads a file and saves its records onto the statistics manager"""
        data = fetch_dict(endpoint)

        if data:
            route_records = dictor(data, '_checkpoint.records')
            if route_records:
                for record in data['_checkpoint']['records']:
                    self._results.checkpoint.records.append(to_route_record(record))

    def clear_records(self):
        """Cleanes up the file"""
        if not self._endpoint.startswith(('http:', 'https:', 'ftp:')):
            with open(self._endpoint, 'w') as fd:
                fd.truncate(0)

    def sort_records(self):
        """Sorts the route records according to their route id (This being i.e RouteScenario0_rep0)"""
        self._results.checkpoint.records.sort(key=lambda x: (
            int(x.route_id.split('_')[1]),
            int(x.route_id.split('_rep')[-1])
        ))

    def write_live_results(self, index, ego_speed, ego_control, ego_location):
        """Writes live results"""
        route_record = self._results.checkpoint.records[index]

        all_events = []
        if self._scenario:
            for node in self._scenario.get_criteria():
                all_events.extend(node.events)

        all_events.sort(key=lambda e: e.get_frame(), reverse=True)

        with open(self._debug_endpoint, 'w') as f:
            f.write("Route id: {}\n\n"
                    "Scores:\n"
                    "    Driving score:      {:.3f}\n"
                    "    Route completion:   {:.3f}\n"
                    "    Infraction penalty: {:.3f}\n\n"
                    "    Route length:    {:.3f}\n"
                    "    Game duration:   {:.3f}\n"
                    "    System duration: {:.3f}\n\n"
                    "Ego:\n"
                    "    Throttle:           {:.3f}\n"
                    "    Brake:              {:.3f}\n"
                    "    Steer:              {:.3f}\n\n"
                    "    Speed:           {:.3f} km/h\n\n"
                    "    Location:           ({:.3f} {:.3f} {:.3f})\n\n"
                    "Total infractions: {}\n"
                    "Last 5 infractions:\n".format(
                        route_record.route_id,
                        route_record.scores["Driving score"],
                        route_record.scores["Route completion"],
                        route_record.scores["Infraction penalty"],
                        route_record.meta["Route length"],
                        route_record.meta["Game duration"],
                        route_record.meta["System duration"],
                        ego_control.throttle,
                        ego_control.brake,
                        ego_control.steer,
                        ego_speed * 3.6,
                        ego_location.x,
                        ego_location.y,
                        ego_location.z,
                        route_record.num_infractions
                    )
                )
            for e in all_events[:5]:
                # Prevent showing the ROUTE_COMPLETION event.
                event_type = e.get_type()
                if event_type == TrafficEventType.ROUTE_COMPLETION:
                    continue
                string = "    " + str(e.get_type()).replace("TrafficEventType.", "")
                if event_type in PENALTY_VALUE_DICT:
                    string += " (penalty: " + str(PENALTY_VALUE_DICT[event_type]) + ")\n"
                elif event_type in PENALTY_PERC_DICT:
                    string += " (value: " + str(round(e.get_dict()['percentage'], 3)) + "%)\n"

                f.write(string)

    def save_sensors(self, sensors):
        self._results.sensors = sensors

    def save_entry_status(self, entry_status):
        if entry_status not in ENTRY_STATUS_VALUES:
            raise ValueError("Found an invalid value for 'entry_status'")
        self._results.entry_status = entry_status
        self._results.eligible = ELIGIBLE_VALUES[entry_status]

    def save_progress(self, route_index, total_routes):
        self._results.checkpoint.progress = [route_index, total_routes]
        self._total_routes = total_routes

    def create_route_data(self, route_id, index):
        """
        Creates the basic route data.
        This is done at the beginning to ensure the data is saved, even if a crash occurs
        """
        route_record = RouteRecord()
        route_record.route_id = route_id

        # Check if we have to overwrite an element (when resuming), or create a new one
        route_records = self._results.checkpoint.records
        if index < len(route_records):
            self._results.checkpoint.records[index] = route_record
        else:
            self._results.checkpoint.records.append(route_record)

    def set_scenario(self, config, scenario):
        """Sets the scenario from which the statistics will be taken"""
        self._scenario = scenario
        self._route_length = round(compute_route_length(config), ROUND_DIGITS)

    def remove_scenario(self):
        """Removes the scenario"""
        self._scenario = None
        self._route_length = 0

    def compute_route_statistics(self, config, duration_time_system=-1, duration_time_game=-1, failure_message=""):
        """
        Compute the current statistics by evaluating all relevant scenario criteria.
        Failure message will not be empty if an external source has stopped the simulations (i.e simulation crash).
        For the rest of the cases, it will be filled by this function depending on the criteria.
        """
        def set_infraction_message():
            infraction_name = PENALTY_NAME_DICT[event.get_type()]
            route_record.infractions[infraction_name].append(event.get_message())

        def set_score_penalty(score_penalty):
            event_value = event.get_dict()['percentage']
            penalty_value, penalty_type = PENALTY_PERC_DICT[event.get_type()]
            if penalty_type == "decreases":
                score_penalty *= (1 - (1 - penalty_value) * (1 - event_value / 100))
            elif penalty_type == "increases":
                score_penalty *= (1 - (1 - penalty_value) * event_value / 100)
            else:
                raise ValueError("Found a criteria with an unknown penalty type")
            return score_penalty

        index = config.index
        route_record = self._results.checkpoint.records[index]

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0
        for event_name in PENALTY_NAME_DICT.values():
            route_record.infractions[event_name] = []

        # Update the route meta
        route_record.meta['Route length'] = self._route_length
        route_record.meta['Game duration'] = round(duration_time_game, ROUND_DIGITS)
        route_record.meta['System duration'] = round(duration_time_system, ROUND_DIGITS)

        # Update the route infractions
        if self._scenario:
            if self._scenario.timeout_node.timeout:
                route_record.infractions['Route timeouts'].append('Route timeout.')
                failure_message = "Agent timed out"

            for node in self._scenario.get_criteria():
                for event in node.events:
                    # Traffic events that substract a set amount of points
                    if event.get_type() in PENALTY_VALUE_DICT:
                        score_penalty *= PENALTY_VALUE_DICT[event.get_type()]
                        set_infraction_message()

                    # Traffic events that substract a varying amount of points
                    elif event.get_type() in PENALTY_PERC_DICT:
                        score_penalty = set_score_penalty(score_penalty)
                        set_infraction_message()

                    # Traffic events that stop the simulation
                    elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                        failure_message = "Agent deviated from the route"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                        failure_message = "Agent got blocked"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                        score_route = event.get_dict()['route_completed']
                        target_reached = score_route >= 100

        # Update route scores
        route_record.scores['Route completion'] = round(score_route, ROUND_DIGITS_SCORE)
        route_record.scores['Infraction penalty'] = round(score_penalty, ROUND_DIGITS_SCORE)
        route_record.scores['Driving score'] = round(max(score_route * score_penalty, 0.0), ROUND_DIGITS_SCORE)

        # Update result
        route_record.num_infractions = sum([len(route_record.infractions[key]) for key in route_record.infractions])

        if target_reached:
            route_record.result = 'Completed' if route_record.num_infractions > 0 else 'Perfect'
        else:
            route_record.result = 'Failed'
            if failure_message:
                route_record.result += ' - ' + failure_message

        # Add the new data, or overwrite a previous result (happens when resuming the simulation)
        record_len = len(self._results.checkpoint.records)
        if index == record_len:
            self._results.checkpoint.records.append(route_record)
        elif index < record_len:
            self._results.checkpoint.records[index] = route_record
        else:
            raise ValueError("Not enough entries in the route record")

    def compute_global_statistics(self):
        """Computes and saves the global statistics of the routes"""
        def get_infractions_value(route_record, key):
            # Special case for the % based criteria. Extract the meters from the message. Very ugly, but it works
            if key == PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                if not route_record.infractions[key]:
                    return 0
                return float(route_record.infractions[key][0].split(" ")[8])/1000

            return len(route_record.infractions[key])

        global_record = GlobalRecord()
        global_result = global_record.result

        route_records = self._results.checkpoint.records

        # Calculate the score's means and result
        for route_record in route_records:

            global_record.scores_mean['Route completion'] += route_record.scores['Route completion'] / self._total_routes
            global_record.scores_mean['Infraction penalty'] += route_record.scores['Infraction penalty'] / self._total_routes
            global_record.scores_mean['Driving score'] += route_record.scores['Driving score'] / self._total_routes

            # Downgrade the global result if need be ('Perfect' -> 'Completed' -> 'Failed'), and record the failed routes
            route_result = 'Failed' if 'Failed' in route_record.result else route_record.result
            if route_result == 'Failed':
                global_record.meta['exceptions'][route_record.route_id] = route_record.result
                global_result = route_result
            elif global_result == 'Perfect' and route_result != 'Perfect':
                global_result = route_result

        global_record.result = global_result

        # Calculate the score's standard deviation
        if self._total_routes == 1:
            for key in global_record.scores_std_dev:
                global_record.scores_std_dev[key] = 'NaN'
        else:
            for route_record in route_records:
                for key in global_record.scores_std_dev:
                    diff = route_record.scores[key] - global_record.scores_mean[key]
                    global_record.scores_std_dev[key] += math.pow(diff, 2)

            for key in global_record.scores_std_dev:
                value = round(math.sqrt(global_record.scores_std_dev[key] / float(self._total_routes - 1)), ROUND_DIGITS)
                global_record.scores_std_dev[key] = value

        # Calculate the number of infractions per km
        km_driven = 0
        for route_record in route_records:
            km_driven += route_record.meta['Route length'] / 1000 * route_record.scores['Route completion'] / 100
            for key in global_record.infractions_per_km:
                global_record.infractions_per_km[key] += get_infractions_value(route_record, key)
        km_driven = max(km_driven, 0.001)

        for key in global_record.infractions_per_km:
            # Special case for the % based criteria.
            if key != PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                global_record.infractions_per_km[key] /= km_driven
            global_record.infractions_per_km[key] = round(global_record.infractions_per_km[key], ROUND_DIGITS)

        # Save the global records
        self._results.checkpoint.global_record = global_record

        # Change the values
        self._results.values = {}
        for key, item in global_record.scores_mean.items():
            self._results.values[key] = item
        for key, item in global_record.infractions_per_km.items():
            self._results.values[key] = item

        # Change the entry status and eligible
        entry_status = 'Finished'
        for route_record in route_records:
            route_status = route_record.result
            if 'Simulation crashed' in route_status:
                entry_status = 'Crashed'
            elif "Agent's sensors were invalid" in route_status:
                entry_status = 'Rejected'

        self.save_entry_status(entry_status)

    def validate_and_write_statistics(self, sensors_initialized, crashed):
        """
        Makes sure that all the relevant data is there.
        Changes the 'entry status' to 'Invalid' if this isn't the case
        """
        error_message = ""
        if sensors_initialized and not self._results.sensors:
            error_message = "Missing 'sensors' data"

        elif not self._results.values:
            error_message = "Missing 'values' data"

        elif self._results.entry_status == 'Started':
            error_message = "'entry_status' has the 'Started' value"

        else:
            global_records = self._results.checkpoint.global_record
            progress = self._results.checkpoint.progress
            route_records = self._results.checkpoint.records

            if not global_records:
                error_message = "Missing 'global_records' data"

            elif not progress:
                error_message = "Missing 'progress' data"

            elif not crashed and (progress[0] != progress[1] or progress[0] != len(route_records)):
                error_message = "'progress' data doesn't match its expected value"

            else:
                for record in route_records:
                    if record.result == 'Started':
                        error_message = "Found a route record with missing data"
                        break

        if error_message:
            print("\n\033[91mThe statistics are badly formed. Setting their status to 'Invalid':")
            print("> {}\033[0m\n".format(error_message))

            self.save_entry_status('Invalid')

        save_dict(self._endpoint, self._results.to_json())
