#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains the result gatherer and write for CARLA scenarios.
It shall be used from the ScenarioManager only.
"""

from __future__ import print_function

import time
from collections import OrderedDict
from tabulate import tabulate


COLORED_STATUS = {
    "FAILURE": '\033[91mFAILURE\033[0m',
    "SUCCESS": '\033[92mSUCCESS\033[0m',
    "ACCEPTABLE": '\033[93mACCEPTABLE\033[0m',
}

STATUS_PRIORITY = {
    "FAILURE": 0,
    "ACCEPTABLE": 1,
    "SUCCESS": 2,
}  # Lower number is higher priority


class ResultOutputProvider(object):

    """
    This module contains the _result gatherer and write for CARLA scenarios.
    It shall be used from the ScenarioManager only.
    """

    def __init__(self, data, global_result):
        """
        - data contains all scenario-related information
        - global_result is overall pass/fail info
        """
        self._data = data
        self._global_result = global_result

        self._start_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(self._data.start_system_time))
        self._end_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(self._data.end_system_time))

        print(self.create_output_text())

    def create_output_text(self):
        """
        Creates the output message
        """

        # Create the title
        output = "\n"
        output += "\033[1m========= Results of {} (repetition {}) ------ {} \033[1m=========\033[0m\n".format(
            self._data.scenario_tree.name, self._data.repetition_number, self._global_result)
        output += "\n"

        # Simulation part
        system_time = round(self._data.scenario_duration_system, 2)
        game_time = round(self._data.scenario_duration_game, 2)
        ratio = round(self._data.scenario_duration_game / self._data.scenario_duration_system, 3)

        list_statistics = [["Start Time", "{}".format(self._start_time)]]
        list_statistics.extend([["End Time", "{}".format(self._end_time)]])
        list_statistics.extend([["Duration (System Time)", "{}s".format(system_time)]])
        list_statistics.extend([["Duration (Game Time)", "{}s".format(game_time)]])
        list_statistics.extend([["Ratio (System Time / Game Time)", "{}".format(ratio)]])

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n\n"

        # Criteria part
        header = ['Criterion', 'Result', 'Value']
        list_statistics = [header]
        criteria_data = OrderedDict()

        scenario_times = {}
        for criterion in self._data.scenario.get_criteria():

            # Ignore the value of the YieldToEV test is not initialized.
            name = criterion.name
            if name == "YieldToEmergencyVehicleTest" and not criterion.initialized:
                continue

            # If two criterion have the same name, their results are shown as one.
            # Criteria with a "%" based value get their mean shown, while the rest show the sum.
            if name in criteria_data:
                result = criterion.test_status
                if STATUS_PRIORITY[result] < STATUS_PRIORITY[criteria_data[name]['result']]:
                    criteria_data[name]['result'] = result
                if criterion.units == "%":
                    if name in scenario_times:
                        scenario_times[name] += 1
                    else:
                        scenario_times[name] = 1
                criteria_data[name]['actual_value'] += criterion.actual_value

            else:
                criteria_data[name] = {
                    'result': criterion.test_status,
                    'actual_value': criterion.actual_value,
                    'units': criterion.units
                }

        # Get the mean value
        for name, times in list(scenario_times.items()):
            criteria_data[name]['actual_value'] /= times
            criteria_data[name]['actual_value'] = round(criteria_data[name]['actual_value'], 2)

        for criterion_name in criteria_data:
            criterion = criteria_data[criterion_name]

            result = criterion['result']
            if result in COLORED_STATUS:
                result = COLORED_STATUS[result]

            if criterion['units'] is None:
                actual_value = ""
            else:
                actual_value = str(criterion['actual_value']) + " " + criterion['units']

            list_statistics.extend([[criterion_name, result, actual_value]])

        # Timeout
        name = "Timeout"

        actual_value = self._data.scenario_duration_game

        if self._data.scenario_duration_game < self._data.scenario.timeout:
            result = '\033[92m'+'SUCCESS'+'\033[0m'
        else:
            result = '\033[91m'+'FAILURE'+'\033[0m'

        list_statistics.extend([[name, result, '']])

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n"

        return output
