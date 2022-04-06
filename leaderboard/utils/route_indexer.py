from collections import OrderedDict
from multiprocessing.sharedctypes import Value
from dictor import dictor

import copy

from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.checkpoint_tools import fetch_dict


class RouteIndexer():
    def __init__(self, routes_file, repetitions, routes_subset):
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.index = 0

        route_configurations = RouteParser.parse_routes_file(routes_file, routes_subset)
        self.total = len(route_configurations) * repetitions

        for i, config in enumerate(route_configurations):
            for repetition in range(repetitions):
                config.index = i * repetitions + repetition
                config.repetition_index = repetition
                self._configs_dict['{}.{}'.format(config.name, repetition)] = copy.copy(config)

        self._configs_list = list(self._configs_dict.values())


    def peek(self):
        return not (self.index >= len(self._configs_list))

    def next(self):
        if self.index >= len(self._configs_list):
            return None

        config = self._configs_list[self.index]
        self.index += 1

        return config

    def resume(self, endpoint):
        data = fetch_dict(endpoint)

        if data:
            checkpoint_dict = dictor(data, '_checkpoint')
            if checkpoint_dict and 'progress' in checkpoint_dict:
                progress = checkpoint_dict['progress']
                current_route = progress[0] if progress else 0
                if current_route <= self.total:

                    self.index = current_route
                    return
 
        print('Problem reading checkpoint. Starting from the first route')

