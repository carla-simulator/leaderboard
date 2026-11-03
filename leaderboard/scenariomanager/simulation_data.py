#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module holds the connection to the simulator and what is derived from it
and too expensive to rebuild: the map and the global route planner.
"""

from numpy import random

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


class SimulationData(object):

    """
    Holds the client, the world, its map and global route planner, and the random
    state, all shared by every scenario, behaviour and criterion of a route.

    Actor state is not cached here: read it from the Python API instead.
    """

    _client = None              # type: carla.Client
    _world = None               # type: carla.World
    _map = None                 # type: carla.Map
    _do_tick = False            # type: bool
    _ego_vehicle_route = None
    _traffic_manager_port = 8000
    _random_seed = 2000
    _rng = random.RandomState(_random_seed)
    _grp = None                 # type: GlobalRoutePlanner

    @staticmethod
    def set_client(client):
        # type: (carla.Client) -> None
        """
        Set the CARLA client
        """
        SimulationData._client = client

    @staticmethod
    def get_client():
        """
        Get the CARLA client
        """
        return SimulationData._client

    @staticmethod
    def prepare_world(world):
        # type: (carla.World) -> None
        """
        Store the world and build everything derived from it
        """
        SimulationData._world = world
        SimulationData._do_tick = world.get_settings().synchronous_mode
        SimulationData._map = world.get_map()
        SimulationData._grp = GlobalRoutePlanner(SimulationData._map, 2.0)

    @staticmethod
    def get_world():
        """
        Return world
        """
        return SimulationData._world

    @staticmethod
    def get_map(world=None):
        # type: (carla.World | None) -> carla.Map
        """
        Get the current map
        """
        if SimulationData._map is None:
            if world is None:
                if SimulationData._world is None:
                    raise ValueError("class member 'world' not initialized yet")
                else:
                    SimulationData._map = SimulationData._world.get_map()
            else:
                SimulationData._map = world.get_map()

        return SimulationData._map

    @staticmethod
    def get_rng():
        """
        @return the random state shared by everything that draws from it
        """
        return SimulationData._rng

    @staticmethod
    def get_global_route_planner():
        """
        @return the global route planner
        """
        return SimulationData._grp
    
    @staticmethod
    def set_ego_route(route):
        """
        set the ego route
        """
        SimulationData._ego_vehicle_route = route

    @staticmethod
    def get_ego_route():
        """
        @return the ego route
        """
        return SimulationData._ego_vehicle_route

    @staticmethod
    def set_do_tick(flag):
        # type: (bool) -> None
        """
        Set whether this client advances the world itself. False either because the
        world is asynchronous, or because the manager is already ticking it
        """
        SimulationData._do_tick = flag

    @staticmethod
    def tick_world():
        """
        Advance the world, or wait for whoever is advancing it
        """
        if SimulationData._do_tick:
            SimulationData._world.tick()
        else:
            SimulationData._world.wait_for_tick()

    @staticmethod
    def set_traffic_manager_port(tm_port):
        # type: (int) -> None
        """
        Set the port to use for the traffic manager.
        """
        SimulationData._traffic_manager_port = tm_port

    @staticmethod
    def get_traffic_manager_port():
        """
        Get the port of the traffic manager.
        """
        return SimulationData._traffic_manager_port


    @staticmethod
    def cleanup():
        """
        Reset every entry, so that the next route starts from a clean state
        """
        SimulationData._map = None
        SimulationData._world = None
        SimulationData._do_tick = False
        SimulationData._ego_vehicle_route = None
        SimulationData._client = None
        SimulationData._rng = random.RandomState(SimulationData._random_seed)
        SimulationData._grp = None
