#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
NPC agent: drives the ego with the CARLA PythonAPI BasicAgent along the dense
scenario route. It is a testing baseline, not a challenge submission: it reads
the full route from the CarlaDataProvider (which real submissions cannot do)
and follows it waypoint by waypoint, braking for vehicles and traffic lights.
Useful to validate that a route and its scenarios are drivable at all.
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return 'NpcAgent'


class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle with the BasicAgent
    """

    _agent = None

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS
        self._agent = None

    def sensors(self):
        """
        Define the sensor suite required by the agent
        """
        return [
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
        ]

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        if self._agent is None:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
                if actor.attributes.get('role_name') == 'hero':
                    hero_actor = actor
                    break

            if hero_actor is None:
                return carla.VehicleControl()

            self._agent = BasicAgent(hero_actor, 30)

            # Follow the dense scenario route directly: re-tracing a plan is
            # fragile inside junctions, where points can snap to overlapping
            # connector roads of other maneuvers.
            route = CarlaDataProvider.get_ego_route()
            wmap = CarlaDataProvider.get_map()
            plan = []
            for transform, road_option in route:
                wp = wmap.get_waypoint(transform.location)
                plan.append((wp, road_option))
            self._agent.set_global_plan(plan)

            return carla.VehicleControl()

        return self._agent.run_step()

    def destroy(self):
        """
        Cleanup
        """
        self._agent = None
