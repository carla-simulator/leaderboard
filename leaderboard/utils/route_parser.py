#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module used to parse all the route and scenario configuration parameters.
"""
from collections import OrderedDict
import json
import math
import xml.etree.ElementTree as ET

import carla
from agents.navigation.local_planner import RoadOption
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData

# Threshold to say if a scenarios trigger position is part of the route
DIST_THRESHOLD = 2.0
ANGLE_THRESHOLD = 10


def convert_dict_to_transform(dicti):
    """Convert a dict to a CARLA transform"""
    return carla.Transform(
        carla.Location(
            float(dicti['x']),
            float(dicti['y']),
            float(dicti['z'])
        ),
        carla.Rotation(
            roll=0.0,
            pitch=0.0,
            yaw=float(dicti['yaw'])
        )
    )


def convert_elem_to_transform(elem):
    """Convert an ElementTree.Element to a CARLA transform"""
    return carla.Transform(
        carla.Location(
            float(elem.attrib.get('x')),
            float(elem.attrib.get('y')),
            float(elem.attrib.get('z'))
        ),
        carla.Rotation(
            roll=0.0,
            pitch=0.0,
            yaw=float(elem.attrib.get('yaw'))
        )
    )


class RouteParser(object):

    """
    Pure static class used to parse all the route and scenario configuration parameters.
    """

    @staticmethod
    def parse_routes_file(route_filename, single_route_id=''):
        """
        Returns a list of route configuration elements.
        :param route_filename: the path to a set of routes.
        :param single_route: If set, only this route shall be returned
        :return: List of dicts containing the waypoints, id and town of the routes
        """

        route_configs = []
        tree = ET.parse(route_filename)
        for route in tree.iter("route"):

            route_id = route.attrib['id']
            if single_route_id and route_id != single_route_id:
                continue

            route_config = RouteScenarioConfiguration()
            route_config.town = route.attrib['town']
            route_config.name = "RouteScenario_{}".format(route_id)
            route_config.weather = RouteParser.parse_weather(route)

            # The list of carla.Location that serve as keypoints on this route
            positions = []
            for position in route.find('waypoints').iter('position'):
                positions.append(carla.Location(x=float(position.attrib['x']),
                                                y=float(position.attrib['y']),
                                                z=float(position.attrib['z'])))
            route_config.keypoints = positions

            # The list of ScenarioConfigurations that store the scenario's data
            scenario_configs = []
            for scenario in route.find('scenarios').iter('scenario'):
                scenario_config = ScenarioConfiguration()
                scenario_config.name = scenario.attrib.get('name')
                scenario_config.type = scenario.attrib.get('type')

                for elem in scenario.getchildren():
                    if elem.tag == 'trigger_point':
                        scenario_config.trigger_points.append(convert_elem_to_transform(elem))
                    elif elem.tag == 'other_actor':
                        scenario_config.other_actors.append(ActorConfigurationData.parse_from_node(elem, 'scenario'))
                    else:
                        scenario_config.other_parameters[elem.tag] = elem.attrib

                scenario_configs.append(scenario_config)
            route_config.scenario_configs = scenario_configs

            route_configs.append(route_config)

        return route_configs

    @staticmethod
    def parse_weather(route):
        """
        Returns a carla.WeatherParameters with the corresponding weather for that route. If the route
        has no weather attribute, the default one is triggered.
        """

        route_weather = route.find("weather")
        if route_weather is None:
            weather = carla.WeatherParameters(sun_altitude_angle=70, cloudiness=30)

        else:
            weather = carla.WeatherParameters()
            for weather_attrib in route.iter("weather"):

                if 'cloudiness' in weather_attrib.attrib:
                    weather.cloudiness = float(weather_attrib.attrib['cloudiness']) 
                if 'precipitation' in weather_attrib.attrib:
                    weather.precipitation = float(weather_attrib.attrib['precipitation'])
                if 'precipitation_deposits' in weather_attrib.attrib:
                    weather.precipitation_deposits =float(weather_attrib.attrib['precipitation_deposits'])
                if 'wind_intensity' in weather_attrib.attrib:
                    weather.wind_intensity = float(weather_attrib.attrib['wind_intensity'])
                if 'sun_azimuth_angle' in weather_attrib.attrib:
                    weather.sun_azimuth_angle = float(weather_attrib.attrib['sun_azimuth_angle'])
                if 'sun_altitude_angle' in weather_attrib.attrib:
                    weather.sun_altitude_angle = float(weather_attrib.attrib['sun_altitude_angle'])
                if 'wetness' in weather_attrib.attrib:
                    weather.wetness = float(weather_attrib.attrib['wetness'])
                if 'fog_distance' in weather_attrib.attrib:
                    weather.fog_distance = float(weather_attrib.attrib['fog_distance'])
                if 'fog_density' in weather_attrib.attrib:
                    weather.fog_density = float(weather_attrib.attrib['fog_density'])
                if 'fog_falloff' in weather_attrib.attrib:
                    weather.fog_falloff = float(weather_attrib.attrib['fog_falloff'])
                if 'scattering_intensity' in weather_attrib.attrib:
                    weather.scattering_intensity = float(weather_attrib.attrib['scattering_intensity'])
                if 'mie_scattering_scale' in weather_attrib.attrib:
                    weather.mie_scattering_scale = float(weather_attrib.attrib['mie_scattering_scale'])
                if 'rayleigh_scattering_scale' in weather_attrib.attrib:
                    weather.rayleigh_scattering_scale = float(weather_attrib.attrib['rayleigh_scattering_scale'])

        return weather

    @staticmethod
    def is_scenario_at_route(trigger_transform, route):
        """
        Check if the scenario is affecting the route.
        This is true if the trigger position is very close to any route point
        """
        def is_trigger_close(trigger_transform, route_transform):
            """Check if the two transforms are similar"""
            dx = trigger_transform.location.x - route_transform.location.x
            dy = trigger_transform.location.y - route_transform.location.y
            dz = trigger_transform.location.z - route_transform.location.z
            dpos = math.sqrt(dx * dx + dy * dy)

            dyaw = (float(trigger_transform.rotation.yaw) - route_transform.rotation.yaw) % 360

            return dz < DIST_THRESHOLD and dpos < DIST_THRESHOLD \
                and (dyaw < ANGLE_THRESHOLD or dyaw > (360 - ANGLE_THRESHOLD))

        for position, [route_transform, _] in enumerate(route):
            if is_trigger_close(trigger_transform, route_transform):
                return position

        return None

    @staticmethod
    def get_scenario_subtype(scenario, route):
        """
        Some scenarios have subtypes depending on the route trajectory,
        even being invalid if there isn't a valid one. As an example,
        some scenarios need the route to turn in a specific direction,
        and if this isn't the case, the scenario should not be considered valid.
        This is currently only used for validity purposes.
        :param scenario: the scenario name
        :param route: route starting at the triggering point of the scenario
        :return: tag representing this subtype
        """

        def is_junction_option(option):
            """Whether or not an option is part of a junction"""
            if option in (RoadOption.LANEFOLLOW, RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                return False
            return True

        subtype = None

        if scenario == 'Scenario4':  # Only if the route turns
            for _, option in route:
                if is_junction_option(option):
                    if option == RoadOption.LEFT:
                        subtype = 'S4left'
                    elif option == RoadOption.RIGHT:
                        subtype = 'S4right'
                    break  # Avoid checking all of them
        elif scenario == 'Scenario7':
            for _, option in route:
                if is_junction_option(option):
                    if RoadOption.STRAIGHT == option:
                        subtype = 'S7opposite'
                    break
        elif scenario == 'Scenario8':  # Only if the route turns left
            for _, option in route:
                if is_junction_option(option):
                    if option == RoadOption.LEFT:
                        subtype = 'S8left'
                    break
        elif scenario == 'Scenario9':  # Only if the route turns right
            for _, option in route:
                if is_junction_option(option):
                    if option == RoadOption.RIGHT:
                        subtype = 'S9right'
                    break
        else:
            subtype = 'valid'

        return subtype
