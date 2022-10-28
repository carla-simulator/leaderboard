#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import argparse
from lxml import etree

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner

MAPS_LOCATIONS = {
    "Town01": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr",
    "Town01_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01_Opt.xodr",
    "Town02": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town02.xodr",
    "Town02_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town02_Opt.xodr",
    "Town03": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr",
    "Town03_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04_Opt.xodr",
    "Town04": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04.xodr",
    "Town04_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04_Opt.xodr",
    "Town05": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr",
    "Town05_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr",
    "Town06": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town06.xodr",
    "Town06_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town06_Opt.xodr",
    "Town07": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town07.xodr",
    "Town07_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town07_Opt.xodr",
    "Town10HD": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD.xodr",
    "Town10HD_Opt": "Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD_Opt.xodr",
    "Town11": "Unreal/CarlaUE4/Content/Carla/Maps/Town11/OpenDrive/Town11.xodr",
    "Town12": "Unreal/CarlaUE4/Content/Carla/Maps/Town12/OpenDrive/Town12.xodr",
}

def prettify_and_save_tree(filename, tree):
    def indent(elem, spaces=3, level=0):
        i = "\n" + level * spaces * " "
        j = "\n" + (level + 1) * spaces * " "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = j
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for subelem in elem:
                indent(subelem, spaces, level+1)
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
        return elem

    # Prettify the xml. A bit of automatic indentation, a bit of manual one
    spaces = 3
    indent(tree.getroot(), spaces)
    tree.write(filename)

    with open(filename, 'r') as f:
        data = f.read()
    temp = data.replace("   </", "</")  # The 'indent' function fails for these cases

    weather_spaces = spaces*4*" "
    temp = temp.replace(" cloudiness", "\n" + weather_spaces + "cloudiness")
    temp = temp.replace(" wind_intensity", "\n" + weather_spaces + "wind_intensity")
    new_data = temp.replace(" mie_scattering_scale", "\n" + weather_spaces + "mie_scattering_scale")

    with open(filename, 'w') as f:
        f.write(new_data)

def convert_elem_to_location(elem):
    """Convert an ElementTree.Element to a CARLA Location"""
    return carla.Location(float(elem.attrib.get('x')), float(elem.attrib.get('y')), float(elem.attrib.get('z')))

def get_scenario_route_position(trigger_location, route_wps):
    position = 0
    distance = float('inf')
    for i, (wp, _) in enumerate(route_wps):
        route_distance = wp.transform.location.distance(trigger_location)
        if route_distance < distance:
            distance = route_distance
            position = i
    return position

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', required=True, help="Route's file path")
    args = argparser.parse_args()

    tree = etree.parse(args.file)
    root = tree.getroot()

    route_wps = []
    prev_route_keypoint = None
    scenario_list = []
    scenario_names = {}

    prev_town = None
    grp = None

    for route in root.iter("route"):

        # Get the map data without using the client
        route_id = route.attrib['id']
        print(f"\033[1m> Parsing route '{route_id}'\033[0m")
        route_town = route.attrib['town']

        if route_town not in MAPS_LOCATIONS:
            print(f"Ignoring route '{route_id}' as it uses an unknown map '{route_town}")
            continue
        elif route_town != prev_town:
            full_name = os.environ["CARLA_ROOT"] + "/" + MAPS_LOCATIONS[route_town]
            with open(full_name, 'r') as f:
                map_contents = f.read()
            tmap = carla.Map(route_town, map_contents)
            grp = GlobalRoutePlanner(tmap, 5.0)
        prev_town = route_town

        # Extract all the route
        print("Extracting the route information")
        for position in route.find('waypoints').iter('position'):
            route_keypoint = convert_elem_to_location(position)
            if prev_route_keypoint:
                route_wps.extend(grp.trace_route(prev_route_keypoint, route_keypoint))
            prev_route_keypoint = route_keypoint

        # Extract all the scenarios
        print("Extracting the scenarios information")
        scenarios = route.find('scenarios')
        for scenario in scenarios.iter('scenario'):
            scenario_list.append(scenario)

        # Order the scenarios according to route position
        print("Ordering the scenarios")
        scenario_and_pos = []
        for scenario in scenario_list:
            trigger_location = convert_elem_to_location(scenario.find('trigger_point'))
            route_position = get_scenario_route_position(trigger_location, route_wps)
            scenario_and_pos.append([scenario, route_position])
        scenario_and_pos = sorted(scenario_and_pos, key=lambda x: x[1])

        # Update the scenarios xml (both 'name' and 'scenario' position inside 'scenarios')
        print("Updating the xml")
        for i, (scenario, _) in enumerate(scenario_and_pos):
            scen_type = scenario.attrib['type']
            if scen_type not in scenario_names:
                scenario_names[scen_type] = 1
            else:
                scenario_names[scen_type] += 1
            scenario.set("name", f"{scen_type}_{scenario_names[scen_type]}")
            scenario.set("order", str(i))

        scenarios[:] = sorted(scenarios, key=lambda child: int(child.get("order")))
        for scenario in scenarios.iter('scenario'):
            scenario.attrib.pop("order", None)

        prev_route_keypoint = None
        route_wps.clear()
        scenario_list.clear()
        scenario_names.clear()

    # Save the xml
    print(f"\033[1m> Saving...\033[0m")
    prettify_and_save_tree(args.file, tree)

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
