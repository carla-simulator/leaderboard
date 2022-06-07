#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
from lxml import etree
import carla

SCENARIO_TYPES ={
    # Old scenarios
    "ControlLoss": [
        ["trigger_point", "transform"]
    ],
    "FollowLeadingVehicleRoute": [
        ["trigger_point", "transform"]
    ],
    "DynamicObjectCrossing": [
        ["trigger_point", "transform"],
        ["distance", "value"]
    ],
    "VehicleTurningRoute": [
        ["trigger_point", "transform"]
    ],
    "SignalizedJunctionLeftTurn": [
        ["trigger_point", "transform"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "SignalizedJunctionRightTurn": [
        ["trigger_point", "transform"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "OppositeVehicleRunningRedLight": [
        ["trigger_point", "transform"],
        ["direction", "choice"],
        ["adversary_speed", "value"],
    ],
    # Old junction scenarios, non signalized version
    "NonSignalizedJunctionLeftTurn": [
        ["trigger_point", "transform"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "NonSignalizedJunctionRightTurn": [
        ["trigger_point", "transform"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "OppositeVehicleTakingPriority": [
        ["trigger_point", "transform"],
        ["direction", "choice"],
        ["adversary_speed", "value"],
    ],
    "ParkingCrossingPedestrian": [
        ["trigger_point", "transform"],
        ["distance", "value"],
        ["direction", "choice"],
    ],
    # Actor flows
    "EnterActorFlow": [
        ["trigger_point", "transform"],
        ["start_actor_flow", "location"],
        ["end_actor_flow", "location"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "InterurbanActorFlow": [
        ["trigger_point", "transform"],
        ["start_actor_flow", "location"],
        ["end_actor_flow", "location"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "InterurbanAdvancedActorFlow": [
        ["trigger_point", "transform"],
        ["start_actor_flow", "location"],
        ["end_actor_flow", "location"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "HighwayExit": [
        ["trigger_point", "transform"],
        ["start_actor_flow", "location"],
        ["end_actor_flow", "location"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "MergerIntoSlowTraffic": [
        ["trigger_point", "transform"],
        ["start_actor_flow", "location"],
        ["end_actor_flow", "location"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
    ],
    "CrossingBycicleFlow": [
        ["trigger_point", "transform"],
        ["start_actor_flow", "location"],
        ["flow_speed", "value"],
        ["source_dist_interval", "interval"],
        ["green_light_delay", "value"],
    ],
    # Route obstacles
    "ConstructionObstacle": [
        ["trigger_point", "transform"],
        ["distance", "value"],
    ],
    "ConstructionObstacleTwoWays": [
        ["trigger_point", "transform"],
        ["distance", "value"],
        ["frequency", "value"],
    ],
    "Accident": [
        ["trigger_point", "transform"],
        ["distance", "value"],
        ["direction", "value"],
    ],
    "AccidentTwoWays": [
        ["trigger_point", "transform"],
        ["distance", "value"],
        ["frequency", "value"],
    ],
    "ParkedObstacle": [
        ["trigger_point", "transform"],
        ["distance", "value"],
    ],
    "ParkedObstacleTwoWays": [
        ["trigger_point", "transform"],
        ["distance", "value"],
        ["frequency", "value"],
    ],
    "VehicleOpensDoor": [
        ["trigger_point", "transform"],
        ["distance", "value"],
    ],
    "VehicleOpensDoorTwoWays": [
        ["trigger_point", "transform"],
        ["distance", "value"],
        ["frequency", "value"],
    ],
    # Cut ins
    "HighwayCutIn": [
        ["trigger_point", "transform"],
        ["other_actor_location", "location"],
    ],
    "ParkingCutIn": [
        ["trigger_point", "transform"],
        ["direction", "choice"],
    ],
    # Special ones
    "ParkingExit": [
        ["trigger_point", "transform"],
        ["direction", "choice"],
        ["front_vehicle_distance", "value"],
        ["behind_vehicle_distance", "value"],
    ],
    "BackgroundActivityParametrizer": [
        ["trigger_point", "transform"],
        ["num_front_vehicles", "value"],
        ["num_back_vehicles", "value"],
        ["road_spawn_dist", "value"],
        ["opposite_source_dist", "value"],
        ["opposite_max_actors", "value"],
        ["opposite_spawn_dist", "value"],
        ["opposite_active", "bool"],
        ["junction_source_dist", "value"],
        ["junction_max_actors", "value"],
        ["junction_spawn_dist", "value"],
        ["junction_source_perc", "value"],
    ],
    "PriorityAtJunction": [
        ["trigger_point", "transform"],
    ],

    # Yield to EV
    # Pedestrian Crossing
    # HighwayStaticCutIn
    # BlockedIntersection
    # HazardMovingAtSideLane
}

def show_saved_scenarios(filename, route_id, world):
    def convert_elem_to_location(elem):
        """Convert an ElementTree.Element to a CARLA Location"""
        return carla.Location(float(elem.attrib.get('x')), float(elem.attrib.get('y')), float(elem.attrib.get('z')))

    tree = etree.parse(filename)
    root = tree.getroot()

    for route in root.iter("route"):
        if route.attrib['id'] != route_id:
            continue

        for scenario in route.find('scenarios').iter('scenario'):
            name = scenario.attrib.get('name')
            trigger_location = convert_elem_to_location(scenario.find('trigger_point'))
            world.debug.draw_point(trigger_location + carla.Location(z=0.2), size=0.3, color=carla.Color(125, 0, 0))
            world.debug.draw_string(trigger_location + carla.Location(z=0.5), name, True, color=carla.Color(0, 0 , 125), life_time=100000)

def get_scenario_type():
    while True:
        scen_type = input("\033[1m> Specify the scenario type \033[0m")
        if scen_type == "Re":
            restart = True
            break
        if scen_type not in list(SCENARIO_TYPES):
            print(f"\033[1m\033[93mScenario type '{scen_type}' doesn't match any of the know scenarios\033[0m")
        else:
            break
    return scen_type

def get_attributes_data(scen_type, tmap, world, spectator):
    attribute_list = SCENARIO_TYPES[scen_type]
    scenario_attributes = []
    for i, attribute in enumerate(attribute_list):
        a_name, a_type = attribute
        if a_type == 'transform':
            a_data = get_transform_data(a_name, scen_type, tmap, world, spectator)
        elif a_type == 'location':
            a_data = get_location_data(a_name, scen_type, tmap, world, spectator)
        elif a_type in ('value', 'choice', 'bool'):
            a_data = get_value_data(a_name)
        elif a_type == 'interval':
            a_data = get_interval_data(a_name)
        else:
            raise ValueError("Unknown attribute type")

        if a_data:  # Ignore the attributes that use default values
            scenario_attributes.append([a_name, a_type, a_data])
    return scenario_attributes

def get_transform_data(a_name, scen_type, tmap, world, spectator):
    input(f"\033[1m> Get the '{a_name}' transform \033[0m")
    wp = tmap.get_waypoint(spectator.get_location())
    world.debug.draw_point(wp.transform.location + carla.Location(z=0.2), size=0.3, color=carla.Color(125, 0, 0))
    world.debug.draw_string(wp.transform.location + carla.Location(z=0.5), scen_type, True, color=carla.Color(0, 0 , 125), life_time=100000)
    return (
        str(round(wp.transform.location.x, 1)),
        str(round(wp.transform.location.y, 1)),
        str(round(wp.transform.location.z, 1)),
        str(round(wp.transform.rotation.yaw, 1))
    )

def get_location_data(a_name, tmap, world, spectator):
    input(f"\033[1m> Get the '{a_name}' location \033[0m")
    wp = tmap.get_waypoint(spectator.get_location())
    world.debug.draw_point(wp.transform.location + carla.Location(z=0.2), size=0.3, color=carla.Color(125, 0, 0))
    world.debug.draw_string(wp.transform.location + carla.Location(z=0.5), scen_type, True, color=carla.Color(0, 0 , 125), life_time=100000)
    return (
        str(round(wp.transform.location.x, 1)),
        str(round(wp.transform.location.y, 1)),
        str(round(wp.transform.location.z, 1))
    )

def get_value_data(a_name):
    value = input(f"\033[1m> Specify the '{a_name}' value \033[0m")
    return value

def get_interval_data(a_name):
    lower_value = input(f"\033[1m> Specify the '{a_name}' from \033[0m")
    upper_value = input(f"\033[1m> Specify the '{a_name}' from \033[0m{lower_value}\033[1m to \033[0m")
    return (lower_value, upper_value)

def print_scenario_data(scen_type, scen_attributes):
    print_dict = f"\n   scenario_type: {scen_type}\n"
    for print_attribute in scen_attributes:
        print_dict += f"   {print_attribute[0]}: {print_attribute[2]}\n"
    print(print_dict)

def save_scenario(filename, route_id, scenario_type, scenario_attributes):
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

    while True:
        save = input("\033[1m> Save the scenario? ('Yes' / 'No'): \033[0m")
        if save == "Yes":
            break
        elif save == "No":
            return
        else:
            print(f"\033[1m\033[93mThis is a 'Yes' or 'No' question, try again.\033[0m")

    tree = etree.parse(filename)
    root = tree.getroot()

    scenario_names = {}
    for scen_type in list(SCENARIO_TYPES):
        scenario_names[scen_type] = 1

    for route in root.iter("route"):
        if route.attrib['id'] != route_id:
            continue

        scenarios = route.find('scenarios')
        for scenario in scenarios.iter('scenario'):
            scen_type = scenario.attrib['type']
            scenario_names[scen_type] += 1

        number = scenario_names[scenario_type]
        new_scenario = etree.SubElement(scenarios, "scenario")

        new_scenario.set("name", scenario_type + "_" + str(number))
        new_scenario.set("type", scenario_type)

        for a_name, a_type, a_value in scenario_attributes:
            data = etree.SubElement(new_scenario, a_name)
            if a_type == 'transform':
                data.set("x", a_value[0])
                data.set("y", a_value[1])
                data.set("z", a_value[2])
                data.set("yaw", a_value[3])
            elif a_type == 'location':
                data.set("x", a_value[0])
                data.set("y", a_value[1])
                data.set("z", a_value[2])
            elif a_type in ('value', 'choice', 'bool'):
                data.set("value", a_value)
            elif a_type == 'interval':
                data.set("from", a_value[0])
                data.set("to", a_value[1])
        break

    # Prettify the xml. A bit of automatic indentation, a bit of manual one
    spaces = 3
    indent(root, spaces)
    tree.write(filename)

    with open(filename, 'r') as f:
        data = f.read()
    temp = data.replace("   </", "</")  # The 'indent' function fails for these cases

    weather_spaces = spaces*4*" "
    temp = temp.replace(" cloudiness", "\n" + weather_spaces + "cloudiness")
    temp = temp.replace(" wind_intensity", "\n" + weather_spaces + "wind_intensity")
    temp = temp.replace(" fog_density", "\n" + weather_spaces + "fog_density")
    new_data = temp.replace(" mie_scattering_scale", "\n" + weather_spaces + "mie_scattering_scale")

    with open(filename, 'w') as f:
        f.write(new_data)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', required=True, help='File at which to place the scenarios')
    argparser.add_argument('-r', '--route-id', required=True, help='Route id of the scenarios')
    args = argparser.parse_args()

    # Get the client
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # # Get the rest
    world = client.get_world()
    spectator = world.get_spectator()
    tmap = world.get_map()

    # Get the data already at the file
    show_saved_scenarios(args.file, args.route_id, world)

    print(" ------------------------------------------------------------ ")
    print(" |               Use Ctrl+C to stop the script              | ")
    print(" |            Any ongoing scenario will be ignored          | ")
    print(" |                                                          | ")
    print(" |   Transform and location parameters will automatically   | ")
    print(" |     get the closest waypoint to the spectator camera     | ")
    try:
        while True:
            print(" ------------------------------------------------------------ ")

            # Get the scenario type
            scen_type = get_scenario_type()

            # Get the attributes
            scen_attributes = get_attributes_data(scen_type, tmap, world, spectator)

            # Give feedback to the user
            print_scenario_data(scen_type, scen_attributes)

            # Save the data
            save_scenario(args.file, args.route_id, scen_type, scen_attributes)

    except KeyboardInterrupt as e:
        print("\n Detected a keyboard interruption, stopping the script ")

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
