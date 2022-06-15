#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
from lxml import etree
import sys

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

LIFE_TIME = 10000

def draw_point(world, wp, option):
    if option == RoadOption.LEFT:  # Yellow
        color = carla.Color(128, 128, 0)
    elif option == RoadOption.RIGHT:  # Cyan
        color = carla.Color(0, 128, 128)
    elif option == RoadOption.CHANGELANELEFT:  # Orange
        color = carla.Color(128, 32, 0)
    elif option == RoadOption.CHANGELANERIGHT:  # Dark Cyan
        color = carla.Color(0, 32, 128)
    elif option == RoadOption.STRAIGHT:  # Gray
        color = carla.Color(64, 64, 64)
    else:  # LANEFOLLOW
        color = carla.Color(0, 128, 0)  # Green

    world.debug.draw_point(wp.transform.location + carla.Location(z=0.2), color=color, life_time=LIFE_TIME)

def draw_keypoint(world, location):
    world.debug.draw_point(location + carla.Location(z=0.2), size=0.15, color=carla.Color(128, 0, 128), life_time=LIFE_TIME)
    string = "(" + str(round(location.x, 1)) + ", " + str(round(location.y, 1)) + ", " + str(round(location.z, 1)) + ")"
    world.debug.draw_string(location + carla.Location(z=0.5), string, True, color=carla.Color(0, 0 , 128), life_time=LIFE_TIME)

def show_all_routes(filename, world, grp):
    def convert_elem_to_location(elem):
        """Convert an ElementTree.Element to a CARLA Location"""
        return carla.Location(float(elem.attrib.get('x')), float(elem.attrib.get('y')), float(elem.attrib.get('z')))

    tree = etree.parse(filename)
    root = tree.getroot()
    for route in root.iter("route"):
        prev_point = None

        for position in route.find('waypoints').iter('position'):
            point = convert_elem_to_location(position)
            draw_keypoint(world, point)

            if prev_point:
                interpolated_trace = grp.trace_route(prev_point, point)
                for wp, option in interpolated_trace:
                    draw_point(world, wp, option)
            prev_point = point

def get_saved_data(filename, route_id, world, grp):
    def convert_elem_to_location(elem):
        """Convert an ElementTree.Element to a CARLA Location"""
        return carla.Location(float(elem.attrib.get('x')), float(elem.attrib.get('y')), float(elem.attrib.get('z')))

    distance = 0

    tree = etree.parse(filename)
    root = tree.getroot()

    points = []
    for route in root.iter("route"):
        if route.attrib['id'] != route_id:
            continue

        for position in route.find('waypoints').iter('position'):
            points.append(convert_elem_to_location(position))

        if points:
            for i in range(len(points) - 1):
                waypoint = points[i]
                waypoint_next = points[i + 1]
                interpolated_trace = grp.trace_route(waypoint, waypoint_next)
                for j in range(len(interpolated_trace) - 1):
                    wp, option = interpolated_trace[j]
                    wp_next = interpolated_trace[j + 1][0]
                    draw_point(world, wp, option)
                    distance += wp.transform.location.distance(wp_next.transform.location)

                draw_keypoint(world, waypoint)
            draw_keypoint(world, points[-1])

    return points, distance

def add_data(points, tmap, world, spectator, grp):
    waypoint = tmap.get_waypoint(spectator.get_location())
    draw_keypoint(world, waypoint.transform.location)
    added_distance = 0
    if points:
        interpolated_trace = grp.trace_route(points[-1], waypoint.transform.location)
        for j in range(len(interpolated_trace) - 1):
            wp, option = interpolated_trace[j]
            wp_next = interpolated_trace[j + 1][0]
            draw_point(world, wp, option)
            added_distance += wp.transform.location.distance(wp_next.transform.location)
    points.append(waypoint.transform.location)
    return added_distance

def save_data(filename, route_id, points):
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

    tree = etree.parse(filename)
    root = tree.getroot()

    for route in root.iter("route"):
        if route.attrib['id'] != route_id:
            continue

        waypoints = route.find('waypoints')
        for position in waypoints.iter('position'):
            waypoints.remove(position)

        for point in points:
            new_point = etree.SubElement(waypoints, "position")
            new_point.set("x", str(round(point.x, 1)))
            new_point.set("y", str(round(point.y, 1)))
            new_point.set("z", str(round(point.z, 1)))
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
    argparser.add_argument('-s', '--show', action='store_true', help='Only shows the route')
    argparser.add_argument('-sa', '--show-all', action='store_true', help='Shows all the routes')
    args = argparser.parse_args()

    # Get the client
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # # Get the rest
    world = client.get_world()
    spectator = world.get_spectator()
    tmap = world.get_map()
    grp = GlobalRoutePlanner(tmap, 2.0)
    points = []

    # Show all data
    if args.show_all:
        show_all_routes(args.file, world, grp)
        sys.exit(0)

    # Get the data already at the file
    points, distance = get_saved_data(args.file, args.route_id, world, grp)
    if args.show:
        sys.exit(0)

    print(" ------------------------------------------------------------ ")
    print(" |               Use Ctrl+C to stop the script              | ")
    print(" |          Any unsaved route points will be lost           | ")
    print(" ------------------------------------------------------------ ")

    print(f"Total accumulated distance is {distance}")

    try:
        while True:
            # Get the scenario type
            action = input(f"\033[1m> Specify the next action ('Add' / 'Save') \033[0m")
            if action == "Add":
                print("Adding a new point")
                added_distance = add_data(points, tmap, world, spectator, grp)
                distance += added_distance
                print(f"Total accumulated distance is {round(distance, 1)}")
            elif action == "Save":
                print("Saving data to the xml file")
                save_data(args.file, args.route_id, points)
            else:
                print(f"\033[1m\033[93mUnknown action '{action}'. Try again\033[0m")

    except KeyboardInterrupt as e:
        print("\n Detected a keyboard interruption, stopping the script. ")

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
