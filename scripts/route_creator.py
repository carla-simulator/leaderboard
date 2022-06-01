#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import argparse
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from leaderboard.utils.route_parser import RouteParser


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

    world.debug.draw_point(wp.transform.location + carla.Location(z=0.2), color=color)


def draw_keypoint(world, location):
    world.debug.draw_point(location + carla.Location(z=0.2), size=0.15, color=carla.Color(128, 0, 128))
    string = "(" + str(round(location.x, 1)) + ", " + str(round(location.y, 1)) + ", " + str(round(location.z, 1)) + ")"
    world.debug.draw_string(location + carla.Location(z=0.5), string, True, color=carla.Color(0, 0 , 128), life_time=100000)


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', action="store_true", help='File to start')
    argparser.add_argument('-r', '--route', default="", help='Route')
    args = argparser.parse_args()

    # Get the client
    print("Initializing...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # Get the rest
    world = client.get_world()
    spectator = world.get_spectator()
    tmap = world.get_map()
    grp = GlobalRoutePlanner(tmap, 2.0)

    points = []
    points_data = ""
    distance = 0

    if args.file:
        print("Reading the given file...")
        if not args.route:
            raise ValueError("'file' argument needs a specified 'route'")

        route = RouteParser.parse_routes_file("/home/glopez/leaderboard/data/new_routes_training.xml", args.route)[0]
        points = route.keypoints
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
            points_data += f"         <position x=\"{round(waypoint.x, 1)}\" y=\"{round(waypoint.y, 1)}\" z=\"{round(waypoint.z, 1)}\"/>\n"
        draw_keypoint(world, points[-1])
        points_data += f"         <position x=\"{round(points[-1].x, 1)}\" y=\"{round(points[-1].y, 1)}\" z=\"{round(points[-1].z, 1)}\"/>\n"

    try:
        while True:

            data = input(f"Accumulated distance: {round(distance)}m. Waiting to get the next point...")
            if data.lower() == "s":
                print(points_data)
                break

            waypoint = tmap.get_waypoint(spectator.get_location())
            draw_keypoint(world, waypoint.transform.location)

            if points:
                interpolated_trace = grp.trace_route(points[-1], waypoint.transform.location)
                for j in range(len(interpolated_trace) - 1):
                    wp, option = interpolated_trace[j]
                    wp_next = interpolated_trace[j + 1][0]
                    draw_point(world, wp, option)
                    distance += wp.transform.location.distance(wp_next.transform.location)

            points.append(waypoint.transform.location)

            loc = waypoint.transform.location
            points_data += f"         <position x=\"{round(loc.x, 1)}\" y=\"{round(loc.y, 1)}\" z=\"{round(loc.z, 1)}\"/>\n"


    except KeyboardInterrupt as e:
        print("")

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)

