#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import argparse
import carla
from leaderboard.utils.route_parser import RouteParser

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

    if args.file:
        if not args.route:
            raise ValueError("'file' argument needs a specified 'route'")

        route = RouteParser.parse_routes_file("/home/glopez/leaderboard/data/new_routes_training.xml", args.route)[0]
        for scenario in route.scenario_configs:
            world.debug.draw_point(scenario.trigger_points[0].location + carla.Location(z=0.2), size=0.3, color=carla.Color(125, 0, 0))
            world.debug.draw_string(scenario.trigger_points[0].location + carla.Location(z=0.5), scenario.name, True, color=carla.Color(0, 0 , 125), life_time=100000)

    try:
        while True:
            input("Waiting to get the next point...")
            wp = tmap.get_waypoint(spectator.get_location())
            world.debug.draw_point(wp.transform.location + carla.Location(z=0.2), size=0.3, color=carla.Color(125, 0, 0))
            world.debug.draw_string(wp.transform.location + carla.Location(z=0.5), "???", True, color=carla.Color(0, 0 , 125), life_time=100000)
            loc = wp.transform.location
            yaw = wp.transform.rotation.yaw
            print(f"            <trigger_point x=\"{round(loc.x, 1)}\" y=\"{round(loc.y, 1)}\" z=\"{round(loc.z, 1)}\" yaw=\"{round(yaw, 1)}\"/>")

    except KeyboardInterrupt as e:
        print("")

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
