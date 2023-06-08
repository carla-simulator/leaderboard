import argparse

from leaderboard.utils.checkpoint_tools import fetch_dict
import sys

PRETTY_SENSORS = {
    'carla_camera': 'RGB Camera',
    'carla_lidar': 'LIDAR',
    'carla_radar': 'Radar',
    'carla_gnss': 'GNSS',
    'carla_imu': 'IMU',
    'carla_opendrive_map': 'OpenDrive Map',
    'carla_speedometer': 'Speedometer'
}

def main():
    """
    Utility script to merge two or more statistics into one.
    While some checks are done, it is best to ensure that merging all files makes sense
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--file-path', required=True, help='path to all the files containing the partial results')
    argparser.add_argument('-e', '--endpoint', required=True, help='path to the endpoint containing the joined results')
    args = argparser.parse_args()

    data = fetch_dict(args.file_path)

    if not data or 'sensors' not in data or '_checkpoint' not in data \
        or 'progress' not in data['_checkpoint'] or 'records' not in data['_checkpoint']:
        pretty_output = "Initializing the submission, no data avaialable yet.\n"
        pretty_output = "More information will be found here once the submission starts running.\n"
        with open(args.endpoint, 'w') as fd:
            fd.write(pretty_output)
        sys.exit(0)

    pretty_output = "Here is a summary of the submission's current results\n\n"
    pretty_output += "Starting with the general information:\n"

    # Sensors
    pretty_output += "- Sensors:\n"
    sensors = {}
    for sensor in data['sensors']:
        pretty_sensor = PRETTY_SENSORS[sensor]
        if pretty_sensor in sensors:
            sensors[pretty_sensor] += 1
        else:
            sensors[pretty_sensor] = 1
    for sensor_type, sensor_number in sensors.items():
        pretty_output += f"  - {sensor_number} {sensor_type}\n"

    # Completed routes
    completed_routes, total_routes = data['_checkpoint']['progress']
    if completed_routes == total_routes:
        pretty_output += f"- All {total_routes} route have been completed\n"
    else:
        pretty_output += f"- Completed {completed_routes} out of the {total_routes} routes\n"

    # Routes data
    total_duration_game = 0
    total_duration_system = 0
    route_records = []
    for record in data['_checkpoint']['records']:
        ratio = 0 if record['meta']['duration_system'] == 0 else record['meta']['duration_game']/record['meta']['duration_system']
        route_records.append({
            "route_id": record['route_id'],
            "index": record['index'],
            "status": record['status'],
            "ratio": ratio,
        })

        total_duration_game += record['meta']['duration_game']
        total_duration_system += record['meta']['duration_system']

    # General duration
    ratio = 0 if total_duration_system == 0 else total_duration_game / total_duration_system
    pretty_output += f"- Submission ratio of {ratio}x\n"
    pretty_output += f"- Submission FPS of {20*ratio}\n"
    pretty_output += "\n"

    # Route data
    pretty_output += "And here is a glossary of each route:\n"

    for route in route_records:
        pretty_output += "\n"
        pretty_output += f"- Index: {route['index']}\n"
        pretty_output += f"  - Route ID: {route['route_id']}\n"
        pretty_output += f"  - Status: {route['status']}\n"
        pretty_output += f"  - Ratio: {route['ratio']}x\n"
        pretty_output += f"  - FPS: {20*route['ratio']}\n"

    with open(args.endpoint, 'w') as fd:
        fd.write(pretty_output)

if __name__ == '__main__':
    main()
