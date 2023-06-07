import argparse

from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict
import sys

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
    if '_checkpoint' in data:
        global_records = data['_checkpoint']['global_record']
        if global_records:
            output = [
                {
                    "split": "leaderboard",
                    "show_to_participant": True,
                    "accuracies": {
                        "Driving score": global_records['scores']['score_composed'],
                        "Route completion": global_records['scores']['score_route'],
                        "Infraction penalty": global_records['scores']['score_penalty'],
                        "Collisions pedestrians": global_records['infractions']['collisions_pedestrian'],
                        "Collisions vehicles": global_records['infractions']['collisions_vehicle'],
                        "Collisions layout": global_records['infractions']['collisions_layout'],
                        "Red light infractions": global_records['infractions']['red_light'],
                        "Stop sign infractions": global_records['infractions']['stop_infraction'],
                        "Off-road infractions": global_records['infractions']['outside_route_lanes'],
                        "Route deviations": global_records['infractions']['route_dev'],
                        "Route timeouts": global_records['infractions']['route_timeout'],
                        "Agent blocked": global_records['infractions']['vehicle_blocked']
                    }
                }
            ]
        else:
            output = [
                {
                    "split": "leaderboard",
                    "show_to_participant": True,
                    "accuracies": {
                        "Driving score": 0,
                        "Route completion": 0,
                        "Infraction penalty": 0,
                        "Collisions pedestrians": 0,
                        "Collisions vehicles": 0,
                        "Collisions layout": 0,
                        "Red light infractions": 0,
                        "Stop sign infractions": 0,
                        "Off-road infractions": 0,
                        "Route deviations": 0,
                        "Route timeouts": 0,
                        "Agent blocked": 0,
                    }
                }
            ]

    save_dict(args.endpoint, output)

if __name__ == '__main__':
    main()
