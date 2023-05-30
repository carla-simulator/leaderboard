import argparse

from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict, create_default_json_msg
from leaderboard.utils.statistics_manager import StatisticsManager


def check_duplicates(route_ids):
    """Checks that all route ids are present only once in the files"""
    for id in route_ids:
        if route_ids.count(id) > 1:
            raise ValueError(f"Stopping. Found that the route {id} has more than one record")


def check_missing_data(route_ids):
    """Checks that there is no missing data, by changing their route id to an integer"""
    rep_num = 1
    prev_rep_int = 0
    prev_total_int = 0
    prev_id = ""

    for id in route_ids:
        route_int = int(id.split('_')[1])
        rep_int = int(id.split('_rep')[-1])

        # Get the amount of repetitions. Done when a reset of the repetition number is found
        if rep_int < prev_rep_int:
            rep_num = prev_rep_int + 1

        # Missing data will create a jump of 2 units
        # (i.e if 'Route0_rep1' is missing, 'Route0_rep0' will be followed by 'Route0_rep2', which are two units)
        total_int = route_int * rep_num + rep_int
        if total_int - prev_total_int > 1: 
            raise ValueError(f"Stopping. Missing some data as the ids jumped from {prev_id} to {id}")

        prev_rep_int = rep_int
        prev_total_int = total_int
        prev_id = id

def sort_records(records):
    records.sort(key=lambda x: (
        int(x['route_id'].split('_')[1])
    ))

    for i, record in enumerate(records):
        record['index'] = i

def main():
    """
    Utility script to merge two or more statistics into one.
    While some checks are done, it is best to ensure that merging all files makes sense
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--file-paths', nargs="+", required=True, help='path to all the files containing the partial results')
    argparser.add_argument('-e', '--endpoint', required=True, help='path to the endpoint containing the joined results')
    args = argparser.parse_args()

    sensors = []
    route_records = []
    total_routes = 0
    total_progress = 0

    # Get the data from the files
    for file in args.file_paths:
        data = fetch_dict(file)
        if not data:
            continue

        route_records.extend(data['_checkpoint']['records'])
        total_routes += len(data['_checkpoint']['records'])
        total_progress += data['_checkpoint']['progress'][1]

        if data['sensors']:
            if not sensors:
                sensors = data['sensors']
            elif data['sensors'] != sensors:
                raise ValueError("Stopping. Found two files with different sensor configurations")

    # Initialize the statistics manager
    statistics_records = create_default_json_msg()

    # Save sensors
    statistics_records['sensors'] = sensors
    statistics_records["_checkpoint"]["progress"] = [total_routes, total_progress]
    statistics_records['entry_status'] = 'Started'
    statistics_records['eligible'] = False

    # Save route records
    sort_records(route_records)
    statistics_records["_checkpoint"]["records"] = route_records
    save_dict(args.endpoint, statistics_records)

    # Save global records
    if total_progress != 0 and total_routes == total_progress:
        statistics_manager = StatisticsManager()
        for file in args.file_paths:
            statistics_manager.resume(file)  # Add the files info to the statistics manager
        global_records = statistics_manager.compute_global_statistics(total_progress)
        StatisticsManager.save_global_record(global_records, sensors, total_progress, args.endpoint)


if __name__ == '__main__':
    main()
