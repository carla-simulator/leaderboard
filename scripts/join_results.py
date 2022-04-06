import argparse

from leaderboard.utils.checkpoint_tools import fetch_dict
from leaderboard.utils.statistics_manager import StatisticsManager

def main():
    """Used to help with the visualization of the scenario trigger points"""
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--file-paths', nargs="+", required=True,
                        help='path to the .json files containing the results')

    argparser.add_argument('-e', '--endpoint', required=True,
                        help='path to the .json files containing the joined results')
    args = argparser.parse_args()

    # Initialize the statistics manager
    statistics_manager = StatisticsManager(args.endpoint)

    # TODO: Make sure that the data is correctly formed + change argument format?

    total_routes = 0
    for file in args.file_paths:
        data = fetch_dict(file)
        statistics_manager.add_file_records(file)
        total_routes += len(data['_checkpoint']['records'])

    statistics_manager.sort_records()

    sample_data = fetch_dict(args.file_paths[0])
    statistics_manager.save_sensors(sample_data['sensors'])
    statistics_manager.save_progress(total_routes, total_routes)

    # Get the global data
    statistics_manager.compute_global_statistics()
    statistics_manager.validate_statistics()


if __name__ == '__main__':
    main()
