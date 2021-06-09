from os import stat
import py_trees

class BackgroundManager(object):

    # TODO: Just here to know all the parameters. To be removed
    # Road variables (these are very much hardcoded so watch out when changing them)
    _road_front_vehicles = 3  # Amount of vehicles in front of the ego. Must be > 0
    _road_back_vehicles = 3  # Amount of vehicles behind the ego

    _road_vehicle_dist = 15  # Starting distance between spawned vehicles
    _leading_dist_interval = [6, 10]

    _base_min_radius = 38  # Should be calculated 
    _base_max_radius = 42

    # Opposite lane variables
    _opposite_sources_dist = 60
    _opposite_vehicle_dist = 15
    _opposite_sources_max_actors = 6  # Maximum vehicles alive at the same time per source

    # Break and lane change scenario
    _lane_change_dist = 50
    _lane_change_leading_dist_interval = [8, 14]

    # Junction variables
    _junction_detection_dist = 45  # Higher than max radius or junction exit dist
    _junction_entry_source_dist = 15  # Distance between spawned actors by the entry sources
    _junction_exit_dist = 15  # Distance between actors at the junction exit
    _junction_exit_space = 15  # Distance between the junction and first actor.
    _entry_sources_dist = 35  # Distance from the entry sources to the junction
    _entry_sources_max_actors = 6  # Maximum vehicles alive at the same time per source

    @staticmethod
    def activate_break_scenario(stop_duration=10):
        """Starts the break scenario"""
        py_trees.blackboard.Blackboard().set(
            "BA_BreakScenario", stop_duration, overwrite=True)
