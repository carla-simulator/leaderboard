## Latest changes (ue58-dev)

* Made the branch work against CARLA UE5 (0.10.x) servers, paired with the `ue58-dev` branch of ScenarioRunner:
    - Sensor callbacks dispatched on `carla.libcarla.*` types, which no longer exist as a submodule in the CARLA 0.10 wheel; they now use the top-level `carla.*` classes (this made every sensorized agent fail with "sensor took too long").
    - Requested towns are resolved against the server's available maps through `resolve_map_name` (CARLA UE5 needs exact names, e.g. the classic towns only ship as `_Opt` layered variants).
    - The ego blueprint `vehicle.lincoln.mkz_2020` is translated by ScenarioRunner's UE5 blueprint aliases.
    - ROS agent support (and its `pexpect`/`transforms3d` dependencies) is imported lazily, so plain Python agents work without them.
* Brought bug fixes and tools from `master`: Python 3.9+ route parser, dependency-version compatibility, Speedometer fix, OpenDRIVE reading frequency removal, the log agent, and the LB1-to-LB2 route bridge script.
* Added `TmAgent`, a TrafficManager-driven ego agent for stress-testing routes without an AD stack.
* The spectator now follows the ego as a third-person chase camera.

* Decoupled the leaderboard from ScenarioRunner.
* Upgraded py_trees from 0.8.3 to 2.6.0.
* Route scenarios now use the route versions of `ConstructionObstacle`, `ConstructionObstacleTwoWays` and `ParkingExit`.
* Removed the `route_mode`, `criteria_enable`, `randomize`, `terminate_on_failure` and `optional` options.
* Fixed `Accident`, `ParkedObstacle` and `HazardAtSideLane` ignoring `debug_mode`.
* Updated requirements to match the new decoupled state.

* Renamed `CarlaDataProvider` to `SimulationData`, in `leaderboard/scenariomanager/simulation_data.py`. It now holds only the client, the world, its map, the global route planner, the random state and the traffic manager port.
* Actor state is no longer cached. Read location and transform from the CARLA Python API; `CarlaDataProvider.get_velocity()` becomes `get_ground_speed()` in `leaderboard/tools/actor_state.py`, which returns the 2D speed as before. An actor spawned with `tick=False` reads as the origin until the world next advances: `has_actor_ticked()`, in the same module, is the check for it.
* Moved spawning to `leaderboard/tools/actor_factory.py`: `request_new_actor`, `request_new_actors`, `request_new_batch_actors`, `handle_actor_batch`, `create_blueprint` and `resolve_model_name`. Moved `resolve_map_name` to `leaderboard/tools/map_utils.py`.
* Whoever creates an actor destroys it. The global actor pool is gone, together with `cleanup()`'s batch destroy, `actor_id_exists`, `remove_actor_by_id` and `get_hero_actor`; a scenario destroys `self.other_actors` in `remove_all_actors()`, an atomic destroys what it spawned in `terminate()`, and `destroy_actor`/`destroy_actors` in `leaderboard/tools/actor_state.py` do it. The evaluator still sweeps the world for leftover sensors.
* Replaced `is_sync_mode()` and the runtime init mode with `SimulationData.set_do_tick()` and `SimulationData.tick_world()`, which advances the world or waits for whoever is advancing it.
* Renamed `get_random_seed()` to `get_rng()` and `set_world()` to `prepare_world()`. Removed `prepare_map()` and the traffic light map.
* Removed `random_location` from spawning and from `ActorConfigurationData`; the `random_location` attribute in a scenario's `<other_actor>` is ignored.
* Replaced `set_latest_scenario()`/`get_latest_scenario()` with the `latest_scenario` blackboard variable.
* `_destroy_other_actors()` is gone; call `remove_all_actors()`.
* Fixed the route's first weather never being applied, and the weather keypoint padding raising on routes that do not define a weather at 0% or 100%.
* Fixed routes crashing with "trying to operate on a destroyed actor" when the Traffic Manager removes a stuck background vehicle. The background activity read its location before noticing it was gone, as the world's actor list still listed it until the next tick; it now also drops actors that are no longer alive. The agent behaviours (`BasicAgentBehavior`, `ConstantVelocityAgentBehavior`, `AdaptiveConstantVelocityAgentBehavior`) also tolerate an actor destroyed before they terminate.

Known gaps on CARLA UE5: scenarios needing two-wheelers (e.g. HazardAtSideLane) require a CARLA build whose content registers them (in progress on the ue58 content branch); on stock 0.10 content they are skipped.

## Latest changes

* Added support to ROS agents, which are meant to inherit from the new `ROS1Agent` and `ROS2Agent` agents. The old `RosAgent` has been deleted.
* Improved the format of the printed runtime information
* Added optional side mirrors to the hyman agent
* Improved the performance of the routes by initializing the scenarios during runtime
* Improved result writer output, in the same wasy as ScenarioRunner's one.
* Improved the initialization and cleanup of the Leaderboard.
* Improved the robustness of the resuming functionality
* Improved the example Dockerfile and added new example for ROS based agents
* Added new utility scripts:
    - merge_statistics.py: join two or more json results into one
    - route_creator.py: simplifies the creation of new routes
    - route_displayer.py: parse and debug route xml files from inside the simulator
    - route_summarizer.py: parses the route xml file into a table.
    - scenario_creator.py: uses the spectator and terminal inputs to automatically add scenarios to a route
    - scenario_orderer.py: modifies the scenarios part of the route file to be ordered according to their route's position
    - weather_creator.py: gets the current simulation's weather in the route format for easy copy.
* The `StatisticsManager` class has been remade to add robustness, remove unneeded complexity and hardcoded values. Its interaction with other classes has remained unchanged
* Routes have had the same changed as the ones in ScenarioRunner
* Added parked vehicles to the routes. These are parsed from a file with all their possible positions.
* Added new arguments to the leaderboard:
    - `routes-subset` allows to run part of the routes. Use `-` to run groups of routes (i.e `0-4`), `,` to run specific routes (i.e `1,6,8,14`), or a combination of the two (i.e `0-2,5,8-10`).
    - `debug-checkpoint` defines the endpoint of the live results, created when the `debug` argument is 2 or higher.
* Added support for traffic manager hybrid mode.
* Added a new attribute to the global statistics, *scores_std_dev*, which calculates the standard deviation of the scores done throughout the simulation.
* Fixed bug causing the global infractions to not be correctly calculated

* Creating stable version for the CARLA online leaderboard
* Initial creation of the repository
