## Latest changes

* Routes will now also take into account the criteria of their specific scenarios, only being active when they are running. The ResultWriter now automaically adds these criteria, grouping them if more than one scenario of the same type is triggered.
* Routes now automatically import all scenarios.
* Routes can now have dynamic weather. These are set by keypoints at a routes percentage, and all values between them are interpolated.
* Added the possibility to run a subset of the routes present at the routes file. Use the `--routes-subset` plus an argument such as `0-2`, to run them.
+ Added a new script a tthe `/scripts` folder, `merge_statistics.py`, that automatically merges two or more statistics into one file.

* Removed `master_scenario.py` and `atomic_criteria.py` as they are old and unused files.

* Added support for traffic manager hybrid mode.
* Added a new attribute to the global statistics, *scores_std_dev*, which calculates the standard deviation of the scores done throughout the simulation.
* Fixed bug causing the global infractions to not be correctly calculated

* Creating stable version for the CARLA online leaderboard
* Initial creation of the repository
