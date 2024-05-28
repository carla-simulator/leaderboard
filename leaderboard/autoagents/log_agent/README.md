# Log agent


This agent provides a user-friendly variant of a human agent, which is controlled by a human operator and records all actions for subsequent analysis. It also outputs a log file.

Control the agent's behavior through keyboard:

- `W` (forward), `A` (turn left), `D` (turn right), `S` (backward), `Q` (reverse)
- `E` Activate/deactivate the controller (default is off). The controller is an assistive feature that automatically adjusts the direction while the agent is moving forward to follow the route. Once the controller is activated, the user only needs to control the agent's forward and backward movement, not the steering.
- `G` Reset the agent's offset. The offset is the agent's deviation from the centerline of the route.
- `F` Adjust the agent's offset to slightly shift to the right, facilitating the avoidance of obstacles. Press `G` to restore the offset.
- `R` Adjust the agent's offset to make a lane change to the left. If you wish to return to the original lane after changing lanes, press `G` to restore the offset.
- `T` Adjust the agent's offset to make a lane change to the right. If you wish to return to the original lane after changing lanes, press `G` to restore the offset.

Additional Information Displayed:

- The top left and top right corners display the images of the left and right rearview mirrors, respectively.
- There are three lines of text displayed on the hood of the vehicle at the bottom:
  - The first line: A warning will be displayed when the EGO is about to hit a vehicle or obstacle in front: "Too Close!".
  - The second line: If the route ahead includes a lane change within a certain distance, it will display the direction and distance of the next lane change.
  - The third line: Displays the name of the most recently triggered scenario.
