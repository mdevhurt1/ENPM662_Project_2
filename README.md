# ENPM 662 Project 2
## Prerequisites
This package requires a few prerequisites to be run.

1. Ubuntu 20.04
2. ROS2 Galactic

There is a Dockerfile included in the git repository that can be run on machines with an Nvidia gpu. If running locally, this file can be referenced for necessary enviornment installations.

In the interest of convienence, here are some commonly needed packages, although the Dockerfile may contain additional requirements. 

pip install:    
- pip install pynput
- pandas
- sympy
- numpy
- matplotlib 

This list is not comprehensive, if you encounter errors, please refer to the Dockerfile.

## Build
### Docker in VS Code
> [!Note]
> The Docker file is built for a machine with Nvidia graphics and running the VSCode Dev Containers add-in.
> If you had issues building or running the docker container, you may need to run this package locally.

1. Clone the package to your machine

2. Open the package in VS Code

3. You should be prompted to reopen the package in the container, if not press ctrl + shift + P and type "reopen in container"

4. Open a terminal with ctrl + shift + `

5. `colcon build`

6. `source install/setup.bash`

You are now ready to follow the steps in project deliverables.

### Locally
If you are running this package locally, some additional libraries may need to be installed. All required libraries are in the .devcontainer/Dockerfile in this repository.

1. Clone the package to your machine

2. navigate in a terminal to the package directory, ls should return README.md robot_arm

3. `colcon build`

4. `source install/setup.bash`

You are now ready to follow the steps in project deliverables.

## Project Deliverables
> [!Note]
>
> Each subsection in this section should be run in its own terminal!
> 
> Be sure to source install/setup.bash in each terminal.

### Launch Gazebo
1. `ros2 launch robot_arm empty_world.launch.py`

### Plan Punches
1. `ros2 run robot_arm boxing.py`

2. Press 'e' to move the robot to the ready position (this only works with a freshly launched gazebo world)

3. Follow the instructions printed to the screen for how to plan, plot, and execute combos

# Additional Package Details

## Source Program Briefs
### boxing.py
This is the control node. It allows users to plan a successio of punches, plot them, and execute them in gazebo. Read the comments to learn more about it.

### teleop.py
This file provides teleop control for the robot through wasd inputs in the terminal. Pressing q resets the velocity to 0.0, esc quits the program.

## Commonly Run Commands
### Launch gazebo
There is one world included in this package

1. empty_world

    `ros2 launch robot_arm empty_world.launch.py`

### Run controls
There are two controllers available in this package.

1. Teleop Control
    
    `ros2 run robot_arm teleop.py`

2. Boxing Open Loop Control

    `ros2 run robot_arm boxing.py`
