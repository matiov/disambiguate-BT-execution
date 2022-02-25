# Overview

This package uses [Grad-Cam](https://github.com/ramprs/grad-cam/) to perform object detection and image captioning.  
The data is then used to perform verbal Human Robot Interaction where the robot queries the human with questions to disambigate the scene and detect a target object.  


# Installation

There are different steps required to install all the packages:
1. Install CUDA, CUDNN, Pytorch and Torch &#8594; see [CUDA.md](./doc/cuda.md);
2. Install pre-trained modules in Grad-Cam &#8594; perform image-captioning install detailed [here](https://github.com/ramprs/grad-cam/#image-captioning);
3. Install requirements for the verbal-HRI framework &#8594; install models as explained below.


# Download Models

Download and unzip the following under `disambiguate/object_detection/available_models`:
https://drive.google.com/file/d/1sPVZ5W4Herhmd-1xOMkgSmRGMA9Pc6eY/view?usp=sharing


# Usage

The speaker/microphone name can be found with the command `$pacmd list-sources`.  


### With ROS
* Launch the robot bringup with the camera: `ros2 launch camera_interface yumi_vision.launch.py`.
    * For the LfD demo, launch instead: `ros2 launch camera_interface yumi_object.launch.py`
* In a second terminal run the ROS service: `ros2 run disambiguate_ros disambiguate_service`.
* In another teminal run the ROS client: `ros2 run disambiguate_ros disambiguate_client`.
    * For the LfD demo, the client will be implemented in a Behavior Tree.

The disambiguation framework supports also verbal interaction, which is by default disabled.  
To enable it, run `ros2 run disambiguate_ros disambiguate_service --ros-args -p verbal_interaction:=True`.


### Without ROS
* Navigate to the `disambiguate` folder.
* Run `python3 test/test_disambiguate.py`.

(**TODO**: make a better README.md with tutorials and explaination and link to paeprs).  
(**TODO**: handle differently the paths in the modules so that they depend from the disambiguation service).
