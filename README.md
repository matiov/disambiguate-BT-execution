# Overview

Repository containing different methods to automatically generate a Behavior Tree (BT) for robotic tasks. Mainly Python based.  
In the learning methods, the BT is represented as a string, e.g. `['s(', 'action1', 'action2', ')']` and then the string is converted in a `py_tree` BT.  

Common behaviors and BT methods are defined in the [behaviors](./behaviors) module.  
Application specific behaviors will be defined in the application's dedicated module, together with the extension of the method `get_node_from_string()`.

## Contents

* The [BT-learning](./bt_learning) module contains methods to automatically generate BTs, e.g.: Learning from Demonstration, Genetic Programming, Planners.
* The [Perception Layer](./perception_layer) module contains ROS packages realizing object detection or marker recognition.
* The [Simulation](./simulation) module contains simulation environments (mainly based on a state machine simulator) for (mobile) manipulation tasks.
* The [World Interface](./world_interface) module contains interfaces to the ABB robots and to the camera in case of vision-based applications.

Please see the documentation in every module for more detailed information.


## Available documentation

### Behavior Tree Learning
* Information on the demonstration format for the Learning from Demonstration framework is found [here](./bt_learning/doc/demonstration.md)

### Human Robot Interaction
* General information about the Human Robot Interaction framework in the [README](./hri/README.md)
* Information on the installation of the CUDA modules is found [here](./hri/doc/cuda.md)

### Perception Layer
* General information about the Human Robot Interaction framework in the [README](./perception_layer/README.md)
* Information on the installation of the MaskRCNN module is found [here](./perception_layer/object_recognition/README.md)
* Instructions for the camera calibration are in the [aruco_calibration](/perception_layer/marker_detection/aruco_calibration/README.md) package

### Py Trees
* General information about the `py_trees` library in the [README](./py_trees/README.md)

### World Interface
* General information about the interface with the ABB robots in the [README](./world_interface/abb_robot/README.md)
* Information on the usage of the LfD GUI is found [here](./world_interface/abb_robot/robot_interface/doc)
* Instructions for robot visualization and spawning are in the [robot_bringup](./world_interface/abb_robot/robot_bringup/README.md) package

# Acknowledgements

We would like to thank:
* __Oscar Gustavsson__ for collaborating on the first, marker-based, version of the Learning from Demonstration framework, that resulted in the paper [Combining Context Awareness and Planning to Learn Behavior Trees from Demonstration](https://arxiv.org/abs/2109.07133).
* __Fethiye Irmak Dogan__ for the Human-Robot Interaction module backbone.
* __Zhanpeng Xie__ for the work on improving the Genetic Programming modules.


---

### Note on test routines for Copyright

The LICENSE notice is of type `BSD-3-Clause License`, that doesn't seem to be automatically recognised by the `test_copyright.py` script.  
To ignore looking for the LICENSE notice, it is necessary to modify the `main` function in the source code of the test routine, located in `/opt/ros/foxy/lib/python3.8/site-packages/ament_copyright/main.py`. To do so, comment out the lines from `157` to `161` and substitute them with this block, which just removes the LICENSE check:
```python
else:
    message = 'copyright=%s' % \
        (', '.join([str(c) for c in file_descriptor.copyrights]))
    has_error = False
```

## DISCLAIMER

The code providing the communication from the proposed framework to the ABB robot is protected by copyright and will not be disclosed. The provided robot interfaces are meant to interact with ABB robots, but can be modified to be compliant with other robot hardware. Said code, as well as the RAPID routines running inside the robot controlle, can be provided upon request. Any case will be treated individually.
