# Overview

In this package the Disambiguation server is defined, together with some client nodes to test it. The server exposes the disambiguation framework defined [here](../disambiguate/disambiguate/disambiguate.py).  

The disambiguation framework can be set to run with the verbal interaction. Otherwise, the interaction will happen by writing on the terminal. This is regulated by the parameter [`verbal_interaction`](https://github.com/matiov/behavior-tree-learning/blob/master/hri/disambiguate_ros/disambiguate_ros/disambiguate_service.py#L60). To set this parameter (that is `False` by default) it is necessary to run the disambiguation service with parameters argument:
```
ros2 run disambiguate_ros disambiguate_service --ros-args -p verbal_interaction:=True
```

The client takes as argument the object to disambiguate, so run it as it follows, substituting `<object>` with an item of your choice that is present in the scene.
```
ros2 run disambiguate_ros disambiguate_client --ros-args -p query:='<object>'
```

If you wish to test how the disambiguation service works in connection to the object detection, i.e. how the frames for the target disambiguated object is updated, then run:
```
ros2 run disambiguate_ros detection_hri_client '<object>'
```