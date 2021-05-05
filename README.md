# TBM map merging

This repository provides an algorithm for merging occupancy grid maps, whose cells are based on the Transferable Belief Model. The developed algorithm eliminates the collision of the merged maps. 

Paper reference will be added later.

## Dependencies

* ROS (>= melodic)
* OpenCV (>= 3.2.0)

## Build

See how to [create a repo](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) and build(http://wiki.ros.org/catkin/Tutorials/using_a_workspace) on ROS Catkin documentation. 

## Run

```bash
source devel/setup.bash
rosrun map_merging map_merging --first_map=<first_map_path> --second_map=<second_map_path> --merged_map=<path_for_result_map>
```

Run with `--help` for see params.