cmake_minimum_required(VERSION 3.0.2)
project(ucar_localization)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED)

set(ROS_DEPENDENCIES
        roscpp
        sensor_msgs
        geometry_msgs
        tf2
        tf
        tf_conversions
        tf2_geometry_msgs
        tf2_ros
        pcl_conversions
        pcl_ros
        )

find_package(catkin REQUIRED COMPONENTS ${ROS_DEPENDENCIES})

catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        CATKIN_DEPENDS
        ${ROS_DEPENDENCIES}
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)

add_executable(scan_to_map_location_node src/scan_to_map_location.cpp)
target_link_libraries(scan_to_map_location_node  ${catkin_LIBRARIES})
add_dependencies(scan_to_map_location_node ${catkin_EXPORTED_TARGETS})

add_executable(relocalization src/icp_relocalization.cpp)
target_link_libraries(relocalization  ${catkin_LIBRARIES})
add_dependencies(relocalization ${catkin_EXPORTED_TARGETS})
