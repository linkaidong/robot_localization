//
// Created by yawara on 25-7-18.
//
#include "icp_relocalization.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "initialposeWithIcp");
    ros::NodeHandle icp_nh("~");

    LaserScanToPointCloud *relocalization_ ;
    relocalization_ = new LaserScanToPointCloud(icp_nh);

    ros::MultiThreadedSpinner spinner(2); // Use 2 threads
    spinner.spin(); // spin() will not return until the node has been shutdown
    return 0;
}