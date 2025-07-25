#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <deque>
#include <mutex>
#include <typeinfo>
#include <thread>

// ros
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>

// tf2
#include <tf2/utils.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_listener.h>
#include "tf2_ros/transform_broadcaster.h"

// pcl
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>

//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

class Scan2MapLocation {
private:
    ros::NodeHandle location_nh;
    ros::NodeHandle private_nh;

    ros::Subscriber laser_sub_;
    ros::Subscriber map_sub_;
    ros::Subscriber odom_sub_;

    ros::Publisher removal_pointcloud_publisher_;
    ros::Publisher location_publisher_;

    geometry_msgs::PoseWithCovarianceStamped location_match;    //定位结果

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    tf2::Transform base_to_laser_;    
    tf2::Transform laser_to_base_; 

    tf2::Transform base_in_odom_;           // base_link在odom坐标系下的坐标
    tf2::Transform base_in_odom_keyframe_;  // base_link在odom坐标系下的keyframe的坐标

    Eigen::Isometry3d base_to_lidar_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d map_to_base_ = Eigen::Isometry3d::Identity();     //map到base的欧式变换矩阵4x4
    Eigen::Isometry3d map_to_lidar_ = Eigen::Isometry3d::Identity();     //map到laser的欧式变换矩阵4x4

    Eigen::Isometry3d match_result_ = Eigen::Isometry3d::Identity();     //icp匹配结果
    Eigen::Isometry3d last_match_result_ = Eigen::Isometry3d::Identity();     //上一帧icp匹配结果

    // parameters
    bool map_initialized_ = false;
    bool scan_initialized_ = false;
    bool odom_initialized_ = true;
    bool need_relocalization = false;

    bool Use_TfTree_Always;
    bool if_debug_;

    std::string odom_frame_;
    std::string base_frame_;
    std::string map_frame_;
    std::string lidar_frame_;

    //用于计算匹配结果方差
    Eigen::Vector3d Residual_error_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Euler_Covariance_;

    std::chrono::steady_clock::time_point start_time_, end_time_;
    std::chrono::steady_clock::time_point tran_start_time_;
    std::chrono::steady_clock::time_point tran_end_time_;
    std::chrono::duration<double> tran_time_used_;

    double match_time_;         //当前匹配的时间
    double last_match_time_ = 0;    //上一帧匹配的时间
    double scan_time_;         //当前用于匹配的雷达数据时间

    double ObstacleRemoval_Distance_Max;        //如果雷达点云中点在地图点云最近点大于此值，就认为该点为障碍点

    double VoxelGridRemoval_LeafSize;           //体素滤波的边长

    ros::Time current_time_;

    //用于odom获取坐标变换
    std::mutex odom_lock_;
    std::deque<nav_msgs::Odometry> odom_queue_;
    int odom_queue_length_;

    // relocation
    double Relocation_Weight_Score_;
    double Relocation_Weight_Distance_;
    double Relocation_Weight_Yaw_;
    double Relocation_ObstacleRemoval_Distance_;
    double Relocation_Maximum_Iterations_;
    double Relocation_Score_Threshold_Max_;

    //icp
    double ANGLE_SPEED_THRESHOLD_;         //角速度阈值，大于此值不发布结果
    double AGE_THRESHOLD_;          //scan与匹配的最大时间间隔
    double ANGLE_UPPER_THRESHOLD_;  //最大变换角度
    double ANGLE_THRESHOLD_;        //最小变换角度
    double DIST_THRESHOLD_;          //最小变换距离
    double SCORE_THRESHOLD_MAX_;        //达到最大迭代次数或者到达差分阈值后后，代价仍高于此值，认为无法收敛
    double Point_Quantity_THRESHOLD_;   //点云数阈值
    double Maximum_Iterations_;           //ICP中的最大迭代次数

    double Variance_X;      //协方差
    double Variance_Y;
    double Variance_Yaw;

    double Scan_Range_Max;  //最大雷达数据距离
    double Scan_Range_Min;  //最小雷达数据距离

    //pcl
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    PointCloudT::Ptr cloud_map_;
    PointCloudT::Ptr cloud_scan_;

    void InitParams();

    bool ReLocationWithICP(Eigen::Isometry3d &trans ,const sensor_msgs::LaserScan::ConstPtr &scan_msg, PointCloudT::Ptr &cloud_map_msg, const Eigen::Isometry3d &robot_pose);

    //scan to map匹配
    bool ScanMatchWithICP(Eigen::Isometry3d &trans , PointCloudT::Ptr &cloud_scan_msg, PointCloudT::Ptr &cloud_map_msg);

    void PointCloudOutlierRemoval(PointCloudT::Ptr &cloud_msg);
    void PointCloudObstacleRemoval(PointCloudT::Ptr &cloud_map_msg, PointCloudT::Ptr &cloud_msg, double Distance_Threshold);
    void PointCloudVoxelGridRemoval(PointCloudT::Ptr &cloud_msg, double leafSize);

    //数据格式转换
    void OccupancyGridToPointCloud(const nav_msgs::OccupancyGrid::ConstPtr &map_msg, PointCloudT::Ptr &cloud_msg);
    void ScanToPointCloudOnMap(const sensor_msgs::LaserScan::ConstPtr &scan_msg, PointCloudT::Ptr &cloud_msg);

    //坐标变换
    bool GetTransform(Eigen::Isometry3d &trans , const std::string parent_frame, 
                                    const std::string child_frame, const ros::Time stamp);
    bool GetOdomTransform(Eigen::Isometry3d &trans, double start_stamp, double end_stamp);
    bool Get2TimeTransform(Eigen::Isometry3d &trans);

public:
    Scan2MapLocation();
    ~Scan2MapLocation() = default;

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &map_msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr &odometryMsg);
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr &scan_msg);
};
