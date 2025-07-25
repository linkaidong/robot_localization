#ifndef SRC_ICP_RELOCALIZATION_H
#define SRC_ICP_RELOCALIZATION_H

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <Eigen/Core>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;

class LaserScanToPointCloud {
public:
    LaserScanToPointCloud(ros::NodeHandle &nh) : T_prev(Eigen::Matrix4f::Identity()) {
        map_sub_ = nh.subscribe("/map", 1, &LaserScanToPointCloud::mapCallback, this);
        scan_sub_ = nh.subscribe("/scan", 1, &LaserScanToPointCloud::scanCallback, this);

        initial_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 10);
        map_to_cloud_pub = nh.advertise<sensor_msgs::PointCloud>("/occupancyGridToCloud", 1);
        pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud>("/scan_to_pointcloud", 1);
        pointcloudIcpndt_pub_ = nh.advertise<sensor_msgs::PointCloud>("/out_icp_ndt_pointcloud", 1);

        map_cloud = PointCloudT::Ptr(new PointCloudT());
    }
    ~LaserScanToPointCloud() = default;

private:
    // Map callback: Convert map occupancy grid to point cloud
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& map_msg) {
        map_initialized_ = true;

        convertMapToPointCloud(map_msg, map_cloud);
        publishPointCloud(map_cloud, map_msg->header.frame_id);
    }

    // Laser scan callback: Process incoming scan data and perform ICP alignment
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_msg) {
        if (!map_initialized_) {
            ROS_ERROR("No map data");
            return;
        }

        PointCloudT::Ptr scan_cloud(new PointCloudT);
        convertLaserScanToPointCloud(scan_msg, scan_cloud);

        if (scan_cloud->empty() || map_cloud->empty()) {
            ROS_ERROR("Empty cloud data received");
            return;
        }

        // Transform the point cloud to map frame(this map frame is not original map frame)
        PointCloudT::Ptr transformed_cloud(new PointCloudT());
        transformPointCloudToMap(scan_msg, scan_cloud, transformed_cloud);

        // Perform ICP
        float score;
        Eigen::Matrix4f matrix4transform;
        if (performICP(map_cloud, transformed_cloud, matrix4transform, score)) {
            updatePose(matrix4transform, score, scan_msg);
        }

        publishTransformedCloud(transformed_cloud, scan_msg->header.stamp);
    }

    // Convert occupancy grid to point cloud
    void convertMapToPointCloud(const nav_msgs::OccupancyGrid::ConstPtr& map_msg, PointCloudT::Ptr& map_cloud) {
        int width = map_msg->info.width;
        int height = map_msg->info.height;
        double resolution = map_msg->info.resolution;
        double origin_x = map_msg->info.origin.position.x;
        double origin_y = map_msg->info.origin.position.y;
        std::vector<int8_t> data = map_msg->data;

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (data[i * width + j] > 50) {
                    double point_x = j * resolution + origin_x;
                    double point_y = i * resolution + origin_y;
                    map_cloud->push_back(PointT(point_x, point_y, 0.0));
                }
            }
        }
    }

    // Convert laser scan data to point cloud
    void convertLaserScanToPointCloud(const sensor_msgs::LaserScan::ConstPtr& scan_msg, PointCloudT::Ptr& cloud) {
        for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
            double range = scan_msg->ranges[i];
            if (std::isnan(range) || range <= scan_msg->range_min || range >= scan_msg->range_max) {
                continue;
            }
            double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
            PointT point;
            point.x = range * std::cos(angle);
            point.y = range * std::sin(angle);
            point.z = 0.0;
            cloud->push_back(point);
        }
    }

    // Transform point cloud to map frame
    void transformPointCloudToMap(const sensor_msgs::LaserScan::ConstPtr& scan_msg, PointCloudT::Ptr& cloud, PointCloudT::Ptr& transformed_cloud) {
        tf::StampedTransform transform;
        try {
            tf_listener_.waitForTransform("/map", scan_msg->header.frame_id, ros::Time(0), ros::Duration(3.0));
            tf_listener_.lookupTransform("/map", scan_msg->header.frame_id, ros::Time(0), transform);
        } catch (const tf::TransformException& ex) {
            ROS_ERROR("%s", ex.what());
            return;
        }

        Eigen::Affine3d eigen_transform;
        tf::transformTFToEigen(transform, eigen_transform);
        pcl::transformPointCloud(*cloud, *transformed_cloud, eigen_transform);
    }

    // Perform ICP alignment
    bool performICP(PointCloudT::Ptr& target_cloud, PointCloudT::Ptr& input_cloud, Eigen::Matrix4f& matrix4transform, float& score) {
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(input_cloud);
        icp.setInputTarget(target_cloud);
        icp.setTransformationEpsilon(1e-6);

        icp.setMaxCorrespondenceDistance(15);
        icp.setMaximumIterations(1);

        PointCloudT final;
        icp.align(final);
        score = icp.getFitnessScore();
        matrix4transform = icp.getFinalTransformation();

        return icp.hasConverged();
    }

    // Update the pose and publish initial pose
    void updatePose(const Eigen::Matrix4f& matrix4transform, float score, const sensor_msgs::LaserScan::ConstPtr& scan_msg) {
        if (score > 0.0004) {
            prev_score = score;
            T_prev = matrix4transform;

            // Convert matrix to PoseWithCovarianceStamped
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.matrix() = matrix4transform.cast<double>();

            geometry_msgs::PoseWithCovarianceStamped pose_msg;
            pose_msg.header.stamp = scan_msg->header.stamp;
            pose_msg.header.frame_id = "map";
            pose_msg.pose.pose.position.x = pose.translation().x();
            pose_msg.pose.pose.position.y = pose.translation().y();
            pose_msg.pose.pose.position.z = pose.translation().z();

            Eigen::Quaterniond quat(pose.rotation());
            quat.normalize();
            pose_msg.pose.pose.orientation.x = quat.x();
            pose_msg.pose.pose.orientation.y = quat.y();
            pose_msg.pose.pose.orientation.z = quat.z();
            pose_msg.pose.pose.orientation.w = quat.w();

            initial_pose_pub.publish(pose_msg);
            std::cout<< "prev_score: " << prev_score << std::endl;
        }
    }

    // Publish point cloud
    void publishPointCloud(PointCloudT::Ptr& cloud, const std::string& frame_id) {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = frame_id;
        cloud_msg.header.stamp = ros::Time::now();
        map_to_cloud_pub.publish(cloud_msg);
    }

    // Publish transformed point cloud
    void publishTransformedCloud(PointCloudT::Ptr& cloud, ros::Time stamp) {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        cloud_msg.header.stamp = stamp;
        pointcloudIcpndt_pub_.publish(cloud_msg);
    }

private:
    ros::Subscriber map_sub_, scan_sub_;
    ros::Publisher pointcloud_pub_, pointcloudIcpndt_pub_, initial_pose_pub, map_to_cloud_pub;
    pcl::registration::TransformationEstimation<PointT, PointT>::Matrix4 T_prev;
    tf::TransformListener tf_listener_;
    PointCloudT::Ptr map_cloud;

    bool map_initialized_{false};
    float prev_score = std::numeric_limits<float>::max();
};

#endif