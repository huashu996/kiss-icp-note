// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"
#include "kiss_icp/pipeline/KissLV.hpp"
#include "livox_ros_driver/CustomMsg.h"
// ROS
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf2_ros/transform_broadcaster.h"
#include <mutex>
#include <thread>
#include <signal.h>
#include <condition_variable>
//
#include <iostream>
#include <fstream>
#include <string>
//
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
namespace kiss_icp_ros {
using namespace std;
Eigen::Matrix<double, 3, 3, Eigen::RowMajor> ros_intrisicMat;
Eigen::Matrix<double, 4, 4, Eigen::RowMajor> ros_extrinsicMat_RT;
Eigen::Matrix<double, 5, 1> ros_distCoeffs;
std::vector< double > ros_extrinsicMat_RT_data, ros_intrisicMat_data, ros_distCoeffs_data;
cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);	
cv::Mat extrinsicMat_RT(4, 4, cv::DataType<double>::type);	
cv::Mat un_intrisicMat(3, 3, cv::DataType<double>::type);	   // 内参3*4的投影矩阵，最后一列是三个零
cv::Mat intrisicMat(3, 4, cv::DataType<double>::type);	   // 内参3*4的投影矩阵，最后一列是三个零
struct MeasureGroup {
	  livox_ros_driver::CustomMsg::ConstPtr lidar;
	  sensor_msgs::CompressedImageConstPtr image;
};
class OdometryServer {
public:
    /// OdometryServer constructor
    OdometryServer(const ros::NodeHandle &nh, const ros::NodeHandle &pnh); //ROS服务

private:
    /// Register new frame
    void RegisterFrame(const sensor_msgs::PointCloud2 &msg); //进入主程序
    void RegisterFrame2(const livox_ros_driver::CustomMsgConstPtr &msg); //进入主程序
    void Register_Color_Frame(const livox_ros_driver::CustomMsg::ConstPtr &lidar_msg, const sensor_msgs::CompressedImageConstPtr &img_msg);
    void LivoxMsgCbk(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void ImageCallback(const sensor_msgs::CompressedImageConstPtr &msg);
    void Test_sync();
    bool SyncMeasure(MeasureGroup &measgroup);
    void Init_param( );
    void resetParameters();

    /// Ros node stuff
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    int queue_size_{1};

    /// Tools for broadcasting TFs.
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    /// Data subscribers.
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber sub_livox_msg;
    ros::Subscriber sub_image;

    /// Data publishers.
    ros::Publisher odom_publisher_;
    ros::Publisher traj_publisher_;
    nav_msgs::Path path_msg_;
    ros::Publisher frame_publisher_;
    ros::Publisher kpoints_publisher_;
    ros::Publisher local_map_publisher_;
    ros::Publisher image_pub;

    /// KISS-ICP
    kiss_icp::pipeline::KissICP odometry_;  //创建一个KissICP类对象
    kiss_icp::pipeline::KissLV LVodometry_;
    kiss_icp::pipeline::KISSConfig config_; //创建一个config_类结构体

    /// Global/map coordinate frame.
	std::string odom_frame_{"odom"};
    std::string child_frame_{"base_link"};
    std::string LidarType{"Livox"};
    
   
    // use livox
	bool cut_frame;
	int cut_frame_num;
	int scan_count = 0;
	std::string lidar_topic;
	std::string image_topic;
	 // use camera
	bool en_cam;
    int W;
	int H;
	//sync

	double last_timestamp_lidar = -1;
	double last_timestamp_image = -1;
	std::deque<livox_ros_driver::CustomMsg::ConstPtr> lidar_buffer;
	std::deque<sensor_msgs::CompressedImageConstPtr> image_buffer;
	std::thread Test_sync_thread_;
	std::mutex mtx_buffer;

	std::condition_variable sig_buffer;

	bool b_exit = false;
	bool b_reset = false;
	bool data_sync = false;
	//pcl
	PointCloudXYZIN::Ptr pcl_points;
	PointCloudXYZRGB::Ptr scan_keypoint_enhance;
	PointCloudXYZRGB::Ptr color_scan;

};

}  // namespace kiss_icp_ros
