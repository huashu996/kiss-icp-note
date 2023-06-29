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
#include <Eigen/Core>
#include <vector>

// KISS-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TransformStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/init.h"
#include "ros/node_handle.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"
namespace kiss_icp_ros {

OdometryServer::OdometryServer(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)  //OdometryServer类里的OdometryServer函数
    : nh_(nh), pnh_(pnh) {
    //设置参数
	OdometryServer::Init_param();
	
    // Construct the main KISS-ICP odometry node
    //创建里程计
    odometry_ = kiss_icp::pipeline::KissICP(config_);//创建里程计  kiss_icp::pipeline是命名空间，KissICP是该空间的显性类
    LVodometry_ = kiss_icp::pipeline::KissLV(config_);//创建里程计  kiss_icp::pipeline是命名空间，KissICP是该空间的显性类

    // Initialize subscribers
    //接收点云
    

	sub_livox_msg = nh_.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", queue_size_, &OdometryServer::LivoxMsgCbk, this ,ros::TransportHints().tcpNoDelay());
	sub_image = nh_.subscribe(image_topic, queue_size_, &OdometryServer::ImageCallback,this, ros::TransportHints().tcpNoDelay());
	Test_sync_thread_ = std::thread(&OdometryServer::Test_sync, this);
	//sub_livox_msg = nh_.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", queue_size_, &OdometryServer::RegisterFrame2, this);
	//else{pointcloud_sub_ = nh_.subscribe<const sensor_msgs::PointCloud2 &>("pointcloud_topic", queue_size_, &OdometryServer::RegisterFrame, this);}
	
    // Initialize publishers
    //kiss_icp::pipeline是命名空间     KissICP是该空间下的类     poses(), RegisterFrame(), LocalMap()是该类的一个函数
    odom_publisher_ = pnh_.advertise<nav_msgs::Odometry>("odometry", queue_size_); //通过kiss_icp::pipeline::KissICP.poses()函数发布计算后的位姿变化
    frame_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("frame", queue_size_); //通过kiss_icp::pipeline::KissICP.RegisterFrame()发布去矫的当前帧，用于建图
    kpoints_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("keypoints", queue_size_); //通过kiss_icp::pipeline::KissICP.RegisterFrame()发布当前帧的特征点
    local_map_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("local_map", queue_size_); //通过kiss_icp::pipeline::KissICP.LocalMap()发布局部地图
	image_pub = pnh_.advertise<sensor_msgs::Image>("new_image", 1);
    // Initialize trajectory publisher
    path_msg_.header.frame_id = odom_frame_;
    traj_publisher_ = pnh_.advertise<nav_msgs::Path>("trajectory", queue_size_);  //发布路线

    // Broadcast a static transformation that links with identity the specified base link to the
    // pointcloud_frame, basically to always be able to visualize the frame in rviz
    if (child_frame_ != "base_link") {
        static tf2_ros::StaticTransformBroadcaster br;
        geometry_msgs::TransformStamped alias_transform_msg;
        alias_transform_msg.header.stamp = ros::Time::now();
        alias_transform_msg.transform.translation.x = 0.0;
        alias_transform_msg.transform.translation.y = 0.0;
        alias_transform_msg.transform.translation.z = 0.0;
        alias_transform_msg.transform.rotation.x = 0.0;
        alias_transform_msg.transform.rotation.y = 0.0;
        alias_transform_msg.transform.rotation.z = 0.0;
        alias_transform_msg.transform.rotation.w = 1.0;
        alias_transform_msg.header.frame_id = child_frame_;
        alias_transform_msg.child_frame_id = "base_link";
        br.sendTransform(alias_transform_msg);
    }
    // publish odometry msg
    ROS_INFO("KISS-ICP ROS 1 Odometry Node Initialized");
}

void OdometryServer::LivoxMsgCbk(const livox_ros_driver::CustomMsg::ConstPtr& msg){
		const double timestamp = msg->header.stamp.toSec();
		mtx_buffer.lock();
		ROS_DEBUG("get point cloud at time: %.6f", timestamp);
		
		if (timestamp < last_timestamp_lidar) 
		{
		    ROS_ERROR("lidar loop back, clear buffer");
		    lidar_buffer.clear();
		}
		last_timestamp_lidar = timestamp;
		lidar_buffer.push_back(msg);
		mtx_buffer.unlock();
		sig_buffer.notify_all();	
	}
void OdometryServer::ImageCallback(const sensor_msgs::CompressedImageConstPtr &msg)
	{
		double timestamp = msg->header.stamp.toSec();
		mtx_buffer.lock();
		if (timestamp < last_timestamp_image) {
		    ROS_ERROR("imu loop back, clear buffer");
		    image_buffer.clear();
		    b_reset = true;
		}
		last_timestamp_image = timestamp;
		image_buffer.push_back(msg);
		mtx_buffer.unlock();
		sig_buffer.notify_all();	
	}
bool OdometryServer::SyncMeasure(MeasureGroup &measgroup) 
	{
		if (lidar_buffer.empty() || image_buffer.empty()) 
		{
		    return false;
		}
		if (image_buffer.front()->header.stamp.toSec() > lidar_buffer.back()->header.stamp.toSec()) 
		{
		    lidar_buffer.clear();
		    ROS_ERROR("clear lidar buffer, only happen at the beginning===============");
		    return false;
		}

		if (image_buffer.back()->header.stamp.toSec() < lidar_buffer.front()->header.stamp.toSec()) 
		{
		    return false;
		}
		measgroup.lidar = lidar_buffer.front();
		lidar_buffer.pop_front();
		double lidar_time = measgroup.lidar->header.stamp.toSec();

		// Find the image message with the shortest time difference from lidar_sync_msg
		double min_time_diff = std::numeric_limits<double>::max();
		int image_cnt = 0;
		for (const auto &image : image_buffer) 
		{
		    double image_time = image->header.stamp.toSec();
		    if (image_time <= lidar_time) 
		    {
		        double time_diff = std::abs(lidar_time - image_time);
		        if (time_diff < min_time_diff)
				{
				    min_time_diff = time_diff;
				    measgroup.image = image;
				}
		        image_cnt++;
		    }
		}
		for (int i = 0; i < image_cnt; ++i) 
		{
		    image_buffer.pop_front();
		}
		return true;
	}
void OdometryServer::Test_sync(){
		while (ros::ok()) {
			MeasureGroup meas;
			std::unique_lock<std::mutex> lock(mtx_buffer);
			sig_buffer.wait(lock, [this, &meas]() -> bool { return SyncMeasure(meas) || b_exit; });//同步
			mtx_buffer.unlock();
			if (b_exit) 
			{
			    ROS_INFO("b_exit=true, exit");
			    break;
			}
			if (b_reset) 
			{
			    ROS_WARN("reset when rosbag play back");
			    b_reset = false;
			    continue;
			}
			OdometryServer::Register_Color_Frame(meas.lidar, meas.image);
	}
}
void OdometryServer::Register_Color_Frame(const livox_ros_driver::CustomMsg::ConstPtr& lidar_msg, const sensor_msgs::CompressedImageConstPtr &img_msg) {
	pcl_points = utils::Livox_pcl_point(lidar_msg);
	const auto timestamps = utils::Livox_time(lidar_msg);
	std::vector<std::vector<cv::Vec3b>> image_color(H, std::vector<cv::Vec3b>(W));
	cv::Mat gray_image = utils::Image_process(img_msg,image_color);
	color_scan = utils::get_color(pcl_points, image_color, intrisicMat, extrinsicMat_RT);
	PointCloudXYZIN::Ptr test_pcl(new PointCloudXYZIN());
	scan_keypoint_enhance = LVodometry_.RegisterFrame(pcl_points, timestamps, gray_image, intrisicMat, extrinsicMat_RT); //返回当前点云和预处理下采样后的特征点云
	const auto pose = LVodometry_.poses().back();
	//4. 将位姿转化为ROS消息类型
    // Convert from Eigen to ROS types
    const Eigen::Vector3d t_current = pose.translation();
    const Eigen::Quaterniond q_current = pose.unit_quaternion();
    //---------------------------------------pub--------------------------------------------
	//5. 发布结果消息
    // Broadcast the tf
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.stamp = ros::Time::now();
    transform_msg.header.frame_id = odom_frame_;
    transform_msg.child_frame_id = child_frame_;
    transform_msg.transform.rotation.x = q_current.x();
    transform_msg.transform.rotation.y = q_current.y();
    transform_msg.transform.rotation.z = q_current.z();
    transform_msg.transform.rotation.w = q_current.w();
    transform_msg.transform.translation.x = t_current.x();
    transform_msg.transform.translation.y = t_current.y();
    transform_msg.transform.translation.z = t_current.z();
    tf_broadcaster_.sendTransform(transform_msg);

    // publish odometry msg
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = child_frame_;
    odom_msg.pose.pose.orientation.x = q_current.x();
    odom_msg.pose.pose.orientation.y = q_current.y();
    odom_msg.pose.pose.orientation.z = q_current.z();
    odom_msg.pose.pose.orientation.w = q_current.w();
    odom_msg.pose.pose.position.x = t_current.x();
    odom_msg.pose.pose.position.y = t_current.y();
    odom_msg.pose.pose.position.z = t_current.z();
    odom_publisher_.publish(odom_msg);

    // publish trajectory msg
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.pose = odom_msg.pose.pose;
    pose_msg.header = odom_msg.header;
    path_msg_.poses.push_back(pose_msg);
    traj_publisher_.publish(path_msg_);

    // Publish KISS-ICP internal data, just for debugging
   	sensor_msgs::PointCloud2 rgb_pointscan;
	pcl::toROSMsg(*color_scan, rgb_pointscan);
	rgb_pointscan.header.stamp = ros::Time::now();
	rgb_pointscan.header.frame_id = child_frame_;
	frame_publisher_.publish(rgb_pointscan);
	
	sensor_msgs::PointCloud2 key_pointcloud;
	pcl::toROSMsg(*scan_keypoint_enhance, key_pointcloud);
	rgb_pointscan.header.stamp = ros::Time::now();
	key_pointcloud.header.frame_id = child_frame_;
	kpoints_publisher_.publish(key_pointcloud);

    // Map is referenced to the odometry_frame
	PointCloudXYZRGB::Ptr map_points(new PointCloudXYZRGB());
	sensor_msgs::PointCloud2 map_msg;
	auto color_local_map_ = LVodometry_.LocalMap();
	map_points = utils::Eigen6dToPointCloud2(color_local_map_);
	pcl::toROSMsg(*map_points, map_msg);	
	map_msg.header.stamp = ros::Time::now();
	map_msg.header.frame_id = odom_frame_;
	local_map_publisher_.publish(map_msg);
	
	sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", gray_image).toImageMsg();
	image_pub.publish(image_msg);

	//------------------------------------------更新----------------------------------------
	OdometryServer::resetParameters();
	
}
void OdometryServer::RegisterFrame2(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    //1. 接收消息
    const auto timestamps = utils::Livox_time(msg);
    const auto points = utils::Livox_point(msg);
	//2. 提取修正当前帧和特征点
    // Register frame, main entry point to KISS-ICP pipeline
    const auto &[frame, keypoints] = odometry_.RegisterFrame(points, timestamps); //返回当前点云和预处理下采样后的特征点云
	//3. 计算位姿pose
    // PublishPose
    const auto pose = odometry_.poses().back();
	//4. 将位姿转化为ROS消息类型
    // Convert from Eigen to ROS types
    const Eigen::Vector3d t_current = pose.translation();
    const Eigen::Quaterniond q_current = pose.unit_quaternion();
	//5. 发布结果消息
    // Broadcast the tf
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.stamp = ros::Time::now();
    transform_msg.header.frame_id = odom_frame_;
    transform_msg.child_frame_id = child_frame_;
    transform_msg.transform.rotation.x = q_current.x();
    transform_msg.transform.rotation.y = q_current.y();
    transform_msg.transform.rotation.z = q_current.z();
    transform_msg.transform.rotation.w = q_current.w();
    transform_msg.transform.translation.x = t_current.x();
    transform_msg.transform.translation.y = t_current.y();
    transform_msg.transform.translation.z = t_current.z();
    tf_broadcaster_.sendTransform(transform_msg);

    // publish odometry msg
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = child_frame_;
    odom_msg.pose.pose.orientation.x = q_current.x();
    odom_msg.pose.pose.orientation.y = q_current.y();
    odom_msg.pose.pose.orientation.z = q_current.z();
    odom_msg.pose.pose.orientation.w = q_current.w();
    odom_msg.pose.pose.position.x = t_current.x();
    odom_msg.pose.pose.position.y = t_current.y();
    odom_msg.pose.pose.position.z = t_current.z();
    odom_publisher_.publish(odom_msg);

    // publish trajectory msg
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.pose = odom_msg.pose.pose;
    pose_msg.header = odom_msg.header;
    path_msg_.poses.push_back(pose_msg);
    traj_publisher_.publish(path_msg_);

    // Publish KISS-ICP internal data, just for debugging
    std_msgs::Header frame_header;
    frame_header.stamp = ros::Time::now();
    frame_header.frame_id = child_frame_;
    frame_publisher_.publish(utils::EigenToPointCloud2(frame, frame_header));
    kpoints_publisher_.publish(utils::EigenToPointCloud2(keypoints, frame_header));

    // Map is referenced to the odometry_frame
    std_msgs::Header local_map_header;
    local_map_header.stamp = ros::Time::now();
    local_map_header.frame_id = odom_frame_;
    local_map_publisher_.publish(utils::EigenToPointCloud2(odometry_.LocalMap(), local_map_header));
}

void OdometryServer::Init_param( )
	{	
		pnh_.getParam("kiss/voxel_size", config_.voxel_size);
		pnh_.getParam("kiss/max_range", config_.max_range);
		pnh_.getParam("kiss/min_range", config_.min_range);
		pnh_.getParam("kiss/max_points_per_voxel", config_.max_points_per_voxel);
		pnh_.getParam("kiss/min_motion_th", config_.min_motion_th);
		pnh_.getParam("kiss/initial_threshold", config_.initial_threshold);
		pnh_.getParam("kiss/deskew", config_.deskew);

		pnh_.getParam("common/cut_frame", cut_frame);
		pnh_.getParam("common/cut_frame_num", cut_frame_num);
		pnh_.getParam("common/lidar_topic", lidar_topic);
		pnh_.getParam("common/image_topic", image_topic);
		pnh_.getParam("common/LidarType", LidarType);
		pnh_.getParam("child_frame", child_frame_);
		pnh_.getParam("odom_frame", odom_frame_);
		
		pnh_.getParam( "kiss_vio/image_height", H);
		pnh_.getParam( "kiss_vio/image_weight", W);
		pnh_.getParam( "kiss_vio/en_cam", en_cam);
		pnh_.getParam( "kiss_vio/extrinsicMat_RT", ros_extrinsicMat_RT_data);
		pnh_.getParam( "kiss_vio/intrisicMat", ros_intrisicMat_data);
		pnh_.getParam( "kiss_vio/distCoeffs", ros_distCoeffs_data);
		
		if (config_.max_range < config_.min_range) {
		    ROS_WARN("[WARNING] max_range is smaller than min_range, setting min_range to 0.0");
		    config_.min_range = 0.0;
		}
		if ( ( ros_intrisicMat_data.size() != 9 ) || ( ros_distCoeffs_data.size() != 5 ) || ( ros_extrinsicMat_RT_data.size() != 16 ))
		{
		    cout << "Load VIO parameter fail!!!, please check!!!" << endl;
		    printf( "Load camera data size = %d, %d, %d, %d\n", ( int ) ros_extrinsicMat_RT_data.size(), ros_intrisicMat_data.size(),ros_distCoeffs_data.size());
		    std::this_thread::sleep_for( std::chrono::seconds( 3000000 ) );
		}
		ros_extrinsicMat_RT = Eigen::Map< Eigen::Matrix< double, 4, 4, Eigen::RowMajor > >( ros_extrinsicMat_RT_data.data());
		ros_intrisicMat = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( ros_intrisicMat_data.data());
		ros_distCoeffs = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( ros_distCoeffs_data.data());
		

		extrinsicMat_RT.at<double>(0, 0) = ros_extrinsicMat_RT(0,0);
		extrinsicMat_RT.at<double>(0, 1) = ros_extrinsicMat_RT(0,1);
		extrinsicMat_RT.at<double>(0, 2) = ros_extrinsicMat_RT(0,2);
		extrinsicMat_RT.at<double>(0, 3) = ros_extrinsicMat_RT(0,3);
		extrinsicMat_RT.at<double>(1, 0) = ros_extrinsicMat_RT(1,0);
		extrinsicMat_RT.at<double>(1, 1) = ros_extrinsicMat_RT(1,1);
		extrinsicMat_RT.at<double>(1, 2) = ros_extrinsicMat_RT(1,2);
		extrinsicMat_RT.at<double>(1, 3) = ros_extrinsicMat_RT(1,3);
		extrinsicMat_RT.at<double>(2, 0) = ros_extrinsicMat_RT(2,0);
		extrinsicMat_RT.at<double>(2, 1) = ros_extrinsicMat_RT(2,1);
		extrinsicMat_RT.at<double>(2, 2) = ros_extrinsicMat_RT(2,2);
		extrinsicMat_RT.at<double>(2, 3) = ros_extrinsicMat_RT(2,3);
		extrinsicMat_RT.at<double>(3, 0) = ros_extrinsicMat_RT(3,0);
		extrinsicMat_RT.at<double>(3, 1) = ros_extrinsicMat_RT(3,1);
		extrinsicMat_RT.at<double>(3, 2) = ros_extrinsicMat_RT(3,2);
		extrinsicMat_RT.at<double>(3, 3) = ros_extrinsicMat_RT(3,3);

		// intrinsic
		intrisicMat.at<double>(0, 0) = un_intrisicMat.at<double>(0, 0) = ros_intrisicMat(0,0);
		intrisicMat.at<double>(0, 1) = 0.000000e+00;
		intrisicMat.at<double>(0, 2) = un_intrisicMat.at<double>(0, 2) = ros_intrisicMat(0,2);
		intrisicMat.at<double>(0, 3) = 0.000000e+00;
		intrisicMat.at<double>(1, 0) = 0.000000e+00;
		intrisicMat.at<double>(1, 1) = un_intrisicMat.at<double>(1, 1) = ros_intrisicMat(1,1);
		intrisicMat.at<double>(1, 2) = un_intrisicMat.at<double>(1, 2) = ros_intrisicMat(1,2);
		intrisicMat.at<double>(1, 3) = 0.000000e+00;
		intrisicMat.at<double>(2, 0) = 0.000000e+00;
		intrisicMat.at<double>(2, 1) = 0.000000e+00;
		intrisicMat.at<double>(2, 2) = un_intrisicMat.at<double>(2, 2) =1.000000e+00;
		intrisicMat.at<double>(2, 3) = 0.000000e+00;
		distCoeffs.at<double>(0) = ros_distCoeffs(0,0);
		distCoeffs.at<double>(1) = ros_distCoeffs(0,1);
		distCoeffs.at<double>(2) = ros_distCoeffs(0,2);
		distCoeffs.at<double>(3) = ros_distCoeffs(0,3);
		distCoeffs.at<double>(4) = ros_distCoeffs(0,4);
		if (child_frame_ != "base_link") {
		    static tf2_ros::StaticTransformBroadcaster br;
		    geometry_msgs::TransformStamped alias_transform_msg;
		    alias_transform_msg.header.stamp = ros::Time::now();
		    alias_transform_msg.transform.translation.x = 0.0;
		    alias_transform_msg.transform.translation.y = 0.0;
		    alias_transform_msg.transform.translation.z = 0.0;
		    alias_transform_msg.transform.rotation.x = 0.0;
		    alias_transform_msg.transform.rotation.y = 0.0;
		    alias_transform_msg.transform.rotation.z = 0.0;
		    alias_transform_msg.transform.rotation.w = 1.0;
		    alias_transform_msg.header.frame_id = child_frame_;
		    alias_transform_msg.child_frame_id = "base_link";
		    br.sendTransform(alias_transform_msg);
    	}
    	//pcl
    	pcl_points.reset(new PointCloudXYZIN());
    	scan_keypoint_enhance.reset(new PointCloudXYZRGB());
    	color_scan.reset(new PointCloudXYZRGB());
    	
		cout << "[Ros_parameter]: common/lidar_topic: " <<lidar_topic<< endl;
		cout << "[Ros_parameter]: common/image_topic: " <<image_topic<< endl;
		cout << "[Ros_parameter]:image_sizeW*H: " <<W<<"*"<<H<< endl;
		cout << "[Ros_parameter]: kiss_vio/extrinsicMat_RT: " << endl;
		cout << ros_extrinsicMat_RT << endl;
		cout << "[Ros_parameter]: kiss_vio/intrisicMatR: " << endl;
		cout << ros_intrisicMat << endl;
		cout << "[Ros_parameter]: config_.voxel_size: " << config_.voxel_size << endl;
		cout << "[Ros_parameter]: config_.max_points_per_voxel: " << config_.max_points_per_voxel<< endl;
		cout << "[Ros_parameter]: config_.initial_threshold: " << config_.initial_threshold << endl;
		cout << "[Ros_parameter]: config_.min_motion_th: " << config_.min_motion_th << endl;
		cout << "[Ros_parameter]: config_.max_range: " <<  config_.max_range << endl;
		cout << "[Ros_parameter]: config_.min_range: " << config_.min_range << endl;
}
void OdometryServer::resetParameters(){ 
	  pcl_points->clear();
	  scan_keypoint_enhance->clear();
	  color_scan->clear();
}
}  // namespace kiss_icp_ros

int main(int argc, char **argv) {
    ros::init(argc, argv, "kiss_icp");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    kiss_icp_ros::OdometryServer node(nh, nh_private);

    ros::spin();

    return 0;
}
