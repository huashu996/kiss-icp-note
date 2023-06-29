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

#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <tuple>
#include <vector>
#include "kiss_icp/core/Deskew.hpp"
#include "kiss_icp/core/Threshold.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/core/RGB_completed.hpp"
struct KISSConfig {
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 5.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    // Motion compensation
    bool deskew = false;
};
namespace kiss_icp::pipeline {
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6dVector = std::vector<Vector6d>;
typedef pcl::PointXYZI PointTypeI;
typedef pcl::PointCloud<PointTypeI> PointCloudXYZI;
typedef pcl::PointXYZINormal PointTypeIN;
typedef pcl::PointCloud<PointTypeIN>    PointCloudXYZIN;
typedef pcl::PointXYZRGB ColorPointType;
typedef pcl::PointCloud<ColorPointType>  PointCloudXYZRGB;
class KissLV {
public:
	typedef pcl::PointXYZI PointTypeI;
	typedef pcl::PointCloud<PointTypeI> PointCloudXYZI;
	typedef pcl::PointXYZRGB ColorPointType;
	typedef pcl::PointCloud<ColorPointType>  PointCloudXYZRGB;
public:
	//类KissLV的构造函数
    explicit KissLV(const KISSConfig &config)
        : config_(config),
          color_local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel), //初始化显式构造函数，定义局部地图的体素尺寸，最大距离，初始化结构体
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {}

    KissLV() : KissLV(KISSConfig{}) {}

public:
	PointCloudXYZRGB::Ptr RegisterFrame(const PointCloudXYZIN::Ptr &frame,
                                                    const std::vector<double> &timestamps, cv::Mat gray_image, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT);
    PointCloudXYZRGB::Ptr RegisterFrame(const PointCloudXYZIN::Ptr &frame, cv::Mat gray_image, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT);                                              
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    bool HasMoved();
    Vector6dVector ConvertPointCloudToVector6(const pcl::PointCloud<ColorPointType>::ConstPtr& frame);
    

public:
    // Extra C++ API to facilitate ROS debugging
    std::vector<Vector6d> LocalMap() const { return color_local_map_.Pointcloud(); }; //发布局部点云
    std::vector<Sophus::SE3d> poses() const { return poses_; };  //发布位姿点

private:
    // KISS-ICP pipeline modules
    std::vector<Sophus::SE3d> poses_;  //包含多个 Sophus::SE3d 类型的数组
    KISSConfig config_;    //建立KISSConfig结构体对象
    Color_VoxelHashMap color_local_map_; //建立VoxelHashMap结构体对象
    AdaptiveThreshold adaptive_threshold_; //建立AdaptiveThreshold结构体对象
};

}  // namespace kiss_icp::pipeline
