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

#include "KissLV.hpp"

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "kiss_icp/core/Deskew.hpp"
#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/core/RGB_completed.hpp"
using namespace std;
namespace kiss_icp::pipeline {

PointCloudXYZRGB::Ptr KissLV::RegisterFrame(const PointCloudXYZIN::Ptr &frame,
                                                    const std::vector<double> &timestamps, cv::Mat gray_image, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT) {
    const auto &deskew_frame = [&]() -> PointCloudXYZIN::Ptr {
        if (!config_.deskew) return frame;
        // TODO(Nacho) Add some asserts here to sanitize the timestamps

        //  If not enough poses for the estimation, do not de-skew
        const size_t N = poses().size();
        if (N <= 2) return frame;

        // Estimate linear and angular velocities
        const auto &start_pose = poses_[N - 2];
        const auto &finish_pose = poses_[N - 1];
        return livox_DeSkewScan(frame, timestamps, start_pose, finish_pose);
    }();
    return RegisterFrame(deskew_frame, gray_image, intrisicMat, extrinsicMat_RT);
}

PointCloudXYZRGB::Ptr KissLV::RegisterFrame(const PointCloudXYZIN::Ptr &frame, cv::Mat gray_image, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT) {
	
	
	PointCloudXYZIN::Ptr good_points(new PointCloudXYZIN());
	PointCloudXYZIN::Ptr scan_downsample(new PointCloudXYZIN());
	PointCloudXYZRGB::Ptr scan_downsample_color(new PointCloudXYZRGB());
	PointCloudXYZRGB::Ptr scan_downsample_enhance(new PointCloudXYZRGB());
	PointCloudXYZRGB::Ptr scan_keypoint_enhance(new PointCloudXYZRGB());
	//率选点及上色
	good_points =  kiss_icp::PCL_Preprocess(frame, config_.max_range, config_.min_range);
	//下采样
	scan_downsample = kiss_icp::PCL_VoxelDownsample(good_points, config_.voxel_size * 0.5);

	//判断是否退化场景
	cv::Mat depthMap = kiss_icp::Get_deepmap(scan_downsample, scan_downsample_color, gray_image ,intrisicMat, extrinsicMat_RT);
	if (scan_downsample->size()<800 && scan_downsample->size()>0){
		cout<<"deep_dense!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
		depthMap = kiss_icp::fillInFast(depthMap, config_.max_range);
		scan_downsample_enhance = kiss_icp::get_completed_rgbpoint(depthMap, intrisicMat, extrinsicMat_RT); //scan_source_color
		scan_keypoint_enhance = kiss_icp::PCL_Color_VoxelDownsample(scan_downsample_enhance, config_.voxel_size*0.5);
		}
	else{
		scan_downsample_enhance = kiss_icp::get_completed_rgbpoint(depthMap, intrisicMat, extrinsicMat_RT); //scan_source_color
		scan_keypoint_enhance = kiss_icp::PCL_Color_VoxelDownsample(scan_downsample_enhance, config_.voxel_size*1.5);
	}
	Vector6dVector cd_scan =  ConvertPointCloudToVector6(scan_downsample_enhance);
	Vector6dVector cd2_scan = ConvertPointCloudToVector6(scan_keypoint_enhance);
	const double sigma = GetAdaptiveThreshold(); 
	const auto prediction = GetPredictionModel(); //prediction 表示历史位姿信息中计算出来的从上一帧点云到当前点云的位姿变换。
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d(); //last_pose 可以看作是当前点云的上一帧点云在全局坐标系下的位姿。
    const auto initial_guess = last_pose * prediction;//一个从上一帧点云的位姿到当前点云的位姿变换
	const Sophus::SE3d new_pose = kiss_icp::Color_RegisterFrame(cd2_scan,  //下采样的点云     //kiss_icp空间下的RegisterFrame函数
                                                          color_local_map_,     //局部地图
                                                          initial_guess,   //初始猜测位姿
                                                          3.0 * sigma,    //ICP最大匹配距离
                                                          sigma / 3.0);  //权重 对于距离较小的点对，它们的权重较大，距离较大的点对，它们的权重较小
	//1.6 更新位姿和局部地图
    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);//根据新得到的位姿变换，更新自适应阈值
    color_local_map_.Color_Update(cd_scan, new_pose);//根据新得到的位姿变换，更新局部地图
    poses_.push_back(new_pose);//将新的位姿加入到历史位姿信息中
	return scan_keypoint_enhance;
}
//-------------------------------------------------------------------------------------------------------
double KissLV::GetAdaptiveThreshold() {
	if (!HasMoved()) {
	    return config_.initial_threshold;
	}
	return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d KissLV::GetPredictionModel() const {
	Sophus::SE3d pred = Sophus::SE3d();
	const size_t N = poses_.size();
	if (N < 2) return pred;
	return poses_[N - 2].inverse() * poses_[N - 1];
}
bool KissLV::HasMoved() {
	if (poses_.empty()) return false;
	const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
	return motion > 5.0 * config_.min_motion_th;
}
Vector6dVector KissLV::ConvertPointCloudToVector6(const pcl::PointCloud<ColorPointType>::ConstPtr& frame) {
	Vector6dVector points;
	points.reserve(frame->size());

	for (const auto& point : frame->points) {
		Vector6d vector;
		vector << point.x, point.y, point.z, static_cast<double>(point.b), static_cast<double>(point.g), static_cast<double>(point.r);
		points.push_back(vector);
	}
	return points;
}
}  // namespace kiss_icp::pipeline
