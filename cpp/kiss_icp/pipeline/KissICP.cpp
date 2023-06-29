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

#include "KissICP.hpp"

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "kiss_icp/core/Deskew.hpp"
#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"
using namespace std;
namespace kiss_icp::pipeline {
std::ostream& operator<<(std::ostream& os, const Sophus::SE3d& se3) {
    os << "Translation: " << se3.translation().transpose() << std::endl;
    os << "Rotation matrix:\n" << se3.rotationMatrix() << std::endl;
    return os;
}
KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps) {
    const auto &deskew_frame = [&]() -> std::vector<Eigen::Vector3d> {
        if (!config_.deskew) return frame;
        // TODO(Nacho) Add some asserts here to sanitize the timestamps

        //  If not enough poses for the estimation, do not de-skew
        const size_t N = poses().size();
        if (N <= 2) return frame;

        // Estimate linear and angular velocities
        const auto &start_pose = poses_[N - 2];
        const auto &finish_pose = poses_[N - 1];
        return DeSkewScan(frame, timestamps, start_pose, finish_pose);
    }();
    return RegisterFrame(deskew_frame);
}

KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame) {
    // 1.1点云预处理Preprocess the input cloud
    const auto &cropped_frame = Preprocess(frame, config_.max_range, config_.min_range);//处理输入点云，输出特定区域的点云

    // Voxelize
    //1.2 点云下采样
    const auto &[source, frame_downsample] = Voxelize(cropped_frame); 

    // Get motion prediction and adaptive_threshold
    //1.3 设置ICP搜索对应点的最大距离
    const double sigma = GetAdaptiveThreshold(); 
	
	//1.4 ICP先验预测位姿
    // Compute initial_guess for ICP
    const auto prediction = GetPredictionModel(); //prediction 表示历史位姿信息中计算出来的从上一帧点云到当前点云的位姿变换。
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d(); //last_pose 可以看作是当前点云的上一帧点云在全局坐标系下的位姿。
    const auto initial_guess = last_pose * prediction;//一个从上一帧点云的位姿到当前点云的位姿变换
	cout<<"sigma"<<sigma<<endl;
	cout<<"prediction"<<prediction<<endl;
	cout<<"initial_guess"<<initial_guess<<endl;
    // Run icp
    //1.5 ICP后验更新位姿
    const Sophus::SE3d new_pose = kiss_icp::RegisterFrame(source,  //下采样的点云     //kiss_icp空间下的RegisterFrame函数
                                                          local_map_,     //局部地图
                                                          initial_guess,   //初始猜测位姿
                                                          3.0 * sigma,    //ICP最大匹配距离
                                                          sigma / 3.0);  //权重 对于距离较小的点对，它们的权重较大，距离较大的点对，它们的权重较小
	//1.6 更新位姿和局部地图
    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);//根据新得到的位姿变换，更新自适应阈值
    local_map_.Update(frame_downsample, new_pose);//根据新得到的位姿变换，更新局部地图
    poses_.push_back(new_pose);//将新的位姿加入到历史位姿信息中
    return {frame, source}; //返回原始点云和下采样点云
}

KissICP::Vector3dVectorTuple KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const { 
    const auto voxel_size = config_.voxel_size;
    const auto frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    const auto source = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample}; //返回两个下采样点云
}
//初始化阈值
double KissICP::GetAdaptiveThreshold() {
    if (!HasMoved()) {
        return config_.initial_threshold;
    }
    return adaptive_threshold_.ComputeThreshold();
}

//这个函数会首先创建一个空的 SE3d 变量 pred，然后检查保存历史位姿信息的 poses_ 数组的长度 N。如果这个长度小于 2，表示历史位姿信息不足以计算预测模型，因此直接返回一个空的 SE3d 变量。否则，函数会计算倒数第二帧位姿到最新一帧位姿的变换，并将其赋值给变量 pred
Sophus::SE3d KissICP::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool KissICP::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

}  // namespace kiss_icp::pipeline
