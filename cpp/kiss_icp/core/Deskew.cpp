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
#include "Deskew.hpp"

#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>
//在这个代码中，使用了namespace来封装了一个名为"kiss_icp"的命名空间，其中包含了DeSkewScan函数的实现。
namespace {
/// TODO(Nacho) Explain what is the very important meaning of this param
constexpr double mid_pose_timestamp{0.5}; //一个点云中的点的时间被归一化为0，1之间，位姿计算是假定在0.5时刻的
}  // namespace

namespace kiss_icp {
std::vector<Eigen::Vector3d> DeSkewScan(const std::vector<Eigen::Vector3d> &frame,
                                        const std::vector<double> &timestamps,
                                        const Sophus::SE3d &start_pose,
                                        const Sophus::SE3d &finish_pose) {
    const auto delta_pose = (start_pose.inverse() * finish_pose).log(); //scan开始到结束的变化位姿
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        const auto motion = Sophus::SE3d::exp((timestamps[i] - mid_pose_timestamp) * delta_pose);
        corrected_frame[i] = motion * frame[i];
    });
    return corrected_frame;
}
PointCloudXYZIN::Ptr livox_DeSkewScan(const PointCloudXYZIN::Ptr& frame,
                                                const std::vector<double> &timestamps,
                                                const Sophus::SE3d &start_pose,
                                                const Sophus::SE3d &finish_pose) {
    const auto delta_pose = (start_pose.inverse() * finish_pose).log();
    PointCloudXYZIN::Ptr corrected_frame(new PointCloudXYZIN);
    corrected_frame->points.resize(frame->points.size());
	tbb::parallel_for(tbb::blocked_range<size_t>(0, frame->points.size()), [&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i) {
		    const auto& point = frame->points[i];
		    const Eigen::Vector3d point_eigen(point.x, point.y, point.z);
		    const auto motion = Sophus::SE3d::exp((timestamps[i] - mid_pose_timestamp) * delta_pose);
		    const Eigen::Vector3d corrected_point = motion * point_eigen;

		    PointTypeIN corrected_point_pcl;
		    corrected_point_pcl.x = corrected_point.x();
		    corrected_point_pcl.y = corrected_point.y();
		    corrected_point_pcl.z = corrected_point.z();
		    corrected_point_pcl.intensity = point.intensity;

		    corrected_frame->points[i] = corrected_point_pcl;
		}
	});

	corrected_frame->width = frame->width;
	corrected_frame->height = frame->height;

	return corrected_frame;
}
}  // namespace kiss_icp
