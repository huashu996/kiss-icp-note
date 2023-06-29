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
#include "Preprocessing.hpp"

#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <vector>

namespace {
using Voxel = Eigen::Vector3i;
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};
}  // namespace

namespace kiss_icp {
std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d> &frame,
                                             double voxel_size) {
    tsl::robin_map<Voxel, Eigen::Vector3d, VoxelHash> grid; //grid用于存储体素下采样后的点云数据
    grid.reserve(frame.size());
    //对于每个点，将其坐标除以体素尺寸并向下取整，得到一个表示体素的整数坐标
    for (const auto &point : frame) {
        const auto voxel = Voxel((point / voxel_size).cast<int>());
        if (grid.contains(voxel)) continue;
        grid.insert({voxel, point});
    }
    //frame_downsampled，用于存储体素下采样后的点云数据。
    std::vector<Eigen::Vector3d> frame_dowsampled;
    frame_dowsampled.reserve(grid.size());
    for (const auto &[voxel, point] : grid) {
        (void)voxel;
        frame_dowsampled.emplace_back(point);
    }
    return frame_dowsampled;
}

std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                        double max_range,
                                        double min_range) {
    std::vector<Eigen::Vector3d> inliers;
    //这行代码使用 STL 中的 std::copy_if 算法，将输入集合 frame 中的元素复制到输出集合 inliers 中，但只有满足特定条件的元素才会被复制。
    std::copy_if(frame.cbegin(), frame.cend(), std::back_inserter(inliers), [&](const auto &pt) {
        const double norm = pt.norm();
        return norm < max_range && norm > min_range;
    });
    return inliers;
}

std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d> &frame) {
    constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        const auto &pt = frame[i];
        const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
        corrected_frame[i] =
            Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
    });
    return corrected_frame;
}

PointCloudXYZIN::Ptr PCL_VoxelDownsample(const PointCloudXYZIN::Ptr& input_cloud, double voxel_size) {
    // 创建一个新的点云对象用于存储下采样后的结果
    PointCloudXYZIN::Ptr downsampled_cloud(new PointCloudXYZIN());

    // 创建一个哈希表用于存储体素格和对应的点云坐标
    tsl::robin_map<Voxel, PointTypeIN, VoxelHash> grid;
    grid.reserve(input_cloud->size());

    // 创建一个互斥锁来保护插入操作
    tbb::spin_mutex grid_mutex;
    // 并行化遍历输入点云的每个点
    tbb::parallel_for(tbb::blocked_range<size_t>(0, input_cloud->size()), [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
            const auto& point = input_cloud->points[i];
            const auto voxel = Voxel((point.getVector3fMap() / voxel_size).cast<int>());
            if (grid.contains(voxel)) continue;
            tbb::spin_mutex::scoped_lock lock(grid_mutex);
            grid.insert({voxel, point});
            }
    });
    // 将哈希表中的点云坐标存储到下采样后的点云对象中
    downsampled_cloud->reserve(grid.size());
    for (const auto& [voxel, point] : grid) {
        (void)voxel;
        downsampled_cloud->push_back(point);
    }
    return downsampled_cloud;
}

pcl::PointCloud<ColorPointType>::Ptr PCL_Color_VoxelDownsample(const pcl::PointCloud<ColorPointType>::ConstPtr& input_cloud, double voxel_size) {
    // 创建一个新的点云对象用于存储下采样后的结果
    pcl::PointCloud<ColorPointType>::Ptr downsampled_cloud(new pcl::PointCloud<ColorPointType>);

    // 创建一个哈希表用于存储体素格和对应的点云坐标
    tsl::robin_map<Voxel, ColorPointType, VoxelHash> grid;
    grid.reserve(input_cloud->size());

    // 遍历输入点云的每个点
    for (const auto& point : input_cloud->points) {
        int voxel_x = static_cast<int>(point.x / voxel_size);
		int voxel_y = static_cast<int>(point.y / voxel_size);
		int voxel_z = static_cast<int>(point.z / voxel_size);
        Voxel voxel;
        voxel << voxel_x, voxel_y, voxel_z;
		if (grid.contains(voxel))
		    continue;
		// 将点云点插入到哈希表中
		grid.insert({voxel, point}); //将这个点插入到一个voxel中
    }

    // 将哈希表中的点云坐标存储到下采样后的点云对象中
    downsampled_cloud->reserve(grid.size());
    for (const auto& [voxel, point] : grid) {
        (void)voxel;
        downsampled_cloud->push_back(point);
    }
    return downsampled_cloud;
}
PointCloudXYZIN::Ptr PCL_Preprocess(const PointCloudXYZIN::Ptr& frame, double max_range,double min_range){
			Eigen::Vector3d origin(0.0, 0.0, 0.0); // 原点坐标
			PointCloudXYZIN::Ptr good_cloud(new PointCloudXYZIN());
			for (size_t j = 0; j < frame->size(); ++j)
			{
				const auto& point = frame->points[j];
				const Eigen::Vector3d point_eigen(point.x, point.y, point.z);
				const double distance = (point_eigen - origin).norm();
				if (distance > min_range && distance < max_range && pcl::isFinite(point))
				{
					PointTypeIN point_new;
					point_new.x = point.x;
					point_new.y = point.y;
					point_new.z = point.z;
					point_new.intensity = point.intensity;
					good_cloud->points.push_back(point_new);
				}
			}
			return good_cloud;
	}
}  // namespace kiss_icp
