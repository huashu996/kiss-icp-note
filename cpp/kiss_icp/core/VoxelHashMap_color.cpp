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
#include "VoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
using Vector6d = Eigen::Matrix<double, 6, 1>;
// This parameters are not intended to be changed, therefore we do not expose it
namespace {
struct ResultTuple {
    ResultTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};
struct ColorResultTuple {
    ColorResultTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Vector6d> source;
    std::vector<Vector6d> target;
};
}  // namespace

namespace kiss_icp {
Color_VoxelHashMap::Vector6dVectorTuple Color_VoxelHashMap::Color_GetCorrespondences(
    const Vector6dVector &points, double max_correspondance_distance) const {
    auto GetClosestNeighboor = [&](const Vector6d &point) {
        // 将点坐标映射到体素坐标系中，kx, ky, kz 分别表示点所在的 x, y, z 轴方向的体素坐标
        auto kx = static_cast<int>(int(point[0]) / voxel_size_);
        auto ky = static_cast<int>(int(point[1]) / voxel_size_);
        auto kz = static_cast<int>(int(point[2]) / voxel_size_);
        std::vector<Voxel> voxels;  //储存体素
         // 枚举以点所在体素为中心的 27 个体素
        voxels.reserve(27);
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                	// 将每个体素加入到 voxels 中
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        using Vector6dVector = std::vector<Vector6d>;
        Vector6dVector neighboors;
        neighboors.reserve(27 * max_points_per_voxel_);
        // 枚举 voxels 中每个体素，从哈希表 colormap_ 中查找相应体素中存储的所有点
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            auto search = colormap_.find(voxel);
            if (search != colormap_.end()) {
                const auto &points = search->second.points;
                if (!points.empty()) {
                    for (const auto &point : points) {
                        neighboors.emplace_back(point); // 将每个点加入到 neighboors 中
                    }
                }
            }
        });
		//neighboors容器中遍历每个点，找到与给定点point距离最近的点，并将其存储在closest_neighbor中。
        Vector6d closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        // 枚举 neighboors 中的每个点，找到其中距离 point 最近的点
        std::for_each(neighboors.cbegin(), neighboors.cend(), [&](const auto &neighbor) {
        	double color_distance = (0.1*neighbor.template tail<3>() - 0.1*point.template tail<3>()).template cast<double>().squaredNorm();
        	double distance = (neighbor.template head<3>() - point.template head<3>()).template cast<double>().squaredNorm()+0.5*color_distance;
        	//std::cout<<"distance"<<distance<<std::endl;
        	//std::cout<<"color_distance"<<distance<<std::endl;
            if (distance < closest_distance2) {
            	closest_distance2 = distance;
                closest_neighbor = neighbor;
            }
        });
        return closest_neighbor;
    };
	using points_iterator = std::vector<Vector6d>::const_iterator;

	const auto [source, target] = tbb::parallel_reduce(
		tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
		ColorResultTuple(points.size()),
		[max_correspondance_distance, &GetClosestNeighboor](
		    const tbb::blocked_range<points_iterator>& r, ColorResultTuple res) -> ColorResultTuple {
		    auto& [src, tgt] = res;
		    src.reserve(r.size());
		    tgt.reserve(r.size());
		    for (const auto& point : r) {
		        Vector6d closest_neighboors = GetClosestNeighboor(point);
		        
		        if (((closest_neighboors.template head<3>()).template cast<double>()-(point.template head<3>()).template cast<double>()).norm() < max_correspondance_distance) {
		            src.emplace_back(point);
		            tgt.emplace_back(closest_neighboors);
		        }
		    }
		    return res;
		},
		[](ColorResultTuple a, const ColorResultTuple& b) -> ColorResultTuple {
		    auto& [src, tgt] = a;
		    const auto& [srcp, tgtp] = b;
		    src.insert(src.end(),
		               std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
		    tgt.insert(tgt.end(),
		               std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
		    return a;
		});


    return std::make_tuple(source, target);
}

std::vector<Vector6d> Color_VoxelHashMap::Pointcloud() const {  //哈希表的所有点云
     // 创建一个vector，用来存储所有的点云
    std::vector<Vector6d> points;
    // 为vector预留空间，避免多次调整vector大小
    points.reserve(max_points_per_voxel_ * colormap_.size());
    // 遍历整个HashMap中的VoxelBlock，将其中的点存储到vector中
    for (const auto &[voxel, voxel_block] : colormap_) {
        (void)voxel;
          // 将VoxelBlock中的所有点存储到vector中
        for (const auto &point : voxel_block.points) {
            points.push_back(point);
        }
    }
    // 返回存储所有点云的vector
    return points;
}

//更新点云哈希表
void Color_VoxelHashMap::Color_Update(const Vector6dVector &points, const Eigen::Vector3d &origin) {
    Color_AddPoints(points);   //1加点
    Color_RemovePointsFarFromLocation(origin); //2 去点
}
//首先对点云进行位姿变换，即将所有点的坐标都从局部坐标系转换到全局坐标系
void Color_VoxelHashMap::Color_Update(const Vector6dVector &points, const Sophus::SE3d &pose) {
	Vector6dVector vectors;
	vectors.reserve(points.size()); // 提前分配足够的空间

	std::transform(points.cbegin(), points.cend(), std::back_inserter(vectors),
		           [&](const auto &point) {
		               Eigen::Vector3d transformed_point = (pose * point.template head<3>()).template cast<double>();
		               Vector6d vector;
		               vector << transformed_point[0], transformed_point[1], transformed_point[2], point[3], point[4], point[5];
		               return vector;
		           });
	const Eigen::Vector3d &origin = pose.translation();
	Color_Update(vectors, origin);


}
//实现了向哈希表中添加点云的功能，将输入点云中的每一个点加入到对应的体素中。如果该体素已经存在，则直接在对应的体素块中添加点，否则新建一个体素块，并将点加入其中。
void Color_VoxelHashMap::Color_AddPoints(const std::vector<Vector6d> &points) {
    //对于每一个点
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
    	//查看点属于哪个体素
        int voxel_x = static_cast<int>(int(point[0]) / voxel_size_);
        int voxel_y = static_cast<int>(int(point[1]) / voxel_size_);
        int voxel_z = static_cast<int>(int(point[2]) / voxel_size_);
        Voxel voxel;
        voxel << voxel_x, voxel_y, voxel_z;
 		// 在哈希表中查找是否已经有这个体素了
        auto search = colormap_.find(voxel);
         // 如果已经存在，则直接在这个体素块中添加点
        if (search != colormap_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } 
        // 如果不存在，则在哈希表中插入一个新的体素块
        else {
            colormap_.insert({voxel, Color_VoxelBlock{{point}, max_points_per_voxel_}});
        }
    });
}
//去除哈希表中距离原点较远的点
void Color_VoxelHashMap::Color_RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    for (const auto &[voxel, voxel_block] : colormap_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt.head<3>() - origin).squaredNorm() > (max_distance2)) {
            colormap_.erase(voxel);
        }
    }
}


}  // namespace kiss_icp
