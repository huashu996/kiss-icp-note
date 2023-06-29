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
}  // namespace

namespace kiss_icp {

VoxelHashMap::Vector3dVectorTuple VoxelHashMap::GetCorrespondences(
    const Vector3dVector &points, double max_correspondance_distance) const {
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighboor = [&](const Eigen::Vector3d &point) {
   		 // 将点坐标映射到体素坐标系中，kx, ky, kz 分别表示点所在的 x, y, z 轴方向的体素坐标
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
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

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        Vector3dVector neighboors;
        neighboors.reserve(27 * max_points_per_voxel_);
         // 枚举 voxels 中每个体素，从哈希表 map_ 中查找相应体素中存储的所有点
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            auto search = map_.find(voxel);
            if (search != map_.end()) {
                const auto &points = search->second.points;
                if (!points.empty()) {
                    for (const auto &point : points) {
                        neighboors.emplace_back(point); // 将每个点加入到 neighboors 中
                    }
                }
            }
        });

        Eigen::Vector3d closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        // 枚举 neighboors 中的每个点，找到其中距离 point 最近的点
        std::for_each(neighboors.cbegin(), neighboors.cend(), [&](const auto &neighbor) {
            double distance = (neighbor - point).squaredNorm();
            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            }
        });

        return closest_neighbor;
    };
	/*	
		1.定义输入迭代器
		定义一个指向输入数据范围的迭代器，通常使用STL容器的迭代器或者原始指针。

		2.定义一个累加器
		定义一个对象来存储最终的结果，并在tbb::parallel_reduce函数调用时传递该对象。累加器可以是STL容器、结构体或任何具有适当类型和接口的自定义对象。

		3.定义一个Lambda函数来执行并行计算
		定义一个Lambda函数，该函数将被并行执行以处理输入数据范围中的每个元素，并将结果添加到累加器中。该Lambda函数应该以输入迭代器范围的tbb::blocked_range作为第一个参数。

		4.定义一个Lambda函数来执行并行规约
		定义一个Lambda函数，用于将多个线程执行的结果合并到一个累加器中，以便生成最终结果。

		5.调用tbb::parallel_reduce函数
		使用前面定义的迭代器、累加器和Lambda函数调用tbb::parallel_reduce函数来启动并行计算。

		6.处理结果
		最终结果存储在累加器对象中，可以根据需要进行处理。
	*/
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
         // 1.Range，指定待处理数据的迭代器范围
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        //  2.Identity 指定源点和最近邻点的容量
        ResultTuple(points.size()),
		//  3.1st lambda  它使用 GetClosestNeighboor 函数计算每个点的最近邻点，并根据最大对应距离筛选出满足条件的点
        [max_correspondance_distance, &GetClosestNeighboor](
            const tbb::blocked_range<points_iterator> &r, ResultTuple res) -> ResultTuple {
            auto &[src, tgt] = res;
             // 预分配存储空间
            src.reserve(r.size());
            tgt.reserve(r.size());
             // 迭代器搜索最近点
            for (const auto &point : r) {
                Eigen::Vector3d closest_neighboors = GetClosestNeighboor(point);
                if ((closest_neighboors - point).norm() < max_correspondance_distance) {
                    src.emplace_back(point);
                    tgt.emplace_back(closest_neighboors);
                }
            }
            // 返回计算结果
            return res;
        },
        // 4. 2nd lambda: Parallel reduction用于在并行计算完成后对结果进行归约
        [](ResultTuple a, const ResultTuple &b) -> ResultTuple {
             // 获取源点和最近邻点的向量引用
            auto &[src, tgt] = a;
            const auto &[srcp, tgtp] = b;
             // 将源点和最近邻点向量的内容合并到 a 中
            src.insert(src.end(),  //
                       std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),  //
                       std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        }
        );
    return std::make_tuple(source, target);
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {  //哈希表的所有点云
     // 创建一个vector，用来存储所有的点云
    std::vector<Eigen::Vector3d> points;
    // 为vector预留空间，避免多次调整vector大小
    points.reserve(max_points_per_voxel_ * map_.size());
    // 遍历整个HashMap中的VoxelBlock，将其中的点存储到vector中
    for (const auto &[voxel, voxel_block] : map_) {
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
void VoxelHashMap::Update(const Vector3dVector &points, const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}
//首先对点云进行位姿变换，即将所有点的坐标都从局部坐标系转换到全局坐标系
void VoxelHashMap::Update(const Vector3dVector &points, const Sophus::SE3d &pose) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}
//实现了向哈希表中添加点云的功能，将输入点云中的每一个点加入到对应的体素中。如果该体素已经存在，则直接在对应的体素块中添加点，否则新建一个体素块，并将点加入其中。
void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points) {
    //对于每一个点
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
    	//查看点属于哪个体素
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
 		// 在哈希表中查找是否已经有这个体素了
        auto search = map_.find(voxel);
         // 如果已经存在，则直接在这个体素块中添加点
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } 
        // 如果不存在，则在哈希表中插入一个新的体素块
        else {
            map_.insert({voxel, VoxelBlock{{point}, max_points_per_voxel_}});
        }
    });
}

//去除哈希表中距离原点较远的点
void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    for (const auto &[voxel, voxel_block] : map_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt - origin).squaredNorm() > (max_distance2)) {
            map_.erase(voxel);
        }
    }
}
}  // namespace kiss_icp
