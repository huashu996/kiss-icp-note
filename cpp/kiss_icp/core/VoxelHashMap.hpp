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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
/*
一种常见的方法是使用Kd-tree进行初步的搜索，然后对搜索到的点云进行体素化并存储到哈希表中，以加快后续的搜索速度。这样可以兼顾Kd-tree和哈希表的优势，既可以利用Kd-tree高效的搜索能力快速定位到感兴趣的区域，又可以通过哈希表高效地存储大量的点云信息，并且在查询时快速地返回相应的体素信息。

另外，还可以使用哈希表来加速Kd-tree的搜索过程。具体来说，可以先将点云数据进行体素化，并使用哈希表将每个体素中包含的点云存储起来。在进行Kd-tree搜索时，可以使用哈希表快速定位到搜索区域中的体素，然后只对这些体素中包含的点云进行Kd-tree搜索，以减少搜索的计算量和时间。
*/
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace kiss_icp {
//使用结构体可以简化代码，使它们更加轻量级和易于使用。
struct VoxelHashMap {
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    using Voxel = Eigen::Vector3i;
    //自定义hash值的类型
    struct VoxelBlock {
        // buffer of points with a max limit of n_points
        //std::vector<Eigen::Vector3d>类型的成员变量points和一个整型成员变量num_points_
        std::vector<Eigen::Vector3d> points;
        int num_points_;
        inline void AddPoint(const Eigen::Vector3d &point) {
            if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
            //在添加之前，这个函数会检查points中已经存在的点数是否已经达到了num_points_的限制。如果points中的点数小于num_points_，则会将新的点添加到points的末尾。
        }
    };
    //这个哈希函数使用了一种常用的哈希函数算法——乘法哈希法（multiplicative hashing），将Voxel类型的三维整数向量转换为一个唯一的哈希值。
    //在点云处理中，通常会使用栅格化（Rasterization）或体素化（Voxelization）的方法将三维点云转换为三维整数向量，以便进行处理和存储。
    //在哈希表中，每个体素都被映射到一个唯一的哈希值，可以通过哈希值快速地查找对应的体素
    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };
	//构造函数VoxelHashMap用于初始化VoxelHashMap对象，并设置体素大小、最大距离和每个体素中最大点数等参数。
	//构造函数（Constructor）是一种特殊的成员函数，用于创建和初始化类或结构体的对象。
    explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    Vector3dVectorTuple GetCorrespondences(const Vector3dVector &points,
                                           double max_correspondance_distance) const;
    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;

    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_; //键的类型为 Voxel，值的类型为 VoxelBlock，哈希函数的类型为 VoxelHash。
    //VoxelHash是一个哈希函数对象，它接受一个Voxel类型的参数，并返回一个哈希值
    //哈希表能够高效地处理大量数据，并且支持快速的查找、插入和删除操作，因此可以大大提高点云数据的处理效率和准确性。
};
}  // namespace kiss_icp
