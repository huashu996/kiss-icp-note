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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};
//点云变换
void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}
//对齐点云
Sophus::SE3d AlignClouds(const std::vector<Eigen::Vector3d> &source,
                         const std::vector<Eigen::Vector3d> &target,
                         double th) {
    auto compute_jacobian_and_residual = [&](auto i) {
        // 计算残差：residual = source[i] - target[i]
        const Eigen::Vector3d residual = source[i] - target[i];
        // 计算雅可比矩阵 J_r
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };
	 // 将JTJ和JTr初始化为单位矩阵和零矩阵
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        ResultTuple(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
        	//使得较大的残差具有较小的权重
            auto Weight = [&](double residual2) { return square(th) / square(th + residual2); };
            auto &[JTJ_private, JTr_private] = J;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                const double w = Weight(residual.squaredNorm());
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;
            }
            return J;
        },
        // 第二个Lambda：并行降低私有雅可比矩阵
        // 2nd Lambda: Parallel reduction of the private Jacboians
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });
	// 使用LDLT分解解线性系统：JTJ * x = -JTr
    const Eigen::Vector6d x = JTJ.ldlt().solve(-JTr);
    // 计算扭矩x的指数映射，得到SE3变换
    return Sophus::SE3d::exp(x);
}

constexpr int MAX_NUM_ITERATIONS_ = 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;

}  // namespace

namespace kiss_icp {

Sophus::SE3d RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                           const VoxelHashMap &voxel_map,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance,
                           double kernel) {
    if (voxel_map.Empty()) return initial_guess;

    //1.根据初始位姿更新点云
    std::vector<Eigen::Vector3d> source = frame; //接收点云
    TransformPoints(initial_guess, source);  //根据先验位姿更新函数

    // ICP-loop
    // 2.ICP计算位姿
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // 2.1 在voxel_map中找到对应的匹配点云
        const auto &[src, tgt] = voxel_map.GetCorrespondences(source, max_correspondence_distance); //匹配对于点
        // 2.2 两点云进行ICP匹配
        auto estimation = AlignClouds(src, tgt, kernel);
        // 2.3 更新位姿迭代
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (estimation.log().norm() < ESTIMATION_THRESHOLD_) break;
    }
    // Spit the final transformation
    return T_icp * initial_guess;
}

}  // namespace kiss_icp
