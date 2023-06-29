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
}
void Color_TransformPoints(const Sophus::SE3d &T, Vector6dVector &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { 
                   Eigen::Vector4d homogeneous_point(point[0], point[1], point[2], 1.0);
				   Eigen::Vector4d transformed_homogeneous_point = T.matrix() * homogeneous_point;
				   Eigen::Vector6d transformed_point;
				   transformed_point << transformed_homogeneous_point[0], transformed_homogeneous_point[1], transformed_homogeneous_point[2], point[3], point[4], point[5];
  				   return transformed_point; });
}

//对齐点云
Sophus::SE3d Color_AlignClouds(const std::vector<Eigen::Vector6d> &source,
                         const std::vector<Eigen::Vector6d> &target,
                         double th) {
    auto compute_jacobian_and_residual = [&](auto i) {  //它计算每个点的雅可比矩阵和残差
     	const Eigen::Vector3d pos_residual = (source[i].template head<3>() - target[i].template head<3>()).template cast<double>();
        const Eigen::Vector3d color_residual = (source[i].template tail<3>() - target[i].template tail<3>()).template cast<double>();
        Eigen::Matrix3_6d J_r;   //创建雅可比矩阵
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat((source[i].template head<3>()).template cast<double>());
        Eigen::Matrix3_6d J_c;   // 创建颜色残差对优化变量的雅可比矩阵
    	J_c.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_c.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat((source[i].template tail<3>()).template cast<double>());
        return std::make_tuple(J_r,J_c, pos_residual, color_residual);
    };

    const auto &[JTJ, JTr] = tbb::parallel_reduce( //函数并行地计算雅可比矩阵和残差的和，以用于后续的优化
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        ResultTuple(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto Weight = [&](double residual2) { return square(th) / square(th + residual2); };
            auto &[JTJ_private, JTr_private] = J;  
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &[J_r, J_c,pos_residual, color_residual] = compute_jacobian_and_residual(i);
                const double w = Weight(pos_residual.squaredNorm());
                const double wc = Weight(color_residual.squaredNorm());
                JTJ_private.noalias() += J_r.transpose() * w * J_r;  //JTJ_private用于存储雅可比矩阵累加结果的矩阵
                JTr_private.noalias() += J_r.transpose() * w * pos_residual;  //JTr_private是用于存储残差累加结果的向量。
    			JTr_private.noalias() += 0.1*(J_c.transpose() * wc * color_residual); // 加入颜色残差部分 最佳0.5 ---------------作用不大
            }
            return J;
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });

    const Eigen::Vector6d x = JTJ.ldlt().solve(-JTr);  //其中JTJ是雅可比矩阵的和，JTr是残差的和，使用Eigen::Vector6d x = JTJ.ldlt().solve(-JTr)求解得到优化变量x。
    return Sophus::SE3d::exp(x); //将优化变量转换为刚体变换
}

constexpr int MAX_NUM_ITERATIONS_ = 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;

namespace kiss_icp {
Sophus::SE3d Color_RegisterFrame(const Vector6dVector &frame,
                           const Color_VoxelHashMap &color_voxel_map,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance,
                           double kernel) {
    if (color_voxel_map.Color_Empty()) return initial_guess;

    // Equation (9)
    Vector6dVector source = frame; //接收点云
    Color_TransformPoints(initial_guess, source);  //根据先验位姿更新函数

    // ICP-loop
    // ICP计算位姿

    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // Equation (10)
        const auto &[src, tgt] = color_voxel_map.Color_GetCorrespondences(source, max_correspondence_distance); //匹配对于点
        // Equation (11)
        auto estimation = Color_AlignClouds(src, tgt, kernel);
        // Equation (12)
        Color_TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (estimation.log().norm() < ESTIMATION_THRESHOLD_) break;
    }
    // Spit the final transformation
    return T_icp * initial_guess;
}

}  // namespace kiss_icp
