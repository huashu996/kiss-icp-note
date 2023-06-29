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
#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <regex>
#include <string>
#include <vector>
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
using namespace cv;
namespace kiss_icp_ros::utils {
using PointCloud2 = sensor_msgs::PointCloud2;
using PointField = sensor_msgs::PointField;
using Header = std_msgs::Header;
typedef pcl::PointXYZINormal PointTypeIN;
typedef pcl::PointCloud<PointTypeIN>    PointCloudXYZIN;
std::string FixFrameId(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}
std::vector<livox_ros_driver::CustomMsgConstPtr> livox_data;
auto GetTimestampField(const PointCloud2 &msg) {
    PointField timestamp_field;
    for (const auto &field : msg.fields) {
        if ((field.name == "t" || field.name == "timestamp" || field.name == "time")) {
            timestamp_field = field;
        }
    }
    if (!timestamp_field.count) {
        throw std::runtime_error("Field 't', 'timestamp', or 'time'  does not exist");
    }
    return timestamp_field;
}

// Normalize timestamps from 0.0 to 1.0
auto NormalizeTimestamps(const std::vector<double> &timestamps) {
    const double max_timestamp = *std::max_element(timestamps.cbegin(), timestamps.cend());
    // check if already normalized
    if (max_timestamp < 1.0) return timestamps;
    std::vector<double> timestamps_normalized(timestamps.size());
    std::transform(timestamps.cbegin(), timestamps.cend(), timestamps_normalized.begin(),
                   [&](const auto &timestamp) { return timestamp / max_timestamp; });
    return timestamps_normalized;
}

auto ExtractTimestampsFromMsg(const PointCloud2 &msg, const PointField &field) {
    // Extract timestamps from cloud_msg
    const size_t n_points = msg.height * msg.width;
    std::vector<double> timestamps;
    timestamps.reserve(n_points);

    // Option 1: Timestamps are unsigned integers -> epoch time.
    if (field.name == "t" || field.name == "timestamp") {
        sensor_msgs::PointCloud2ConstIterator<uint32_t> msg_t(msg, field.name);
        for (size_t i = 0; i < n_points; ++i, ++msg_t) {  //访问每个点的时间戳
            timestamps.emplace_back(static_cast<double>(*msg_t));
        }
        // Covert to normalized time, between 0.0 and 1.0
        return NormalizeTimestamps(timestamps);
    }

    // Option 2: Timestamps are floating point values between 0.0 and 1.0
    // field.name == "timestamp"
    sensor_msgs::PointCloud2ConstIterator<double> msg_t(msg, field.name);
    for (size_t i = 0; i < n_points; ++i, ++msg_t) {
        timestamps.emplace_back(*msg_t);
    }
    return timestamps;
}

auto CreatePointCloud2Msg(const size_t n_points, const Header &header, bool timestamp = false) {
    PointCloud2 cloud_msg;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    cloud_msg.header = header;
    cloud_msg.header.frame_id = FixFrameId(cloud_msg.header.frame_id);
    cloud_msg.fields.clear();
    int offset = 0;
    offset = addPointField(cloud_msg, "x", 1, PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "y", 1, PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "z", 1, PointField::FLOAT32, offset);
    offset += sizeOfPointField(PointField::FLOAT32);
    if (timestamp) {
        // asuming timestamp on a velodyne fashion for now (between 0.0 and 1.0)
        offset = addPointField(cloud_msg, "time", 1, PointField::FLOAT64, offset);
        offset += sizeOfPointField(PointField::FLOAT64);
    }

    // Resize the point cloud accordingly
    cloud_msg.point_step = offset;
    cloud_msg.row_step = cloud_msg.width * cloud_msg.point_step;
    cloud_msg.data.resize(cloud_msg.height * cloud_msg.row_step);
    modifier.resize(n_points);
    return cloud_msg;
}

void FillPointCloud2XYZ(const std::vector<Eigen::Vector3d> &points, PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    for (size_t i = 0; i < points.size(); i++, ++msg_x, ++msg_y, ++msg_z) {
        const Eigen::Vector3d &point = points[i];
        *msg_x = point.x();
        *msg_y = point.y();
        *msg_z = point.z();
    }
}

void FillPointCloud2Timestamp(const std::vector<double> &timestamps, PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<double> msg_t(msg, "time");
    for (size_t i = 0; i < timestamps.size(); i++, ++msg_t) *msg_t = timestamps[i];
}

std::vector<double> GetTimestamps(const PointCloud2 &msg) {
    auto timestamp_field = GetTimestampField(msg); //获得点云的时间戳

    // Extract timestamps from cloud_msg
    std::vector<double> timestamps = ExtractTimestampsFromMsg(msg, timestamp_field); //获得点云每个点相对于点云的时间戳

    return timestamps;
}

std::vector<Eigen::Vector3d> PointCloud2ToEigen(const PointCloud2 &msg) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(msg.height * msg.width);
    sensor_msgs::PointCloud2ConstIterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(msg, "z");
    for (size_t i = 0; i < msg.height * msg.width; ++i, ++msg_x, ++msg_y, ++msg_z) {
        points.emplace_back(*msg_x, *msg_y, *msg_z);
    }
    return points;
}

PointCloud2 EigenToPointCloud2(const std::vector<Eigen::Vector3d> &points, const Header &header) {
    PointCloud2 msg = CreatePointCloud2Msg(points.size(), header);
    FillPointCloud2XYZ(points, msg);
    return msg;
}

PointCloud2 EigenToPointCloud2(const std::vector<Eigen::Vector3d> &points,
                               const std::vector<double> &timestamps,
                               const Header &header) {
    PointCloud2 msg = CreatePointCloud2Msg(points.size(), header, true);
    FillPointCloud2XYZ(points, msg);
    FillPointCloud2Timestamp(timestamps, msg);
    return msg;
}
std::vector<double> Livox_time(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in){
	  livox_data.clear();
      std::vector<double> timestamps;
      timestamps.clear();
      livox_data.push_back(livox_msg_in);
	  for (size_t j = 0; j < livox_data.size(); j++) {
		auto& livox_msg = livox_data[j];
		auto time_end = livox_msg->points.back().offset_time; //最后一点的时间戳偏移量
		for (unsigned int i = 0; i < livox_msg->point_num; ++i) {
		  double s = livox_msg->points[i].offset_time / (double)time_end;
		  timestamps.emplace_back(s);
		}
	  }
	  return timestamps;
	}
std::vector<Eigen::Vector3d> Livox_point(const livox_ros_driver::CustomMsg::ConstPtr& livox_msg_in){
	  livox_data.clear();
      livox_data.push_back(livox_msg_in);
      std::vector<Eigen::Vector3d> points;
      points.clear();
	  points.reserve(livox_data.size());
	  for (size_t j = 0; j < livox_data.size(); j++) {
		auto& livox_msg = livox_data[j];
		for (unsigned int i = 0; i < livox_msg->point_num; ++i) {
		  Eigen::Vector3d vector(livox_msg->points[i].x, livox_msg->points[i].y, livox_msg->points[i].z);
		  points.push_back(vector);
		}
	  }
	 return points;
	}
	
PointCloudXYZIN::Ptr Livox_pcl_point(const livox_ros_driver::CustomMsg::ConstPtr& livox_msg_in) {
    livox_data.clear();
    livox_data.push_back(livox_msg_in);
    PointCloudXYZIN::Ptr pcl_in(new PointCloudXYZIN);

    pcl_in->reserve(livox_msg_in->point_num);

    for (size_t j = 0; j < livox_data.size(); j++) {
        auto& livox_msg = livox_data[j];
        for (unsigned int i = 0; i < livox_msg->point_num; ++i) {
            PointTypeIN pt;
            pt.x = livox_msg->points[i].x;
            pt.y = livox_msg->points[i].y;
            pt.z = livox_msg->points[i].z;
            pt.intensity = livox_msg->points[i].reflectivity;

            pcl_in->push_back(pt);
        }
    }
    return pcl_in;
}

cv::Mat GuidedFilter(cv::Mat& I, cv::Mat& p, int r, float eps)
{
#define _cv_type_	CV_32FC1
	/*
	% GUIDEDFILTER   O(N) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, _cv_type_);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, _cv_type_);
	p = _p;

	//[hei, wid] = size(I);  
	int hei = I.rows;
	int wid = I.cols;

	r = 2 * r + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4 

	//mean_I = boxfilter(I, r) ./ N;  
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, _cv_type_, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;  
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, _cv_type_, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;  
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, _cv_type_, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.  
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;  
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, _cv_type_, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;  
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;     
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;  
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;  
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, _cv_type_, cv::Size(r, r));

	//mean_b = boxfilter(b, r) ./ N;  
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, _cv_type_, cv::Size(r, r));

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;  
	cv::Mat q = mean_a.mul(I) + mean_b;

	return q;
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}
cv::Mat ALTM_retinex(const cv::Mat& img, bool LocalAdaptation = true, bool ContrastCorrect = true)
{
    // Adaptive Local Tone Mapping Based on Retinex for HDR Image
    // https://github.com/IsaacChanghau/OptimizedImageEnhance
	/*
    const int cx = img.cols / 2;
    const int cy = img.rows / 2;
    cv::Mat temp, Lw;

    img.convertTo(temp, CV_32FC3);
    cv::cvtColor(temp, Lw, cv::COLOR_BGR2GRAY);

    // global adaptation
    double LwMax;
    cv::minMaxLoc(Lw, NULL, &LwMax);

    cv::Mat Lw_;
    cv::log(Lw + 1e-3f, Lw_);
    float LwAver = exp(cv::sum(Lw_)[0] / (img.rows * img.cols));

    // globally compress the dynamic range of a HDR scene
    cv::Mat Lg;
    cv::log(Lw / LwAver + 1.f, Lg);
    cv::divide(Lg, log(LwMax / LwAver + 1.f), Lg);

    // local adaptation
    cv::Mat Lout;
    if (LocalAdaptation) {
        int kernelSize = std::max(3, std::max(img.rows / 100, img.cols / 100));
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
        cv::Mat Lp;
        cv::dilate(Lg, Lp, kernel);
        cv::Mat Hg = GuidedFilter(Lg, Lp, 10, 0.01f);
        double eta = 36;
        double LgMax;
        cv::minMaxLoc(Lg, NULL, &LgMax);
        cv::Mat alpha = 1.0f + Lg * (eta / LgMax);
        cv::log(Lg + 1e-3f, Lout);
        cv::multiply(alpha, Lout, Lout);
        cv::normalize(Lout, Lout, 0, 255, cv::NORM_MINMAX);
    } else {
        cv::normalize(Lg, Lout, 0, 255, cv::NORM_MINMAX);
    }

    cv::Mat gain;
    gain.create(img.rows, img.cols, CV_32F);

    cv::MatIterator_<float> it_Lw = Lw.begin<float>();
    cv::MatIterator_<float> it_Lw_end = Lw.end<float>();
    cv::MatIterator_<float> it_Lout = Lout.begin<float>();
    cv::MatIterator_<float> it_gain = gain.begin<float>();

    for (; it_Lw != it_Lw_end; ++it_Lw, ++it_Lout, ++it_gain) {
        float x = *it_Lw;
        float y = *it_Lout;
        if (0 == x) *it_gain = y;
        else *it_gain = y / x;
    }

    cv::Mat out;
    std::vector<cv::Mat> bgr;
    cv::split(temp, bgr);

    if (ContrastCorrect) {
        // Contrast image correction method
        // https://www.researchgate.net/publication/220051147_Contrast_image_correction_method
        for (int i = 0; i < 3; i++) {
            bgr[i] = (gain.mul(bgr[i] + Lw) + bgr[i] - Lw) * 0.5f;
        }
    } else {
        for (int i = 0; i < 3; i++) {
            cv::multiply(bgr[i], gain, bgr[i]);
        }
    }
    cv::merge(bgr, out);
    out.convertTo(out, CV_8UC3);
    
	Mat ldrDrago, result;
	img.convertTo(ldrDrago, CV_32FC3, 1.0f/255);
	cvtColor(ldrDrago, ldrDrago, cv::COLOR_BGR2XYZ);
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.f, 1.f, 0.85f);
	tonemapDrago->process(ldrDrago, result);
	cvtColor(result, result, cv::COLOR_XYZ2BGR);
	result.convertTo(result, CV_8UC3, 255);
	*/
	cv::Mat hist_equalized_image;
	cv::cvtColor(img, hist_equalized_image, cv::COLOR_BGR2YCrCb);
	std::vector<cv::Mat> vec_channels;
	cv::split(hist_equalized_image, vec_channels);

	cv::Mat img_temp;
	cv::Size eqa_img_size = cv::Size(std::max(vec_channels[0].cols * 32.0 / 640, 4.0), std::max(vec_channels[0].cols * 32.0 / 640, 4.0));
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1, eqa_img_size);
	clahe->apply(vec_channels[0], img_temp);
	vec_channels[0] = img_temp;

	cv::merge(vec_channels, hist_equalized_image);
	cv::cvtColor(hist_equalized_image, hist_equalized_image, cv::COLOR_YCrCb2BGR);
    return hist_equalized_image;
}
cv::Mat Image_process(const sensor_msgs::CompressedImageConstPtr &msg, std::vector<std::vector<cv::Vec3b>>& image_color)
	{
		cv::Mat gray_image;
		try
		{
		    int H = image_color.size(); 
			int W = (H > 0) ? image_color[0].size() : 0;
		    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
			cv::Mat image = cv_ptr->image;
			cv::Mat new_image = ALTM_retinex(image);
			
		    //cv::undistort(new_image, undistortedImage, un_intrisicMat, distCoeffs);
		    for (int row = 0; row < H; row++)
		    {
		        for (int col = 0; col < W; col++)
		        {
		            image_color[row][col] = (cv::Vec3b)image.at<cv::Vec3b>(row, col);
		        }
		    }
			cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
		}
		catch (cv_bridge::Exception& e)
		{
		    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->format.c_str());
		}
		return gray_image;
	}
PointCloudXYZRGB::Ptr Eigen6dToPointCloud2(const std::vector<Vector6d> &points) {
    PointCloudXYZRGB::Ptr output_cloud(new PointCloudXYZRGB());
    for (const auto& point : points)
    {
        ColorPointType pcl_point;
        pcl_point.x = point[0];
        pcl_point.y = point[1];
        pcl_point.z = point[2];
        pcl_point.r = point[3];
        pcl_point.g = point[4];
        pcl_point.b = point[5];
        output_cloud->push_back(pcl_point);
    }
    return output_cloud;
}

PointCloudXYZRGB::Ptr get_color(PointCloudXYZIN::Ptr pcl_point, std::vector<std::vector<cv::Vec3b>> image_color, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT){
	int W = image_color[0].size(); // 假设所有行的长度相同，因此取第一行的长度
	int H = image_color.size();
	PointCloudXYZRGB::Ptr  fusion_pcl_ptr(new PointCloudXYZRGB());
	fusion_pcl_ptr->reserve(pcl_point->size());  // 预先分配点云内存空间
    cv::Mat pointLidar(4, 1, CV_64FC1);
    cv::Mat pointImage(3, 1, CV_64FC1);

    for (size_t j = 0; j < pcl_point->size(); ++j) {
        pointLidar.at<double>(0, 0) = pcl_point->points[j].x;
        pointLidar.at<double>(1, 0) = pcl_point->points[j].y;
        pointLidar.at<double>(2, 0) = pcl_point->points[j].z;
        pointLidar.at<double>(3, 0) = 1.0;

        pointImage = intrisicMat * extrinsicMat_RT * pointLidar;

        cv::Point2f pixelPoint(pointImage.at<double>(0, 0) / pointImage.at<double>(2, 0),
                               pointImage.at<double>(1, 0) / pointImage.at<double>(2, 0));

        // 如果点在图像范围内，则更新深度图
        if (pixelPoint.x >= 0 && pixelPoint.x < W && pixelPoint.y >= 0 && pixelPoint.y < H) {
            ColorPointType p;
            p.x = pcl_point->points[j].x;
            p.y = pcl_point->points[j].y;
            p.z = pcl_point->points[j].z;
            p.b = image_color[pixelPoint.y][pixelPoint.x][0];
            p.g = image_color[pixelPoint.y][pixelPoint.x][1];
            p.r = image_color[pixelPoint.y][pixelPoint.x][2];
            fusion_pcl_ptr->push_back(p);
        }
    }
    return fusion_pcl_ptr;
}
}  // namespace kiss_icp_ros::utils
