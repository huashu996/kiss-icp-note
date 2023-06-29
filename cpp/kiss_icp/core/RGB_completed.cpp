
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

cv::Mat FULL_KERNEL_3 = cv::Mat::ones(3, 3, CV_8UC1);
cv::Mat FULL_KERNEL_5 = cv::Mat::ones(5, 5, CV_8UC1);
cv::Mat FULL_KERNEL_7 = cv::Mat::ones(7, 7, CV_8UC1);
cv::Mat FULL_KERNEL_9 = cv::Mat::ones(9, 9, CV_8UC1);
cv::Mat FULL_KERNEL_31 = cv::Mat::ones(31, 31, CV_8UC1);
cv::Mat CROSS_KERNEL_3 = (cv::Mat_<uchar>(3, 3) << 0, 1, 0,
																  1, 1, 1,
																  0, 1, 0);
														  
cv::Mat CROSS_KERNEL_5 = (cv::Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0,
																  0, 0, 1, 0, 0,
																  1, 1, 1, 1, 1,
																  0, 0, 1, 0, 0,
																  0, 0, 1, 0, 0);
														  
cv::Mat DIAMOND_KERNEL_5 = (cv::Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0,
																	  0, 1, 1, 1, 0,
																	  1, 1, 1, 1, 1,
																	  0, 1, 1, 1, 0,
																	  0, 0, 1, 0, 0);
														  
cv::Mat CROSS_KERNEL_7 = (cv::Mat_<uchar>(7, 7) <<    0, 0, 0, 1, 0, 0, 0,
																	  0, 0, 0, 1, 0, 0, 0,
																	  0, 0, 0, 1, 0, 0, 0,
																	  1, 1, 1, 1, 1, 1, 1,
																	  0, 0, 0, 1, 0, 0, 0,
																	  0, 0, 0, 1, 0, 0, 0,
																	  0, 0, 0, 1, 0, 0, 0);		
cv::Mat DIAMOND_KERNEL_7 = (cv::Mat_<uchar>(7, 7) << 0, 0, 0, 1, 0, 0, 0,
																	  0, 0, 1, 1, 1, 0, 0,
																	  0, 1, 1, 1, 1, 1, 0,
																	  1, 1, 1, 1, 1, 1, 1,
																	  0, 1, 1, 1, 1, 1, 0,
																	  0, 0, 1, 1, 1, 0, 0,
																	  0, 0, 0, 1, 0, 0, 0);			
cv::Mat DIAMOND_KERNEL_9 = (cv::Mat_<uchar>(9, 9) << 0, 0, 0, 0, 1, 0, 0, 0, 0,
																	  0, 0, 0, 1, 1, 1, 0, 0, 0,
																	  0, 0, 1, 1, 1, 1, 1, 0, 0,
																	  0, 1, 1, 1, 1, 1, 1, 1, 0,
																	  1, 1, 1, 1, 1, 1, 1, 1, 1,
																	  0, 1, 1, 1, 1, 1, 1, 1, 0,
																	  0, 0, 1, 1, 1, 1, 1, 0, 0,
																	  0, 0, 0, 1, 1, 1, 0, 0, 0,
																	  0, 0, 0, 0, 1, 0, 0, 0, 0);			

namespace kiss_icp {
typedef pcl::PointXYZINormal PointTypeIN;
typedef pcl::PointCloud<PointTypeIN>    PointCloudXYZIN;
typedef pcl::PointXYZRGB ColorPointType;
typedef pcl::PointCloud<ColorPointType>  PointCloudXYZRGB;
using namespace std;
cv::Mat Get_deepmap(PointCloudXYZIN::Ptr laser_data, PointCloudXYZRGB::Ptr fusion_pcl_ptr, cv::Mat gray_image, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT) {
    int W = gray_image.cols;
	int H = gray_image.rows;
    cv::Mat depthMap(H, W, CV_64FC3); //包含深度、反射率、灰度
    depthMap.setTo(0.0);  // 将矩阵中的所有像素值设置为零
    fusion_pcl_ptr->reserve(laser_data->size());  // 预先分配点云内存空间
    cv::Mat pointLidar(4, 1, CV_64FC1);
    cv::Mat pointImage(3, 1, CV_64FC1);

    for (size_t j = 0; j < laser_data->size(); ++j) {
        pointLidar.at<double>(0, 0) = laser_data->points[j].x;
        pointLidar.at<double>(1, 0) = laser_data->points[j].y;
        pointLidar.at<double>(2, 0) = laser_data->points[j].z;
        pointLidar.at<double>(3, 0) = 1.0;
        pointImage = intrisicMat * extrinsicMat_RT * pointLidar;

        cv::Point2f pixelPoint(pointImage.at<double>(0, 0) / pointImage.at<double>(2, 0),
                               pointImage.at<double>(1, 0) / pointImage.at<double>(2, 0));

        // 如果点在图像范围内，则更新深度图
        if (int(pixelPoint.x) >= 0 && int(pixelPoint.x) < W && int(pixelPoint.y) >= 0 && int(pixelPoint.y) < H) {
            // 将对应像素位置的深度值设置为点的深度值
           	depthMap.at<cv::Vec3d>(pixelPoint.y, pixelPoint.x)[0] = pointImage.at<double>(2, 0); //深度
           	depthMap.at<cv::Vec3d>(pixelPoint.y, pixelPoint.x)[1] = laser_data->points[j].intensity; //反射率
           	depthMap.at<cv::Vec3d>(pixelPoint.y, pixelPoint.x)[2] = gray_image.at<float>(pixelPoint.y, pixelPoint.x); //灰度
            ColorPointType p;
            p.x = laser_data->points[j].x;
            p.y = laser_data->points[j].y;
            p.z = laser_data->points[j].z;
            // 点云颜色由图像上对应点确定
            p.b = 0;
            p.g = laser_data->points[j].intensity;
            p.r = gray_image.at<float>(pixelPoint.y, pixelPoint.x);
            fusion_pcl_ptr->push_back(p);
        }
    }
    return depthMap;
}


cv::Mat fillInFast(cv::Mat& depthMap, double maxDepth) {

	cv::Mat channels[3];
	cv::split(depthMap, channels);

	cv::Mat valid_pixels = (channels[0] > 0.1);
	cv::Mat depth_map_valid = depthMap.clone();

	cv::MatIterator_<bool> valid_pixels_it = valid_pixels.begin<bool>();
	cv::MatIterator_<cv::Vec3d> depth_map_valid_it = depth_map_valid.begin<cv::Vec3d>();

	for (; valid_pixels_it != valid_pixels.end<bool>(); ++valid_pixels_it, ++depth_map_valid_it) {
		if (*valid_pixels_it) {
			(*depth_map_valid_it)[0] = maxDepth - (*depth_map_valid_it)[0];
		}
	}

	depthMap = depth_map_valid;

	cv::Mat morph;
	cv::morphologyEx(depthMap, morph, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
	depthMap = morph;

	cv::Mat empty_pixels = (depthMap < 0.1);
	cv::Mat dilated;
	cv::dilate(depthMap, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
	cv::Mat depth_map_test = depthMap.clone();

	cv::MatIterator_<bool> empty_pixels_it = empty_pixels.begin<bool>();
	cv::MatIterator_<cv::Vec3d> depthMap_it = depth_map_test.begin<cv::Vec3d>();
	cv::MatIterator_<cv::Vec3d> dilated_it = dilated.begin<cv::Vec3d>();
	for (; empty_pixels_it != empty_pixels.end<bool>(); ++empty_pixels_it, ++depthMap_it, ++dilated_it) {
		if (*empty_pixels_it) {
			*depthMap_it = *dilated_it;
		}
	}
	depthMap = depth_map_test;

	cv::Mat valid_pixels2 = (depthMap > 0.1);
	cv::Mat depth_map_valid2 = depthMap.clone();

	cv::MatIterator_<bool> valid_pixels_it2 = valid_pixels2.begin<bool>();
	cv::MatIterator_<cv::Vec3d> depth_map_valid_it2 = depth_map_valid2.begin<cv::Vec3d>();

	for (; valid_pixels_it2 != valid_pixels2.end<bool>(); ++valid_pixels_it2, ++depth_map_valid_it2) {
		if (*valid_pixels_it2) {
			(*depth_map_valid_it2)[0] = maxDepth - (*depth_map_valid_it2)[0];
		}
	}

	depthMap = depth_map_valid2;

	return depthMap;
}


PointCloudXYZRGB::Ptr get_completed_rgbpoint(cv::Mat depthMap, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT) 
{
    CV_Assert(depthMap.type() == CV_64FC3);

    int rows = depthMap.rows;
    int cols = depthMap.cols;
    double fx = intrisicMat.at<double>(0, 0);
    double fy = intrisicMat.at<double>(1, 1);
    double cx = intrisicMat.at<double>(0, 2);
    double cy = intrisicMat.at<double>(1, 2);

    PointCloudXYZRGB::Ptr dense_pcl_ptr(new PointCloudXYZRGB());
    dense_pcl_ptr->points.reserve(rows * cols);

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            double depth = depthMap.at<cv::Vec3d>(y, x)[0];
            if (depth > 0) {
                double u = (x - cx) / fx;
                double v = (y - cy) / fy;
                cv::Mat pointCamera = (cv::Mat_<double>(4, 1) << depth * u, depth * v, depth, 1.0);
                cv::Mat lidarPoint = extrinsicMat_RT.inv() * pointCamera;

                ColorPointType p;
                p.x = lidarPoint.at<double>(0, 0);
                p.y = lidarPoint.at<double>(1, 0);
                p.z = lidarPoint.at<double>(2, 0);
                p.b = 0;
                p.g = depthMap.at<cv::Vec3d>(y, x)[1]; //i
                p.r = depthMap.at<cv::Vec3d>(y, x)[2]; //灰度

#pragma omp critical
                dense_pcl_ptr->points.push_back(p);
            }
        }
    }
    return dense_pcl_ptr;
}


}  // namespace core
