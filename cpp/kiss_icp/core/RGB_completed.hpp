#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
typedef pcl::PointXYZINormal PointTypeIN;
typedef pcl::PointCloud<PointTypeIN>    PointCloudXYZIN;
typedef pcl::PointXYZRGB ColorPointType;
typedef pcl::PointCloud<ColorPointType>  PointCloudXYZRGB;
namespace kiss_icp {

cv::Mat Get_deepmap(PointCloudXYZIN::Ptr laser_data, PointCloudXYZRGB::Ptr fusion_pcl_ptr, cv::Mat gray_image, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT);
cv::Mat fillInFast(cv::Mat& depthMap, double maxDepth);
PointCloudXYZRGB::Ptr get_completed_rgbpoint(cv::Mat depthMap, cv::Mat intrisicMat, cv::Mat extrinsicMat_RT);
}//namespace
