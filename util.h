#ifndef UTIL_H
#define UTIL_H

#include <Core/Core.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Utility/Helper.h>
#include <IO/IO.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace open3d;
using namespace cv;
using namespace std;

shared_ptr<RGBDImage> ReadRGBDImage(
        const char* color_filename, const char* depth_filename,
        const PinholeCameraIntrinsic &intrinsic)
{
    double depth_scale = 5000.0, depth_trunc = 5000.0;
    bool convert_rgb_to_intensity = false;

    Image color, depth;
    ReadImage(color_filename, color);
    ReadImage(depth_filename, depth);

    PrintDebug("Reading RGBD image : \n");
    PrintDebug("     Color : %d x %d x %d (%d bits per channel)\n",
               color.width_, color.height_,
               color.num_of_channels_, color.bytes_per_channel_ * 8);
    PrintDebug("     Depth : %d x %d x %d (%d bits per channel)\n",
               depth.width_, depth.height_,
               depth.num_of_channels_, depth.bytes_per_channel_ * 8);

    std::shared_ptr<RGBDImage> rgbd_image =
            CreateRGBDImageFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
    return rgbd_image;
}

Mat normalsFromDepth(Mat &depth, PinholeCameraIntrinsic &intrisecs){

    Mat normals(depth.size(), CV_32FC3);
    for(int x = 1; x < depth.cols-1; ++x){
        for(int y = 1; y < depth.rows-1; ++y){
            // use float instead of double otherwise you will not get the correct result
            // check my updates in the original post. I have not figure out yet why this
            // is happening.
            float x1 = depth.at<ushort>(y, x+1);
            float x2 = depth.at<ushort>(y, x-1);
            float y1 = depth.at<ushort>(y+1, x);
            float y2 = depth.at<ushort>(y-1, x);

            float dzdx = x1 - x2;
            float dzdy = y1 - y2;

            Vec3f d(-dzdy, -dzdx, 0.001);

            Vec3f n = normalize(d);
            normals.at<Vec3f>(y, x) = n;
        }
    }
    return normals;
}
#endif
