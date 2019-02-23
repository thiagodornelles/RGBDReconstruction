#ifndef UTIL_H
#define UTIL_H

#include <Core/Core.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Utility/Helper.h>
#include <IO/IO.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui.hpp>

using namespace open3d;
using namespace cv;
using namespace std;
using namespace rgbd;

shared_ptr<RGBDImage> ReadRGBDImage(
        const char* colorFilename, const char* depthFilename,
        const PinholeCameraIntrinsic &intrinsic, double depthTrunc,
        Mat *depthMask = NULL)
{
    double depthScale = 5000.0;
    bool convertRgbToIntensity = false;

    Image color, depth;
    ReadImage(colorFilename, color);
    ReadImage(depthFilename, depth);

    PrintDebug("Reading RGBD image : \n");
    PrintDebug("     Color : %d x %d x %d (%d bits per channel)\n",
               color.width_, color.height_,
               color.num_of_channels_, color.bytes_per_channel_ * 8);
    PrintDebug("     Depth : %d x %d x %d (%d bits per channel)\n",
               depth.width_, depth.height_,
               depth.num_of_channels_, depth.bytes_per_channel_ * 8);

    uint8_t *data = depth.data_.data();
    if(depthMask != NULL){
        depthMask->forEach<double>(
                    [data, depthMask](double &pixel, const int *position) -> void
        {
            if(pixel == 0){
                uint8_t *pData = &(data[(position[0] * depthMask->cols + position[1]) * 2]);
                *pData++ = 0;
                *pData = 0;
            }
        }
        );
    }
    std::shared_ptr<RGBDImage> rgbd_image =
            CreateRGBDImageFromColorAndDepth(color, depth, depthScale, depthTrunc,
                                             convertRgbToIntensity);

    return rgbd_image;
}

// Level comes in reverse order 2 > 1 > 0
Mat getMaskOfNormalsFromDepth(Mat &depth, PinholeCameraIntrinsic intrinsecs, int level,
                              bool dilateBorders = false){

    double scaleFactor = 1.0 / pow(2, level);
    double thresNormals[] = { 0.6, 0.6, 0.6 };
    double nearbNormals[] = { 7, 7, 3};

    Mat K = (Mat_<double>(3, 3) << intrinsecs.GetFocalLength().first * scaleFactor,
             0, intrinsecs.GetPrincipalPoint().first * scaleFactor,
             0, intrinsecs.GetFocalLength().second * scaleFactor,
             intrinsecs.GetPrincipalPoint().second * scaleFactor,
             0, 0, 1);

    RgbdNormals normalComp = RgbdNormals(depth.rows, depth.cols,
                                         CV_64F, K, nearbNormals[level],
                                         RgbdNormals::RGBD_NORMALS_METHOD_FALS);
    Mat tempDepth, normals;
    Mat refPoints3d;
    depth.convertTo(tempDepth, CV_16UC1, 5000);
    //    blur(tempDepth, tempDepth, Size(3,3));
    depthTo3d(tempDepth, K, refPoints3d);
    normalComp(refPoints3d, normals);

    Mat filtered = Mat::zeros(normals.rows, normals.cols, CV_64FC1);
    Vec3d camAxis (0, 0, -1);
    normals.forEach<Vec3d>(
        [&filtered, &camAxis, &thresNormals, &level](Vec3d &pixel, const int *pos) -> void
    {
        double uv = isnan(pixel[0]) ? 0 : pixel.dot(camAxis);
        filtered.at<double>(pos[0], pos[1]) = uv > thresNormals[level] ? 1 : 0;
    }
    );
    //    for (int r = 0; r < normals.rows; ++r) {
    //        for (int c = 0; c < normals.cols; ++c) {
    //            Vec3d pixel = normals.at<Vec3d>(r, c);
    //            double uv = isnan(pixel[0]) ? 0 : pixel.dot(camAxis);
    ////            double uv = pixel.dot(camAxis);
    //            filtered.at<double>(r, c) = uv > thresNormals[level] ? 1 : 0;
    //        }
    //    }
    if (dilateBorders){
        Mat element = getStructuringElement( MORPH_RECT,
                                             Size( 2*3 + 1, 2*3+1 ),
                                             Point( 3, 3 ) );
        erode( filtered, filtered, element );
    }
    return filtered;
}
#endif
