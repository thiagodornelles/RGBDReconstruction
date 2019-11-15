#ifndef UTIL_H
#define UTIL_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/RGBDImage.h>
#include <Open3D/Utility/Helper.h>
#include <Open3D/IO/ClassIO/ImageIO.h>
#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui.hpp>
#include "pointcloudextend.h"
#include "visibility.h"

#include <GL/gl.h>
#include <GL/glu.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>

using namespace open3d;
using namespace open3d::io;
using namespace open3d::camera;
using namespace open3d::utility;
using namespace cv;
using namespace std;
using namespace rgbd;
using namespace Eigen;

const int confThresh = 0;
double angleInsert = 0;

double interpolateDepth(Mat &depth, double &x, double &y, double &z);
double interpolateIntensityWithDepth(Mat &intensity, Mat &depth, double &x, double &y, double &z);

Image CreateDepthImageFromMat(Mat *matImage, Mat *normalMap = NULL)
{
    Image open3dImage;
    open3dImage.PrepareImage(matImage->cols, matImage->rows, 1, sizeof(uint16_t));
    uint8_t *data = open3dImage.data_.data();
    matImage->forEach<ushort>(
                [&](ushort &pixel, const int *pos) -> void
    {
        if(normalMap == NULL){
            uint8_t *pData = &(data[(pos[0] * matImage->cols + pos[1])*2]);
            *pData++ = pixel & 0xff;
            *pData = pixel >> 8;
        }
        else if(*normalMap->ptr(pos[0], pos[1]) > 0){
            uint8_t *pData = &(data[(pos[0] * matImage->cols + pos[1])*2]);
            *pData++ = pixel & 0xff;
            *pData = pixel >> 8;
        }
    });
    return open3dImage;
}


Image CreateRGBImageFromMat(Mat *matImage)
{
    Image open3dImage;
    open3dImage.PrepareImage(matImage->cols, matImage->rows, matImage->channels(), sizeof(uint8_t));
    uint8_t *data = open3dImage.data_.data();
    matImage->forEach<Vec3b>(
                [&data, &matImage](Vec3b &pixel, const int *position) -> void
    {
        uint8_t *pData = &(data[(position[0] * matImage->cols + position[1]) * 3]);
        *pData++ = pixel(2);
        *pData++ = pixel(1);
        *pData = pixel(0);
    });
    return open3dImage;
}

shared_ptr<RGBDImage> ReadRGBDImage(
        const char* colorFilename, const char* depthFilename,
        const PinholeCameraIntrinsic &intrinsic, double depthTrunc,
        double depthScale, Mat *depthMask = NULL)
{    
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

Mat convertDepthTo16bit(std::shared_ptr<Image> depth, double depthScale){
    Mat depth16;
    Mat depth32(depth->height_, depth->width_, CV_32FC1, depth->data_.data());
    depth32.convertTo(depth16, CV_16UC1, depthScale);
    return depth16;
}

Mat convertDepthToDouble(std::shared_ptr<Image> depth, double depthScale){
    Mat depth16;
    Mat depth32(depth->height_, depth->width_, CV_32FC1, depth->data_.data());
    depth32.convertTo(depth16, CV_64FC1, 1);
    return depth16;
}

Mat getNormalMap(Mat &depth, double depthScale, bool dilateBorders = false, double angleThres = 1.222){
    Mat normals = Mat::zeros(depth.rows, depth.cols, CV_32FC3);
    Mat temp;
    depth.convertTo(temp, CV_32FC1, 1.0/depthScale);
    Mat filteredDepth;
    GaussianBlur(temp, filteredDepth, Size(3,3), 1);
    for(int x = 1; x < depth.cols - 1; ++x){
        for(int y = 1; y < depth.rows - 1; ++y){
            if (*filteredDepth.ptr<float>(y, x) == 0)
                continue;

            Vec3f t(x,y-1, *filteredDepth.ptr<float>(y-1, x) * depthScale);
            Vec3f l(x-1,y, *filteredDepth.ptr<float>(y, x-1) * depthScale);
            Vec3f c(x,y, *filteredDepth.ptr<float>(y, x) * depthScale);

            Vec3f d = (l-c).cross(t-c);

            Vec3f n = normalize(d);
            *normals.ptr<Vec3f>(y,x) = n;
        }
    }

    Mat filtered = Mat::zeros(normals.rows, normals.cols, CV_64FC1);
    Vec3f camAxis (0, 0, 1);
    normals.forEach<Vec3f>([&](Vec3f &pixel, const int *pos) -> void
    {
        double uv = pixel.dot(camAxis);
        //        uv = cos(acos(uv) * angleThres);
        uv = -cos(uv)+1;
        *filtered.ptr<double>(pos[0], pos[1]) = uv;// < 0 ? 0 : uv;
    }
    );

    if (dilateBorders){
        erode(filtered, filtered, getStructuringElement(MORPH_CROSS, Size(3, 3)));
    }

    return filtered;
}


Mat getNormalMapFromPointCloud(shared_ptr<PointCloud> pointCloud, PinholeCameraIntrinsic intrinsics, bool dilateBorders = false, double angleThres = 1.222){
    int width = intrinsics.width_;
    int height = intrinsics.height_;

    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    Mat normals(intrinsics.height_, intrinsics.width_, CV_64FC3);
    EstimateNormals(*pointCloud);
    OrientNormalsTowardsCameraLocation(*pointCloud);

    for (int i = 0; i < pointCloud->points_.size(); i++) {

        Eigen::Vector3d pcdPoint = pointCloud->points_[i];
        Eigen::Vector4d refPoint3d;
        refPoint3d(0) = pcdPoint(0);
        refPoint3d(1) = pcdPoint(1);
        refPoint3d(2) = pcdPoint(2);
        refPoint3d(3) = 1;

        double invTransfZ = 1.0 / refPoint3d(2);

        //******* BEGIN Projection of PointCloud on the image plane ********
        double transfC = (refPoint3d(0) * fx) * invTransfZ + cx;
        double transfR = (refPoint3d(1) * fy) * invTransfZ + cy;
        int transfR_int = static_cast<int>(round(transfR));
        int transfC_int = static_cast<int>(round(transfC));
        //******* END Projection of PointCloud on the image plane ********

        //Checks if this pixel projects inside of the image
        if((transfR_int >= 0 && transfR_int < height) &&
                (transfC_int >= 0 && transfC_int < width)) {

            Eigen::Vector3d n = pointCloud->normals_[i];
            Vec3d normal(n(0), n(1), n(2));
            *normals.ptr<Vec3d>(transfR_int, transfC_int) = normal;
        }
    }

    Mat filtered = Mat::zeros(normals.rows, normals.cols, CV_64FC1);
    normals.forEach<Vec3d>(
                [&](Vec3d &pixel, const int *pos) -> void
    {
        Vec3d camAxis(0, 0, -1);
        double uv = pixel.dot(camAxis);
        uv = cos(acos(uv) * angleThres);
        filtered.at<double>(pos[0], pos[1]) = uv < 0.0 ? 0 : uv;
    }
    );

    return filtered;
}

// Level comes in reverse order 2 > 1 > 0
Mat getNormalMapFromDepth(Mat &depth, PinholeCameraIntrinsic intrinsics, int level,
                          double depthScale){

    double scaleFactor = 1.0 / pow(2, level);
    double nearbNormals[] = { 5, 5, 3, 3};

    Mat K = (Mat_<double>(3, 3) << intrinsics.GetFocalLength().first * scaleFactor,
             0, intrinsics.GetPrincipalPoint().first * scaleFactor,
             0, intrinsics.GetFocalLength().second * scaleFactor,
             intrinsics.GetPrincipalPoint().second * scaleFactor,
             0, 0, 1);

    RgbdNormals normalComp = RgbdNormals(depth.rows, depth.cols,
                                         CV_64F, K, nearbNormals[level],
                                         RgbdNormals::RGBD_NORMALS_METHOD_FALS);
    Mat tempDepth, normals;
    Mat refPoints3d;
    depth.convertTo(tempDepth, CV_16UC1, depthScale);
    depthTo3d(tempDepth, K, refPoints3d);
    normalComp(refPoints3d, normals);

    return normals;
}

Mat getNormalWeight(Mat normalMap, Mat depth, PinholeCameraIntrinsic intrinsics,
                    bool dilateBorders = false, double angleThres = 1.222,
                    double angleCut = 0){

    Mat filtered = Mat::zeros(normalMap.rows, normalMap.cols, CV_64FC1);
    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;
    double width = intrinsics.width_;
    double height = intrinsics.height_;
    double fovX = 2 * atan(width / (2 * fx)) * 180.0 / CV_PI;
    double fovY = 2 * atan(height / (2 * fy)) * 180.0 / CV_PI;
    float aspectRatio = depth.cols / (float)depth.rows; // assuming width > height
    double maxRadius = sqrt(cx*cx + cy*cy);

    normalMap.forEach<Vec3d>([&](Vec3d &pixel, const int *pos) -> void
    {
        if(depth.at<double>(pos[0], pos[1]) > 0 ){
            float px = (2 * ((pos[1] + 0.5) / depth.cols) - 1) *
                    tan(fovX / 2 * M_PI / 180) * aspectRatio;
            float py = (1 - 2 * ((pos[0] + 0.5) / depth.rows)) *
                    tan(fovY / 2 * M_PI / 180);
            //            cerr << "(" << px << "," << py << ")\n";
            double x2 = (height/2 - pos[0]) * (height/2 - pos[0]);
            double y2 = (width/2 - pos[1]) * (width/2 - pos[1]);
            double radius = sqrt(x2 + y2);
            double wRadius = (1 - radius/maxRadius);
            //            wRadius = wRadius < 0.3 ? 0 : 1;
            //            wRadius = 1;
            //            Vec3d camAxis(px, py, -1);
            Vec3d camAxis(0, 0, -1);
            camAxis = normalize(camAxis);
            pixel = normalize(pixel);
            double uv = pixel.dot(camAxis);
            uv = cos(acos(uv) * angleThres);
            filtered.at<double>(pos[0], pos[1]) = uv < angleCut ? 0 : uv;// * wRadius;
        }
    }
    );


    if (dilateBorders){
        erode(filtered, filtered, getStructuringElement(MORPH_CROSS, Size(3, 3)));
    }

    return filtered;
}

void projectPointCloudExtended(PointCloudExtended pointCloud, double maxDist,
                               Eigen::Matrix4d Rt, PinholeCameraIntrinsic intrinsics,
                               double depthScale, double radius, int initialFrame, int actualFrame,
                               Mat &depthMap, Mat &indexMap){

    int width = intrinsics.width_;
    int height = intrinsics.height_;

    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    depthMap.create(height, width, CV_16UC1);
    depthMap.setTo(65535);
    indexMap.create(height, width, CV_32SC1);
    indexMap.setTo(-1);

    for (int i = 0; i < pointCloud.points_.size(); i++) {

        //        if(pointCloud.hitCounter_[i] < confThresh && actualFrame + initialFrame > confThresh){
        //            continue;
        //        }

        Eigen::Vector3d pcdPoint = pointCloud.points_[i];
        Eigen::Vector4d refPoint3d;
        refPoint3d(0) = pcdPoint(0);
        refPoint3d(1) = pcdPoint(1);
        refPoint3d(2) = pcdPoint(2);
        refPoint3d(3) = 1;
        refPoint3d = Rt * refPoint3d;
        double invTransfZ = 1.0 / refPoint3d(2);

        //******* BEGIN Projection of PointCloud on the image plane ********
        double transfC = (refPoint3d(0) * fx) * invTransfZ + cx;
        double transfR = (refPoint3d(1) * fy) * invTransfZ + cy;
        int transfR_int = static_cast<int>(round(transfR));
        int transfC_int = static_cast<int>(round(transfC));
        //******* END Projection of PointCloud on the image plane ********

        //Checks if this pixel projects inside of the image
        if((transfR_int > 0 && transfR_int < height) &&
                (transfC_int > 0 && transfC_int < width)) {

            ushort lastZ = *depthMap.ptr<ushort>(transfR_int, transfC_int);
            ushort value = refPoint3d(2) * depthScale;

            if(value < lastZ){
                *depthMap.ptr<ushort>(transfR_int, transfC_int) = value;
                *indexMap.ptr<int>(transfR_int, transfC_int) = i;
                //circle(indexMap, Point(transfC_int, transfR_int), radius, i, -1);
                //circle(depthMap, Point(transfC_int, transfR_int), radius, value, -1);
            }
        }
    }
    depthMap.setTo(0, depthMap == 65535);
    medianBlur(depthMap, depthMap, 3);
}

void projectPointCloudWithColor(PointCloud pointCloud, double maxDist, double depthScale,
                                Eigen::Matrix4d Rt, PinholeCameraIntrinsic intrinsics,
                                Mat &depthMap, Mat &colorMap, double radius = 1){

    int width = intrinsics.width_;
    int height = intrinsics.height_;

    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    depthMap.create(height, width, CV_16UC1);
    depthMap.setTo(65535);
    colorMap.create(height, width, CV_8UC3);
    colorMap.setTo(0);

    for (int i = 0; i < pointCloud.points_.size(); i++) {

        Eigen::Vector3d pcdPoint = pointCloud.points_[i];
        Eigen::Vector4d refPoint3d;
        refPoint3d(0) = pcdPoint(0);
        refPoint3d(1) = pcdPoint(1);
        refPoint3d(2) = pcdPoint(2);
        if(refPoint3d(2) > maxDist) continue;
        refPoint3d(3) = 1;
        refPoint3d = Rt * refPoint3d;
        double invTransfZ = 1.0 / refPoint3d(2);

        //******* BEGIN Projection of PointCloud on the image plane ********
        double transfC = (refPoint3d(0) * fx) * invTransfZ + cx;
        double transfR = (refPoint3d(1) * fy) * invTransfZ + cy;
        int transfR_int = static_cast<int>(round(transfR));
        int transfC_int = static_cast<int>(round(transfC));
        //******* END Projection of PointCloud on the image plane ********

        //Checks if this pixel projects inside of the image
        if((transfR_int > 0 && transfR_int < height) &&
                (transfC_int > 0 && transfC_int < width)) {

            ushort lastZ = *depthMap.ptr<ushort>(transfR_int, transfC_int);
            ushort value = refPoint3d(2) * depthScale;
            Vec3b color = Vec3b(pointCloud.colors_[i](2)*255,pointCloud.colors_[i](1)*255, pointCloud.colors_[i](0)*255);
            if(value < lastZ){
                    *depthMap.ptr<ushort>(transfR_int, transfC_int) = value;
                    //*colorMap.ptr<Vec3b>(transfR_int, transfC_int) = color;
                    circle(colorMap, Point(transfC_int, transfR_int), radius, color, -1);
            }
        }
    }
    depthMap.setTo(0, depthMap == 65535);
    medianBlur(depthMap, depthMap, 3);
}

void projectMesh(shared_ptr<TriangleMesh> mesh, double maxDist, double depthScale,
                 Eigen::Matrix4d Rt, PinholeCameraIntrinsic intrinsics,
                 Mat &depthMap, Mat &colorMap){

    int width = intrinsics.width_;
    int height = intrinsics.height_;

    visualization::Visualizer vis;
    vis.CreateVisualizerWindow("render", width, height, 0, 0, false);
    vis.AddGeometry({mesh});
    PinholeCameraParameters params;
    vis.GetViewControl().ConvertToPinholeCameraParameters(params);    
    intrinsics.intrinsic_matrix_(0, 2) = width/2 - 0.5;
    intrinsics.intrinsic_matrix_(1, 2) = height/2 - 0.5;
    params.intrinsic_ = intrinsics;
    params.extrinsic_ = Rt;
    vis.GetViewControl().ConvertFromPinholeCameraParameters(params);
    shared_ptr<Image> image = vis.CaptureDepthFloatBuffer(true);
    depthMap = convertDepthTo16bit(image, depthScale);
    vis.DestroyVisualizerWindow();
/*
    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    depthMap.create(height, width, CV_16UC1);
    depthMap.setTo(65535);
    colorMap.create(height, width, CV_8UC3);
    colorMap.setTo(0);

    for (int i = 0; i < mesh->triangles_.size(); i++) {
        for(int j = 0; j < 3; j++){
            Eigen::Vector3d point = mesh->vertices_[mesh->triangles_[i](j)];
            Eigen::Vector4d refPoint3d;
            refPoint3d(0) = point(0);
            refPoint3d(1) = point(1);
            refPoint3d(2) = point(2);
            refPoint3d(3) = 1;
            refPoint3d = Rt * refPoint3d;
            double invTransfZ = 1.0 / refPoint3d(2);

            //******* BEGIN Projection of PointCloud on the image plane ********
            double transfC = (refPoint3d(0) * fx) * invTransfZ + cx;
            double transfR = (refPoint3d(1) * fy) * invTransfZ + cy;
            int transfR_int = static_cast<int>(round(transfR));
            int transfC_int = static_cast<int>(round(transfC));
            //******* END Projection of PointCloud on the image plane ********

            if(refPoint3d(2) > maxDist)
                break;

            //Checks if this pixel projects inside of the image
            if((transfR_int > 0 && transfR_int < height) &&
                    (transfC_int > 0 && transfC_int < width)) {

                ushort lastZ = *depthMap.ptr<ushort>(transfR_int, transfC_int);
                ushort value = refPoint3d(2) * depthScale;
                Eigen::Vector3d col = mesh->vertex_colors_[mesh->triangles_[i](j)];
                Vec3b color = Vec3b(col(2) * 255, col(1) * 255, col(0) * 255);
                if(value < lastZ){
                    //*depthMap.ptr<ushort>(transfR_int, transfC_int) = value;
                    *colorMap.ptr<Vec3b>(transfR_int, transfC_int) = color;                    
                    //circle(colorMap, Point(transfC_int, transfR_int), 1, color, -1);
                }
            }
        }
    }
    depthMap.setTo(0, depthMap == 65535);
*/
}

void projectPointCloud(PointCloud pointCloud, double maxDist, double depthScale,                       
                       Eigen::Matrix4d Rt, PinholeCameraIntrinsic intrinsics,
                       Mat &depthMap, Mat &indexMap, double radius = 1){

    int width = intrinsics.width_;
    int height = intrinsics.height_;

    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    depthMap.create(height, width, CV_16UC1);
    depthMap.setTo(65535);
    indexMap.create(height, width, CV_32SC1);
    indexMap.setTo(-1);

    for (int i = 0; i < pointCloud.points_.size(); i++) {

        Eigen::Vector3d pcdPoint = pointCloud.points_[i];
        Eigen::Vector4d refPoint3d;
        refPoint3d(0) = pcdPoint(0);
        refPoint3d(1) = pcdPoint(1);
        refPoint3d(2) = pcdPoint(2);
        if(refPoint3d(2) > maxDist) continue;
        refPoint3d(3) = 1;
        refPoint3d = Rt * refPoint3d;
        double invTransfZ = 1.0 / refPoint3d(2);

        //******* BEGIN Projection of PointCloud on the image plane ********
        double transfC = (refPoint3d(0) * fx) * invTransfZ + cx;
        double transfR = (refPoint3d(1) * fy) * invTransfZ + cy;
        int transfR_int = static_cast<int>(round(transfR));
        int transfC_int = static_cast<int>(round(transfC));
        //******* END Projection of PointCloud on the image plane ********

        //Checks if this pixel projects inside of the image
        if((transfR_int > 0 && transfR_int < height) &&
                (transfC_int > 0 && transfC_int < width)) {

            ushort lastZ = *depthMap.ptr<ushort>(transfR_int, transfC_int);
            ushort value = refPoint3d(2) * depthScale;

            if(value < lastZ){
                *depthMap.ptr<ushort>(transfR_int, transfC_int) = value;
                *indexMap.ptr<int>(transfR_int, transfC_int) = i;
                //circle(indexMap, Point(transfC_int, transfR_int), radius, i, -1);
                //circle(depthMap, Point(transfC_int, transfR_int), radius, value, -1);
            }
        }
    }
    depthMap.setTo(0, depthMap == 65535);
    medianBlur(depthMap, depthMap, 3);
}

Mat transfAndProject(Mat &depthMap, double maxDist, Eigen::Matrix4d Rt, PinholeCameraIntrinsic intrinsics){

    double nPoints = depthMap.cols * depthMap.rows;
    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    Eigen::Vector4d refPoint3d;
    Mat projected = Mat::zeros(depthMap.rows, depthMap.cols, CV_64FC1);

    for (int i = 0; i < nPoints; ++i) {
        int py = i / depthMap.cols;
        int px = i % depthMap.cols;
        double z = *depthMap.ptr<double>(py, px);
        if(z > maxDist) continue;
        double x = (px - cx) * z * 1/fx;
        double y = (py - cy) * z * 1/fy;

        refPoint3d(0) = x;
        refPoint3d(1) = y;
        refPoint3d(2) = z;
        refPoint3d(3) = 1;

        refPoint3d = Rt * refPoint3d;
        double invTransfZ = 1.0 / refPoint3d(2);

        //******* BEGIN Projection of PointCloud on the image plane ********
        double transfC = (refPoint3d(0) * fx) * invTransfZ + cx;
        double transfR = (refPoint3d(1) * fy) * invTransfZ + cy;
        int transfR_int = static_cast<int>(round(transfR));
        int transfC_int = static_cast<int>(round(transfC));
        //******* END Projection of PointCloud on the image plane ********

        //Checks if this pixel projects inside of the source image
        if((transfR_int >= 0 && transfR_int < depthMap.rows) &&
                (transfC_int >= 0 && transfC_int < depthMap.cols) &&
                refPoint3d(2) > 0.1 && refPoint3d(2) < maxDist) {
            *projected.ptr<double>(transfR_int, transfC) = refPoint3d(2);
        }

    }
    return projected;
}

void removeUnstablePoints(shared_ptr<PointCloudExtended> model){
    //Aging of points
    vector<int> keepPoints;
    for (int i = 0; i < model->points_.size(); ++i) {
        model->frameCounter_[i]++;
        if(model->frameCounter_[i] > 3 && model->confidence_[i] < 2){
            model->frameCounter_[i] = 0;
            keepPoints.push_back(0);
        }
        else{
            keepPoints.push_back(1);
        }
    }
    int k = 0;
    int totalPoints = model->points_.size();
    for(int i = 0; i < model->points_.size(); i++){
        if(keepPoints[i] > 0){
            model->points_[k] = model->points_[i];
            model->colors_[k] = model->colors_[i];
            model->frameCounter_[k] = model->frameCounter_[i];
            model->hitCounter_[k] = model->hitCounter_[i];
            model->confidence_[k] = model->confidence_[i];
            model->radius_[k] = model->radius_[i];
            k++;
        }
    }
    model->points_.resize(k);
    model->colors_.resize(k);
    model->frameCounter_.resize(k);
    model->hitCounter_.resize(k);
    model->confidence_.resize(k);
    model->radius_.resize(k);
    cerr << "Removed points " << totalPoints - k << endl;
}

shared_ptr<PointCloudExtended> VoxelDownSampleExt(shared_ptr<PointCloudExtended> &input,
                                                  double voxel_size){
    //    PointCloud pc = *input;
    shared_ptr<PointCloudExtended> pointCloud = std::make_shared<PointCloudExtended>();
    shared_ptr<PointCloud> pcd = std::make_shared<PointCloud>();
    pcd = std::get<0>(geometry::RemoveStatisticalOutliers(*input, 10, 0.1));
    pcd = VoxelDownSample(*input, voxel_size);
    pointCloud->reserveAndAssign(pcd->points_.size());
    for (int i = 0; i < pcd->points_.size(); ++i) {
        pointCloud->points_.push_back(pcd->points_[i]);
        pointCloud->colors_.push_back(pcd->colors_[i]);
        pointCloud->confidence_[i] = 0;
        pointCloud->frameCounter_[i] = 0;
        pointCloud->hitCounter_[i] = confThresh;
    }

    return pointCloud;
}

double merge(shared_ptr<PointCloudExtended> model, shared_ptr<PointCloud> lastFrame,
             Eigen::Matrix4d prevTransf,
             Eigen::Matrix4d transf, PinholeCameraIntrinsic intrinsics,
             double depthScale, double angle,
             Mat normalWeight = Mat(), double radius = 0.5){

    //At first model comes empty then will be fullfilled with first frame's points
    if(model->IsEmpty()){
        for (int i = 0; i < lastFrame->points_.size(); ++i) {
            model->points_.push_back(lastFrame->points_[i]);
            model->colors_.push_back(lastFrame->colors_[i]);
        }
        model->reserveAndAssign(lastFrame->points_.size());
        return 0;
    }

    //Point removal
    //removeUnstablePoints(model);

    Mat depth1, depth2;
    Mat modelIdx, lastFrameIdx;

    projectPointCloud(*model, 2, depthScale, prevTransf.inverse(), intrinsics, depth1,
                      modelIdx, radius);
    projectPointCloud(*lastFrame, 2, depthScale, transf.inverse(), intrinsics, depth2,
                      lastFrameIdx, radius);

    //    imshow("model", depth1*10);
    //    imshow("lastframe", depth2*10);
    //    waitKey(0);

    int addNew = 0;
    int addInFrontOf = 0;
    int fused = 0;

    for (int r = 0; r < modelIdx.rows; ++r) {
        for (int c = 0; c < modelIdx.cols; ++c) {
            int modIdx = *modelIdx.ptr<int>(r, c);
            int lastIdx = *lastFrameIdx.ptr<int>(r, c);

            double pixelNormal = 1;
            if(normalWeight.rows != 0)
                pixelNormal = *normalWeight.ptr<double>(r, c);

            if(pixelNormal <= 0)
                continue;

            ushort depth = *depth2.ptr<ushort>(r, c);
            double newRadius = depth;
            //if(modIdx >= 0 && model->confidence_[modIdx] > 10){
            //    continue;
            //}
            //Two points are into a line of sight from camera
            if(lastIdx >= 0 && modIdx >= 0){
                Eigen::Vector3d modelPoint = model->points_[modIdx];
                Eigen::Vector3d modelColor = model->colors_[modIdx];
                Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];
                //Two points close enough to get fused
                double diff = abs(modelPoint(2) - framePoint(2));
                if(diff < 0.02){
                    model->hitCounter_[modIdx]++;
                    double n = model->confidence_[modIdx];
                    model->radius_[modIdx] = newRadius;
                    model->points_[modIdx](2) = (modelPoint(2) * n + framePoint(2) * pixelNormal)/
                            (n + pixelNormal);
                    model->colors_[modIdx] = (modelColor * n + frameColor * pixelNormal)/
                            (n + pixelNormal);
                    model->confidence_[modIdx] += pixelNormal;
                    fused++;
                }
                //Far enough to be another point in front of this point
                else if (modelPoint(2) - framePoint(2) >= 0.03){
                    Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                    Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];
                    model->points_.push_back(framePoint);
                    model->colors_.push_back(frameColor);
                    model->hitCounter_.push_back(0);
                    model->frameCounter_.push_back(0);
                    model->confidence_.push_back(0);
                    model->radius_.push_back(3);
                    addInFrontOf++;
                }
            }
            //A new point
            else if(lastIdx >= 0 && modIdx == -1){
                Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];
                //Eigen::Vector3d frameColor(1,0,0);
                model->points_.push_back(framePoint);
                model->colors_.push_back(frameColor);
                model->hitCounter_.push_back(0);
                model->frameCounter_.push_back(0);
                model->confidence_.push_back(0);
                model->radius_.push_back(3);
                addNew++;
            }
        }
    }
    cerr << "Fused points: " << fused << endl;
    cerr << "New points: " << addNew << endl;
    cerr << "New points in front of " << addInFrontOf << endl;
    //    waitKey(100);
    //    cerr << "Total points " << model->points_.size() << endl;
    return (addInFrontOf + 1.0)/(model->points_.size()+1.0);
}

void removeMaskBorders(Mat &mask, int sizeX, int sizeY){
    //    cerr << "TIPO: " << mask->type() << endl;
    int xlenght = mask.cols;
    int ylenght = mask.rows;
    for (int y = 0; y < ylenght; ++y) {
        for (int x = 0; x < xlenght; ++x) {
            if(x < sizeX || y < sizeY || y > ylenght - sizeY || x > xlenght - sizeX){
                *mask.ptr<Vec3b>(y, x) = 0;
            }
        }
    }
}

Mat applyMaskDepth(Mat depthMap, Mat maskMap){
    Mat depthClean;
    depthClean = Mat::zeros(depthMap.rows, depthMap.cols, CV_16UC1);
    for (int r = 0; r < depthMap.rows; ++r) {
        for (int c = 0; c < depthMap.cols; ++c) {
            ushort mask = *maskMap.ptr<ushort>(r, c);
            if(mask > 0){
                double d = *depthMap.ptr<ushort>(r, c);
                *depthClean.ptr<ushort>(r, c) = d;
            }
            else{
                *depthClean.ptr<ushort>(r, c) = 0;
            }
        }
    }
    return depthClean;
}

void weighting(MatrixXd &residuals) {
    int n = residuals.rows();
    for(int i = 0; i < n; i++){
        double ad = residuals(i, 0);//sigma;        
        residuals(i, 0) *= ad > 0.01 ? 0 : 1;
    }
}


Eigen::Matrix4d TransformVector6dToMatrix4d(Eigen::VectorXd &input) {

    double x = input(0);
    double y = input(1);
    double z = input(2);
    double yaw = input(3);
    double pitch = input(4);
    double roll = input(5);

    Matrix4d lie;
    lie << 0.f,  -roll, pitch,    x,
            roll,   0.f,  -yaw,    y,
            -pitch,   yaw,    0.f,   z,
            0.f,   0.f,    0.f, 0.f;

    lie = lie.exp();
    return lie;
    /*
    Eigen::Matrix4d output;
    output.setIdentity();
    output.block<3, 3>(0, 0) =
            (Eigen::AngleAxisd(input(5), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(input(4), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(input(3), Eigen::Vector3d::UnitX()))
                    .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(0, 0);
    //output = output.exp();
    return output;
    */
}

Eigen::VectorXd TransformMatrix4dToVector6d(Eigen::Matrix4d &input) {
    Eigen::VectorXd output;
    output.setZero(6);
    Eigen::Matrix3d R = input.block<3, 3>(0, 0);
    double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    if (!(sy < 1e-6)) {
        output(3) = atan2(R(2, 1), R(2, 2));
        output(4) = atan2(-R(2, 0), sy);
        output(5) = atan2(R(1, 0), R(0, 0));
    } else {
        output(3) = atan2(-R(1, 2), R(1, 1));
        output(4) = atan2(-R(2, 0), sy);
        output(5) = 0;
    }
    output.block<3, 1>(0, 0) = input.block<3, 1>(0, 3);
    return output;
}

double interpolateIntensityWithDepth(Mat &intensity, Mat &depth, double &x, double &y, double &z)
{
    const int x0 = (int) floor(x);
    const int y0 = (int) floor(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    if(x1 >= intensity.cols || y1 >= intensity.rows){
        return 0;
    }

    const double x1_weight = x - x0;
    const double x0_weight = 1.0f - x1_weight;
    const double y1_weight = y - y0;
    const double y0_weight = 1.0f - y1_weight;
    const double z_eps = z - 0.05f;

    double val = 0.0f;
    double sum = 0.0f;

    if(isfinite(depth.at<double>(y0, x0)) && depth.at<double>(y0, x0) > z_eps){
        val += x0_weight * y0_weight * (double)(intensity.at<uchar>(y0, x0))/255.;
        sum += x0_weight * y0_weight;
    }

    if(isfinite(depth.at<double>(y0, x1)) && depth.at<double>(y0, x1) > z_eps){
        val += x1_weight * y0_weight * (double)(intensity.at<uchar>(y0, x1))/255.;
        sum += x1_weight * y0_weight;
    }

    if(isfinite(depth.at<double>(y1, x0)) && depth.at<double>(y1, x0) > z_eps){
        val += x0_weight * y1_weight * (double)(intensity.at<uchar>(y1, x0))/255.;
        sum += x0_weight * y1_weight;
    }

    if(isfinite(depth.at<double>(y1, x1)) && depth.at<double>(y1, x1) > z_eps){
        val += x1_weight * y1_weight * (double)(intensity.at<uchar>(y1, x1))/255.;
        sum += x1_weight * y1_weight;
    }

    if(sum > 0.0f){
        val /= sum;
    }
    else{
        val = 0;
    }

    return val;
}

double interpolateDepth(Mat &depth, double &x, double &y, double &z)
{
    const int x0 = (int) floor(x);
    const int y0 = (int) floor(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    if(x1 >= depth.cols || y1 >= depth.rows){
        return 0;
    }

    const double x1_weight = x - x0;
    const double x0_weight = 1.0f - x1_weight;
    const double y1_weight = y - y0;
    const double y0_weight = 1.0f - y1_weight;
    const double z_eps = z - 0.05f;

    double val = 0.0f;
    double sum = 0.0f;

    if(isfinite(depth.at<double>(y0, x0)) && depth.at<double>(y0, x0) > z_eps){
        val += x0_weight * y0_weight * depth.at<double>(y0, x0);
        sum += x0_weight * y0_weight;
    }

    if(isfinite(depth.at<double>(y0, x1)) && depth.at<double>(y0, x1) > z_eps){
        val += x1_weight * y0_weight * depth.at<double>(y0, x1);
        sum += x1_weight * y0_weight;
    }

    if(isfinite(depth.at<double>(y1, x0)) && depth.at<double>(y1, x0) > z_eps){
        val += x0_weight * y1_weight * depth.at<double>(y1, x0);
        sum += x0_weight * y1_weight;
    }

    if(isfinite(depth.at<double>(y1, x1)) && depth.at<double>(y1, x1) > z_eps){
        val += x1_weight * y1_weight * depth.at<double>(y1, x1);
        sum += x1_weight * y1_weight;
    }

    if(sum > 0.0f){
        val /= sum;
    }
    else{
        val = 0;
    }

    return val;
}
#endif
