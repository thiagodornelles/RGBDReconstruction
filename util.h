#ifndef UTIL_H
#define UTIL_H

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

using namespace open3d;
using namespace open3d::io;
using namespace open3d::camera;
using namespace open3d::utility;
using namespace cv;
using namespace std;
using namespace rgbd;

const int confThresh = 3;
double angleInsert = 0;

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

Mat getNormalMap(Mat &depth, double depthScale, bool dilateBorders = false, double angleThres = 1.222){
    Mat normals = Mat::zeros(depth.rows, depth.cols, CV_32FC3);
    Mat temp;
    depth.convertTo(temp, CV_32FC1, 1.0/depthScale);
    Mat filteredDepth;
//    bilateralFilter(temp, filteredDepth, 3, 1, 1);
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


Mat getNormalMapFromPointCloud(shared_ptr<PointCloud> pointCloud, PinholeCameraIntrinsic intrinsics,
                               bool dilateBorders = false, double angleThres = 1.222){
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
    double nearbNormals[] = { 7, 7, 3};

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
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;
    double width = intrinsics.width_;
    double height = intrinsics.height_;    
//    double fovX = 2 * atan(width / (2 * fx)) * 180.0 / CV_PI;
//    double fovY = 2 * atan(height / (2 * fy)) * 180.0 / CV_PI;
//    float aspectRatio = depth.cols / (float)depth.rows; // assuming width > height

    normalMap.forEach<Vec3d>([&](Vec3d &pixel, const int *pos) -> void
    {
        if(depth.at<double>(pos[0], pos[1]) > 0 ){
//            float Px = (2 * ((pos[1] + 0.5) / depth.cols) - 1) *
//                    tan(fovX / 2 * M_PI / 180) * aspectRatio;
//            float Py = (1 - 2 * ((pos[0] + 0.5) / depth.rows)) *
//                    tan(fovY / 2 * M_PI / 180);
            Vec3d camAxis(0, 0, -1);
            camAxis = normalize(camAxis);
            pixel = normalize(pixel);
            double uv = pixel.dot(camAxis);
            uv = cos(acos(uv) * angleThres);
            filtered.at<double>(pos[0], pos[1]) = uv < angleCut ? 0 : uv;
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
                circle(indexMap, Point(transfC_int, transfR_int), radius, i, -1);
                circle(depthMap, Point(transfC_int, transfR_int), radius, value, -1);
            }
        }
    }
    depthMap.setTo(0, depthMap == 65535);
    medianBlur(depthMap, depthMap, 3);
//    erode(depthMap, depthMap, getStructuringElement(MORPH_CROSS, Size(3, 3)));
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
                circle(indexMap, Point(transfC_int, transfR_int), radius, i, -1);
                circle(depthMap, Point(transfC_int, transfR_int), radius, value, -1);
            }
        }
    }
    depthMap.setTo(0, depthMap == 65535);
    medianBlur(depthMap, depthMap, 3);
//    erode(depthMap, depthMap, getStructuringElement(MORPH_CROSS, Size(3, 3)));
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
    int removed = 0;
    for (int i = 0; i < model->points_.size(); ++i) {
        if(model->hitCounter_[i] < confThresh){
            model->points_.erase(model->points_.begin() + i);
            model->colors_.erase(model->colors_.begin() + i);
            model->frameCounter_.erase(model->frameCounter_.begin() + i);
            model->hitCounter_.erase(model->hitCounter_.begin() + i);
            model->angle_.erase(model->angle_.begin() + i);
            removed++;
        }
    }
    cerr << removed << " points removed\n";
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
        pointCloud->angle_[i] = 0;
        pointCloud->frameCounter_[i] = 0;
        pointCloud->hitCounter_[i] = confThresh;
    }

    return pointCloud;
}

double merge(shared_ptr<PointCloudExtended> model, shared_ptr<PointCloud> lastFrame,
           Eigen::Matrix4d prevTransf,
           Eigen::Matrix4d transf, PinholeCameraIntrinsic intrinsics,
           double depthScale, double angle, Mat normalMap,
           Mat normalWeight = Mat(), double radius = 1){

    //At first model comes empty then will be fullfilled with first frame's points
    if(model->IsEmpty()){
        for (int i = 0; i < lastFrame->points_.size(); ++i) {
            model->points_.push_back(lastFrame->points_[i]);
            model->colors_.push_back(lastFrame->colors_[i]);
        }
        model->reserveAndAssign(lastFrame->points_.size());
        return 0;
    }

    int removed = 0;
    //Aging of points
    for (int i = 0; i < model->points_.size(); ++i) {
        model->frameCounter_[i]++;
        if(model->frameCounter_[i] > confThresh && model->hitCounter_[i] < confThresh){
            model->points_.erase(model->points_.begin() + i);
            model->colors_.erase(model->colors_.begin() + i);
            model->frameCounter_.erase(model->frameCounter_.begin() + i);
            model->hitCounter_.erase(model->hitCounter_.begin() + i);
            model->angle_.erase(model->angle_.begin() + i);
            model->radius_.erase(model->radius_.begin() + i);
            removed++;
        }
    }

    cerr << removed << " points removed\n";

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

            double normal = 1;
            if(normalWeight.rows != 0)
                normal = *normalWeight.ptr<double>(r, c);

            if(normal <= 0)
                continue;

            double newRadius = 1;
//            if(lastIdx >= 0){
//                double zfn = -lastFrame->points_[lastIdx](2)*depthScale;
//                zfn /= normal;
//                newRadius = 1/sqrt(2) * zfn;
//                cerr << "new radius " << newRadius << endl;
//            }

            //Two points are into a line of sight from camera
            if(lastIdx >= 0 && modIdx >= 0){

                Eigen::Vector3d modelPoint = model->points_[modIdx];
                Eigen::Vector3d modelColor = model->colors_[modIdx];
                Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];
                //Two points close enough to get fused
                double diff = abs(modelPoint(2) - framePoint(2));
                double angleDiff = abs(model->angle_[modIdx] - angle);

                if(diff < 0.005){
                    model->hitCounter_[modIdx]++;
                    int n = model->hitCounter_[modIdx];// > 10 ? 10 : model->hitCounter_[modIdx];
                    model->points_[modIdx] = (modelPoint * (n-normal)/n + framePoint * normal/n);
                    model->colors_[modIdx] = (modelColor * (n-normal)/n + frameColor * normal/n);
                    model->radius_[modIdx] = newRadius;
                    fused++;
                }
                //Far enough to be another point in front of this point
                else if (modelPoint(2) - framePoint(2) >= 0.05){
                    Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                    Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];
                    model->points_.push_back(framePoint);
                    model->colors_.push_back(frameColor);
                    model->hitCounter_.push_back(0);
                    model->frameCounter_.push_back(0);
                    model->angle_.push_back(angle);
                    model->radius_.push_back(newRadius);
                    addInFrontOf++;
                }
            }
            //A new point
            else if(lastIdx >= 0 && modIdx == -1){
                Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];
                model->points_.push_back(framePoint);
                model->colors_.push_back(frameColor);
                model->hitCounter_.push_back(0);
                model->frameCounter_.push_back(0);
                model->angle_.push_back(angle);
                model->radius_.push_back(newRadius);
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
#endif
