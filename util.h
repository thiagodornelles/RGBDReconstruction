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

Mat convertDepthTo16bit(std::shared_ptr<Image> depth){
    Mat depth16;
    Mat depth32(depth->height_, depth->width_, CV_32FC1, depth->data_.data());
    depth32.convertTo(depth16, CV_16UC1, 5000);
    return depth16;
}

// Level comes in reverse order 2 > 1 > 0
Mat getMaskOfNormalsFromDepth(Mat &depth, PinholeCameraIntrinsic intrinsics, int level,
                              bool dilateBorders = false, double angleFilter = 1.222){

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
    depth.convertTo(tempDepth, CV_16UC1, 5000);
    depthTo3d(tempDepth, K, refPoints3d);
    normalComp(refPoints3d, normals);

    Mat filtered = Mat::zeros(normals.rows, normals.cols, CV_64FC1);
    Vec3d camAxis (0, 0, -1);
    normals.forEach<Vec3d>(
                [&filtered, &camAxis, &angleFilter, &level](Vec3d &pixel, const int *pos) -> void
    {
        double uv = isnan(pixel[0]) ? 0 : pixel.dot(camAxis);
        uv = cos(acos(uv) * angleFilter);
        filtered.at<double>(pos[0], pos[1]) = uv < 0 ? 0 : uv;
    }
    );

    if (dilateBorders){
        Mat element = getStructuringElement( MORPH_CROSS,
                                             Size( 2*3 + 1, 2*3+1 ),
                                             Point( 3, 3 ) );
        erode( filtered, filtered, element );
    }
    return filtered;
}

void projectPointCloud(PointCloud pointCloud, double maxDist,
                       Eigen::Matrix4d Rt, PinholeCameraIntrinsic intrinsics,
                       Mat &depthMap, Mat &indexMap){

    int width = intrinsics.width_;
    int height = intrinsics.height_;

    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    depthMap.create(height, width, CV_16UC1);
    depthMap.setTo(65535);
    indexMap.create(height, width, CV_64FC1);
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
        if((transfR_int >= 0 && transfR_int < height) &&
                (transfC_int >= 0 && transfC_int < width)) {

            ushort lastZ = *depthMap.ptr<ushort>(transfR_int, transfC_int);
            ushort value = refPoint3d(2) * 5000;            

            if(value < lastZ){
                *depthMap.ptr<ushort>(transfR_int, transfC_int) = value;
                *indexMap.ptr<double>(transfR_int, transfC_int) = i;
            }
        }
    }
    depthMap.setTo(0, depthMap == 65535);
    medianBlur(depthMap, depthMap, 7);
}

void combineDepthsFilter(Mat &originalDepth, Mat &projectedDepth){
    projectedDepth.forEach<ushort>([&](ushort &pixel, const int *pos) -> void
    {
        ushort p = *originalDepth.ptr<ushort>(pos[0], pos[1]);
        if(pixel == 0 &&  p > 0){
            *projectedDepth.ptr<ushort>(pos[0], pos[1]) = p;
        }
    }
    );
    medianBlur(projectedDepth, projectedDepth, 5);
}

void projectMesh(shared_ptr<TriangleMesh> mesh, double maxDist,
                       Eigen::Matrix4d Rt, PinholeCameraIntrinsic intrinsics,
                       Mat &depthMap, Mat &indexMap){

    int width = intrinsics.width_;
    int height = intrinsics.height_;

    double cx = intrinsics.GetPrincipalPoint().first;
    double cy = intrinsics.GetPrincipalPoint().second;
    double fx = intrinsics.GetFocalLength().first;
    double fy = intrinsics.GetFocalLength().second;

    depthMap.create(height, width, CV_16UC1);
    depthMap.setTo(0);
    indexMap.create(height, width, CV_64FC1);
    indexMap.setTo(-1);

    for (int i = 0; i < mesh->triangles_.size(); i++) {

        Eigen::Vector3i triangle = mesh->triangles_[i];
        Eigen::Vector4d refPoint3d;
        refPoint3d(0) = triangle(0);
        refPoint3d(1) = triangle(1);
        refPoint3d(2) = triangle(2);
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
        if((transfR_int >= 0 && transfR_int < height) &&
                (transfC_int >= 0 && transfC_int < width) &&
                refPoint3d(2) > 0.1 && refPoint3d(2) < maxDist) {

            double lastZ = *depthMap.ptr<ushort>(transfR_int, transfC_int)/5000.f;

            if(refPoint3d(2) > lastZ){
                ushort value = refPoint3d(2) * 5000;
                *depthMap.ptr<ushort>(transfR_int, transfC_int) = value;
                *indexMap.ptr<double>(transfR_int, transfC_int) = i;
            }
        }
    }
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

void merge(shared_ptr<PointCloud> model, shared_ptr<PointCloud> lastFrame, PinholeCameraIntrinsic intrinsics){

    if(model->IsEmpty()){
        *model = *model + *lastFrame;
        return;
    }

    Eigen::Matrix4d transf = Eigen::Matrix4d::Identity();
//    transf <<  1,  0,  0,  0,
//            0, -1,  0,  0,
//            0,  0, -1,  0,
//            0,  0,  0,  1;

    Mat depth1, depth2;
    Mat modelIdx, lastFrameIdx;
    projectPointCloud(*model, 5, transf, intrinsics, depth1, modelIdx);
    projectPointCloud(*lastFrame, 5, transf, intrinsics, depth2, lastFrameIdx);

    int added1 = 0;
    int fused = 0;
    int added2 = 0;

    for (int r = 0; r < modelIdx.rows; ++r) {
        for (int c = 0; c < modelIdx.cols; ++c) {
            double modIdx = *modelIdx.ptr<double>(r, c);
            double lastIdx = *lastFrameIdx.ptr<double>(r, c);
            if(lastIdx >= 0 && modIdx >= 0){
                Eigen::Vector3d modelPoint = model->points_[modIdx];
                Eigen::Vector3d modelColor = model->colors_[modIdx];
                Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];

                //Two points close enough to get fused
                if(abs(modelPoint(2) - framePoint(2)) < 0.02){
                    model->points_[modIdx] = modelPoint * 0.9 + framePoint * 0.1;
                    model->colors_[modIdx] = modelColor * 0.9 + frameColor * 0.1;
                    fused++;
                }
                else if(framePoint(2) < modelPoint(2)){
                    model->points_.push_back(framePoint);
                    model->colors_.push_back(frameColor);
                    added1++;
                }
            }
            else if(lastIdx >= 0 && modIdx == -1){
                Eigen::Vector3d framePoint = lastFrame->points_[lastIdx];
                Eigen::Vector3d frameColor = lastFrame->colors_[lastIdx];
                model->points_.push_back(framePoint);
                model->colors_.push_back(frameColor);
                added2++;
            }
        }
    }
    cerr << "Fused points " << fused << endl;
    cerr << "Added1 points " << added1 << endl;
    cerr << "Added2 points " << added2 << endl;
    cerr << "Total points " << model->points_.size() << endl;

}
#endif
