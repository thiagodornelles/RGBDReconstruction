#ifndef ALIGNER_H
#define ALIGNER_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui.hpp>
#include <Open3D/Camera/PinholeCameraIntrinsic.h>

#include "util.h"

using namespace std;
using namespace cv;
using namespace rgbd;
using namespace Eigen;
using namespace open3d;

class Aligner{

public:

    VectorXd actualPoseVector6D;
    VectorXd bestPoseVector6D;

    PinholeCameraIntrinsic intrinsics;

    Mat sourceIntensityImage;
    Mat sourceDepthImage;
    Mat targetIntensityImage;
    Mat targetDepthImage;

    MatrixXd jacobianRt = MatrixXd(3,6);
    MatrixXd jacobianRt_z = MatrixXd(1,6);
    MatrixXd jacobianProj = MatrixXd(2,3);
    MatrixXd jacobianIntensity = MatrixXd(1,6);
    MatrixXd jacobianDepth = MatrixXd(1,6);
    MatrixXd gradPixIntensity = MatrixXd(1,2);
    MatrixXd gradPixDepth = MatrixXd(1,2);

    double lastGradientNorm = DBL_MAX;
    double sum = DBL_MAX;
    double minDist = 0.1;
    double maxDist = 3.0;
    double depthScale = 5000.0;

    Aligner(PinholeCameraIntrinsic intrinsecs){
        actualPoseVector6D.setZero(6);
        bestPoseVector6D.setZero(6);
        this->intrinsics = intrinsecs;
    }

    VectorXd getPose6D(){
        return bestPoseVector6D;
    }

    void setDist(double minDist, double maxDist){
        this->minDist = minDist;
        this->maxDist = maxDist;
    }

    void setsourceRGBImages(Mat sourceIntensityImage, Mat sourceDepthImage){
        this->sourceIntensityImage = sourceIntensityImage;
        this->sourceDepthImage = sourceDepthImage;
    }

    void setsourceDepthImages(Mat targetIntensityImage, Mat targetDepthImage){
        this->targetIntensityImage = targetIntensityImage;
        this->targetDepthImage = targetDepthImage;
    }

    void setInitialPoseVector(VectorXd poseVector6D){
        this->actualPoseVector6D = poseVector6D;
    }

    void calculateDerivativeX(const cv::Mat& img, cv::Mat& result){
        result.create(img.size(), CV_64FC1);
        for(int y = 0; y < img.rows; ++y){
            for(int x = 0; x < img.cols; ++x){
                int prev = std::max(x - 1, 0);
                int next = std::min(x + 1, img.cols - 1);

                result.at<double>(y, x) = (double)(img.at<double>(y, next) - img.at<double>(y, prev)) * 0.5f;
            }
        }
    }

    void calculateDerivativeY(const cv::Mat& img, cv::Mat& result){
        result.create(img.size(), CV_64FC1);
        for(int y = 0; y < img.rows; ++y){
            for(int x = 0; x < img.cols; ++x){
                int prev = std::max(y - 1, 0);
                int next = std::min(y + 1, img.rows - 1);
                result.at<double>(y, x) = (double)(img.at<double>(next, x) - img.at<double>(prev, x)) * 0.5f;
            }
        }
    }

    double computeResidualsAndJacobians(Mat &refIntImage, Mat &refDepImage,
                                        Mat &actIntImage, Mat &actDepImage,
                                        Mat &weight, Mat &normalMap,
                                        MatrixXd &residuals, MatrixXd &jacobians, double depthWeight,
                                        int level, bool color = true, bool depth = true)
    {
        double scaleFactor = 1.0 / pow(2, level);
        //level comes in reverse order 2 > 1 > 0
        double intScales[] = { 0.0001, 0.0001, 0.0001, 0.0001 };
        //        double depScales[] = { 0.0001, 0.0001, 0.0001, 0.0001 };
        //        double intScales[] = { 0.001, 0.001, 0.001 };
        //        double depScales[] = { 0.001, 0.001, 0.001 };

        Mat residualImage = Mat::zeros(refIntImage.rows * 2, refIntImage.cols, CV_64FC1);
        Mat actIntDerivX = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);
        Mat actIntDerivY = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);
        //        Mat actDepDerivX = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);
        //        Mat actDepDerivY = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);

        Scharr(actIntImage, actIntDerivX, CV_64F, 1, 0, intScales[level], 0.0, cv::BORDER_DEFAULT);
        Scharr(actIntImage, actIntDerivY, CV_64F, 0, 1, intScales[level], 0.0, cv::BORDER_DEFAULT);
        //        Sobel(actIntImage, actIntDerivX, CV_64F, 1, 0, 3, intScales[level], 0.0, cv::BORDER_DEFAULT);
        //        Sobel(actIntImage, actIntDerivY, CV_64F, 0, 1, 3, intScales[level], 0.0, cv::BORDER_DEFAULT);
        //        imshow("derX", actIntDerivX);
        //        imshow("derY", actIntDerivY);
        //        Scharr(actDepImage, actDepDerivX, CV_64F, 1, 0, depScales[level], 0.0, cv::BORDER_DEFAULT);
        //        Scharr(actDepImage, actDepDerivY, CV_64F, 0, 1, depScales[level], 0.0, cv::BORDER_DEFAULT);
        //        Sobel(actDepImage, actDepDerivX, CV_64F, 1, 0, 3, depScales[level], 0.0, cv::BORDER_DEFAULT);
        //        Sobel(actDepImage, actDepDerivY, CV_64F, 0, 1, 3, depScales[level], 0.0, cv::BORDER_DEFAULT);

        //        imshow("wDep", normalsWeight);

        //Extrinsecs
        //        double x = actualPoseVector6D(0);
        //        double y = actualPoseVector6D(1);
        //        double z = actualPoseVector6D(2);
        //        double yaw = actualPoseVector6D(3);
        //        double pitch = actualPoseVector6D(4);
        //        double roll = actualPoseVector6D(5);

        //Intrinsecs
        double cx = scaleFactor * intrinsics.GetPrincipalPoint().first;
        double cy = scaleFactor * intrinsics.GetPrincipalPoint().second;
        double fx = scaleFactor * intrinsics.GetFocalLength().first;
        double fy = scaleFactor * intrinsics.GetFocalLength().second;

        //Matrix |R|t| from 6D vector
        //        Matrix4d Rt = getMatrixRtFromPose6D(actualPoseVector6D);
        Matrix4d Rt = TransformVector6dToMatrix4d(actualPoseVector6D);
        //        cerr << Rt << endl;

        //Calculation of jacobians and residuals
        int nCols = refIntImage.cols;
        int nRows = refIntImage.rows;
        double thisSum = 0;
        const int nthreads = 4;
        double count[nthreads] = {0};

        cv::setNumThreads(nthreads);
        parallel_for_(Range(0, nCols*nRows), [&](const Range& range)
        {
            MatrixXd jacobianRt = MatrixXd(3,6);
            //            MatrixXd jacobianRt_z = MatrixXd(1,6);
            MatrixXd jacobianProj = MatrixXd(2,3);
            MatrixXd jacobianIntensity = MatrixXd(1,6);
            MatrixXd jacobianDepth = MatrixXd(1,6);
            MatrixXd gradPixIntensity = MatrixXd(1,2);
            //            MatrixXd gradPixDepth = MatrixXd(1,2);

            int nThread = range.start/range.size();
            for (int r = range.start; r < range.end; r++) {

                int y = r / refDepImage.cols;
                int x = r % refDepImage.cols;

                if(*refDepImage.ptr<double>(y, x) < minDist)
                    continue;

                gradPixIntensity(0,0) = *actIntDerivX.ptr<double>(y, x);
                gradPixIntensity(0,1) = *actIntDerivY.ptr<double>(y, x);
                //                gradPixDepth(0,0) = *actDepDerivX.ptr<double>(y, x);
                //                gradPixDepth(0,1) = *actDepDerivY.ptr<double>(y, x);

                //                double wInt = gradPixIntensity(0,0) * gradPixIntensity(0,0);
                //                wInt += gradPixIntensity(0,1) * gradPixIntensity(0,1);
                //                wInt = sqrt(wInt) > 0.02 ? 1 : 0;
                //                cerr << wInt << endl;
                double wInt = 1;

                //******* BEGIN Unprojection of DepthMap ********
                Vector4d refPoint3D;
                refPoint3D(2) = *refDepImage.ptr<double>(y, x);
                refPoint3D(0) = (x - cx) * refPoint3D(2) * 1/fx;
                refPoint3D(1) = (y - cy) * refPoint3D(2) * 1/fy;
                refPoint3D(3) = 1;

                Vector4d actPoint3D;
                actPoint3D(2) = *actDepImage.ptr<double>(y, x);
                actPoint3D(0) = (x - cx) * refPoint3D(2) * 1/fx;
                actPoint3D(1) = (y - cy) * refPoint3D(2) * 1/fy;
                actPoint3D(3) = 1;
                //******* END Unprojection of DepthMap ********

                //******* BEGIN Transformation of PointCloud ********
                Vector4d trfPoint3D = Rt * refPoint3D;
                double invTransfZ = 1.0 / trfPoint3D(2);
                //******* END Transformation of PointCloud ********

                double px = refPoint3D(0);
                double py = refPoint3D(1);
                double pz = refPoint3D(2);

                //Derivative w.r.t x
                jacobianRt(0,0) = 1.;
                jacobianRt(1,0) = 0.;
                jacobianRt(2,0) = 0.;

                //Derivative w.r.t y
                jacobianRt(0,1) = 0.;
                jacobianRt(1,1) = 1.;
                jacobianRt(2,1) = 0.;

                //Derivative w.r.t z
                jacobianRt(0,2) = 0.;
                jacobianRt(1,2) = 0.;
                jacobianRt(2,2) = 1.;

                //Derivative w.r.t yaw
                jacobianRt(0,3) = 0;
                jacobianRt(1,3) = -pz;
                jacobianRt(2,3) = py;

                //Derivative w.r.t pitch
                jacobianRt(0,4) = pz;
                jacobianRt(1,4) = 0;
                jacobianRt(2,4) = -px;

                //Derivative w.r.t roll
                jacobianRt(0,5) = -py;
                jacobianRt(1,5) = px;
                jacobianRt(2,5) = 0;

                //                //Derivative w.r.t yaw
                //                jacobianRt(0,3) = py*(-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw))+pz*(sin(roll)*cos(yaw)-sin(pitch)*cos(roll)*sin(yaw))-cos(pitch)*px*sin(yaw);
                //                jacobianRt(1,3) = pz*(sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw))+py*(sin(pitch)*sin(roll)*cos(yaw)-cos(roll)*sin(yaw))+cos(pitch)*px*cos(yaw);
                //                jacobianRt(2,3) = 0.;

                //                //Derivative w.r.t pitch
                //                jacobianRt(0,4) = cos(pitch)*py*sin(roll)*cos(yaw)+cos(pitch)*pz*cos(roll)*cos(yaw)-sin(pitch)*px*cos(yaw);
                //                jacobianRt(1,4) = cos(pitch)*py*sin(roll)*sin(yaw)+cos(pitch)*pz*cos(roll)*sin(yaw)-sin(pitch)*px*sin(yaw);
                //                jacobianRt(2,4) = -sin(pitch)*py*sin(roll)-sin(pitch)*pz*cos(roll)-cos(pitch)*px;

                //                //Derivative w.r.t roll
                //                jacobianRt(0,5) = py*(sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw))+pz*(cos(roll)*sin(yaw)-sin(pitch)*sin(roll)*cos(yaw));
                //                jacobianRt(1,5) = pz*(-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw))+py*(sin(pitch)*cos(roll)*sin(yaw)-sin(roll)*cos(yaw));
                //                jacobianRt(2,5) = cos(pitch)*py*cos(roll)-cos(pitch)*pz*sin(roll);

                //Derivative w.r.t x
                jacobianProj(0,0) = fx*invTransfZ;
                jacobianProj(1,0) = 0.;

                //Derivative w.r.t y
                jacobianProj(0,1) = 0.;
                jacobianProj(1,1) = fy*invTransfZ;

                //Derivative w.r.t z
                jacobianProj(0,2) = -(fx*trfPoint3D(0))*invTransfZ*invTransfZ;
                jacobianProj(1,2) = -(fy*trfPoint3D(1))*invTransfZ*invTransfZ;

                //                jacobianRt_z(0,0) = jacobianRt(2,0);
                //                jacobianRt_z(0,1) = jacobianRt(2,1);
                //                jacobianRt_z(0,2) = jacobianRt(2,2);
                //                jacobianRt_z(0,3) = jacobianRt(2,3);
                //                jacobianRt_z(0,4) = jacobianRt(2,4);
                //                jacobianRt_z(0,5) = jacobianRt(2,5);

                double nx = (*normalMap.ptr<Vec3d>(y, x))[0];
                double ny = (*normalMap.ptr<Vec3d>(y, x))[1];
                double nz = (*normalMap.ptr<Vec3d>(y, x))[2];
//                double w1 = -ny*pz + nz*py;
//                double w2 =  nx*pz - nz*px;
//                double w3 = -nx*py + ny*px;

                Eigen::Vector3d vt(trfPoint3D(0), trfPoint3D(1), trfPoint3D(2));
                Eigen::Vector3d vs(actPoint3D(0), actPoint3D(1), actPoint3D(2));
                Eigen::Vector3d nt(nx, ny, nz);

                Eigen::Vector3d cros = vs.cross(nt);

                jacobianDepth(0) = nt(0);
                jacobianDepth(1) = nt(1);
                jacobianDepth(2) = nt(2);
                jacobianDepth(3) = cros(0);
                jacobianDepth(4) = cros(1);
                jacobianDepth(5) = cros(2);

                jacobianIntensity = gradPixIntensity * jacobianProj * jacobianRt;
                //jacobianDepth = gradPixDepth * jacobianProj * jacobianRt - jacobianRt_z;

                //******* BEGIN Projection of PointCloud on the image plane ********
                double transfC = (trfPoint3D(0) * fx) * invTransfZ + cx;
                double transfR = (trfPoint3D(1) * fy) * invTransfZ + cy;
                int transfR_int = static_cast<int>(round(transfR));
                int transfC_int = static_cast<int>(round(transfC));
                //******* END Projection of PointCloud on the image plane ********
                //Checks if this pixel projects inside of the source image
                if((transfR_int >= 0 && transfR_int < nRows) &&
                        (transfC_int >= 0 && transfC_int < nCols)
                        && refPoint3D(2) > minDist && refPoint3D(2) < maxDist) {
//                    double xd = x;
//                    double yd = y;

//                    double pixInt1 = interpolateInt(refIntImage, xd, yd);
//                    double pixInt2 = (double)*(actIntImage.ptr<uchar>(transfR_int, transfC_int))/255.0;
//                    double pixInt2 = interpolateIntensityWithDepth(actIntImage, actDepImage,
//                                                                   transfC, transfR,
//                                                                   trfPoint3D(2));
//                    double pixDep1 = interpolateDep(refDepImage, xd, yd);
//                    double pixDep1 = *refDepImage.ptr<double>(y, x);
//                    double pixDep2 = *actDepImage.ptr<double>(transfR_int, transfC_int);
//                    double pixDep2 = interpolateDep(actDepImage, transfC, transfR);

//                    double pixInt1 = interpolateIntensityWithDepth(refIntImage, refDepImage,
//                                                                   xd, yd, trfPoint3D(2));
                    double pixInt1 = (double)*(refIntImage.ptr<uchar>(y, x))/255.0;
                    double pixInt2 = interpolateIntensityWithDepth(actIntImage, actDepImage,
                                                                   transfC, transfR,
                                                                   trfPoint3D(2));
                    double pixDep1 = trfPoint3D(2);
                    double pixDep2 = *actDepImage.ptr<double>(transfR_int, transfC_int);

                    //Assign the pixel residual and jacobian to its corresponding row
                    uint i = nCols * y + x;

                    //Residual of the pixel
                    double dInt = pixInt2 - pixInt1;
                    dInt = dInt < 0.025 ? 0 : dInt;
                    //double dDep = (vs - vt).dot(nt);
                    double dDep = nz * (pixDep2 - pixDep1);
                    dDep = pixDep1 < minDist ? 0 : dDep;
                    dDep = pixDep2 < minDist ? 0 : dDep;
                    double diff = abs(dDep);
                    dDep = diff > 0.05 ? 0 : dDep;
                    double wDep = *weight.ptr<double>(transfR_int, transfC_int) * depth;
                    double wInt = wDep * color;
                    wDep *= depthWeight;
                    //double maxCurv = 0.5;
                    //double minCurv = 0.2;
                    //wDep = wDep >= minCurv && wDep <= maxCurv ? wDep : 0;

                    jacobians(i,0)   = wInt * jacobianIntensity(0,0);
                    jacobians(i,1)   = wInt * jacobianIntensity(0,1);
                    jacobians(i,2)   = wInt * jacobianIntensity(0,2);
                    jacobians(i,3)   = wInt * jacobianIntensity(0,3);
                    jacobians(i,4)   = wInt * jacobianIntensity(0,4);
                    jacobians(i,5)   = wInt * jacobianIntensity(0,5);
                    jacobians(i*2,0) = wDep * jacobianDepth(0,0);
                    jacobians(i*2,1) = wDep * jacobianDepth(0,1);
                    jacobians(i*2,2) = wDep * jacobianDepth(0,2);
                    jacobians(i*2,3) = wDep * jacobianDepth(0,3);
                    jacobians(i*2,4) = wDep * jacobianDepth(0,4);
                    jacobians(i*2,5) = wDep * jacobianDepth(0,5);

                    residuals(nCols * transfR_int + transfC_int, 0) = wInt * dInt * color;
                    residuals(nCols * 2 * transfR_int + 2 * transfC_int, 0) = wDep * dDep * depth;

                    residualImage.at<double>(transfR_int, transfC_int) = wInt * abs(dInt) * color * 1;
                    residualImage.at<double>(nRows-1 + transfR_int, nCols-1 + transfC_int) = wDep * abs(dDep) * depth * 1;
                    if(diff > 0.01){
                        count[nThread] += 1;
                    }
                }
            }
        }, nthreads
        );

        imshow("Residual", residualImage);
        waitKey(1);
        double totalCount = 0;
        for (int i = 0; i < nthreads; ++i) {
            totalCount += count[i];
        }
        double outlierRatio = 100 * totalCount / (double)(nCols * nRows);
        //        cerr << "Count " << outlierRatio << " %" << endl;

        return outlierRatio;
    }

    Matrix4d getPoseExponentialMap2(VectorXd poseVector6D){
        double x = poseVector6D(0);
        double y = poseVector6D(1);
        double z = poseVector6D(2);
        double yaw = poseVector6D(3);
        double pitch = poseVector6D(4);
        double roll = poseVector6D(5);

        Matrix4d lie;
        lie << 0.f,  -roll, pitch,    x,
                roll,   0.f,  -yaw,    y,
                -pitch,   yaw,    0.f,   z,
                0.f,   0.f,    0.f, 0.f;

        lie = lie.exp();
        return lie;
    }

    //*********** Compute the rigid transformation matrix from the poseVector6D ************
    Matrix4d getMatrixRtFromPose6D(VectorXd poseVector6D){
        Matrix4d Rt = Matrix4d::Zero();
        double x = poseVector6D(0);
        double y = poseVector6D(1);
        double z = poseVector6D(2);
        double yaw = poseVector6D(3);
        double pitch = poseVector6D(4);
        double roll = poseVector6D(5);
        double sin_yaw = sin(yaw);
        double cos_yaw = cos(yaw);
        double sin_pitch = sin(pitch);
        double cos_pitch = cos(pitch);
        double sin_roll = sin(roll);
        double cos_roll = cos(roll);
        Rt(0,0) = cos_yaw * cos_pitch;
        Rt(0,1) = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
        Rt(0,2) = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll;
        Rt(0,3) = x;
        Rt(1,0) = sin_yaw * cos_pitch;
        Rt(1,1) = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;
        Rt(1,2) = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll;
        Rt(1,3) = y;
        Rt(2,0) = -sin_pitch;
        Rt(2,1) = cos_pitch * sin_roll;
        Rt(2,2) = cos_pitch * cos_roll;
        Rt(2,3) = z;
        Rt(3,0) = 0.0;
        Rt(3,1) = 0.0;
        Rt(3,2) = 0.0;
        Rt(3,3) = 1.0;
        return Rt;
    }

    bool doSingleIteration(MatrixXd &residuals, MatrixXd &jacobians,
                           double lambda, double speed, bool robust = false){

        //Weighting of residuals
//        if(robust){
//            weighting(residuals);
//        }
        MatrixXd gradients = jacobians.transpose() * residuals;
        MatrixXd hessian = jacobians.transpose() * jacobians;
        MatrixXd identity;
        identity.setZero(6,6);
//        identity(0,0) = hessian(0,0);
//        identity(1,1) = hessian(1,1);
//        identity(2,2) = hessian(2,2);
//        identity(3,3) = hessian(3,3);
//        identity(4,4) = hessian(4,4);
//        identity(5,5) = hessian(5,5);
        identity(0,0) = 1;
        identity(1,1) = 1;
        identity(2,2) = 1;
        identity(3,3) = 1;
        identity(4,4) = 1;
        identity(5,5) = 1;
        hessian += lambda * identity;
        //actualPoseVector6D -= speed * hessian.ldlt().solve(gradients);
        actualPoseVector6D -= hessian.inverse() * gradients;
        //Gets best transform until now
        double gradientNorm = gradients.norm();
        if(gradientNorm < lastGradientNorm){
            lastGradientNorm = gradientNorm;
            bestPoseVector6D = actualPoseVector6D;
            return true;
        }
        return false;
    }

    Matrix4d getPoseTransformOpen3d(shared_ptr<RGBDImage> source,
                                                       shared_ptr<RGBDImage> target,
                                                       Matrix4d odo_init){
        odometry::OdometryOption option = odometry::OdometryOption();
        //option.max_depth_diff_ = 0.3;
        std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> rgbd_odo =
                odometry::ComputeRGBDOdometry(
                    *source, *target, intrinsics, odo_init,
                    odometry::RGBDOdometryJacobianFromHybridTerm(),
                    option);
        Matrix4d trf = get<1>(rgbd_odo);        
        return trf;
    }

    std::pair<VectorXd, double> getPoseTransform(Mat &refGray, Mat &refDepth, Mat &actGray, Mat &actDepth,
                                                 bool refinement){

        int rows = refGray.rows;
        int cols = refGray.cols;
        //Generate weight based on normal map
        actDepth.convertTo(actDepth, CV_64FC1, 1.f/depthScale);
        refDepth.convertTo(refDepth, CV_64FC1, 1.f/depthScale);

        Mat normalMap = getNormalMapFromDepth(refDepth, intrinsics, 0, depthScale);
        Mat normalsWeight = getNormalWeight(normalMap, refDepth, intrinsics);
        Mat pyrWeights[4];
        pyrWeights[0] = normalsWeight;
        resize(normalsWeight, pyrWeights[1], Size(cols/2, rows/2), 0, 0, INTER_NEAREST);
        resize(normalsWeight, pyrWeights[2], Size(cols/4, rows/4), 0, 0, INTER_NEAREST);
        resize(normalsWeight, pyrWeights[3], Size(cols/8, rows/8), 0, 0, INTER_NEAREST);

        Mat pyrNormalMap[4];
        pyrNormalMap[0] = normalMap;
        pyrNormalMap[1] = getNormalMapFromDepth(refDepth, intrinsics, 1, depthScale);
        pyrNormalMap[2] = getNormalMapFromDepth(refDepth, intrinsics, 2, depthScale);
        pyrNormalMap[3] = getNormalMapFromDepth(refDepth, intrinsics, 3, depthScale);

        Mat tempRefGray, tempActGray;
        Mat tempRefDepth, tempActDepth;

        int iteratLevel[] = { 35, 5, 5, 5 };
        double lambdas[] = { 0.0002, 0.0002, 0.0002, 0.0002 };
        double threshold[] = { 80, 160, 160, 160 };
        int initialPyr = 0;
        for (int l = initialPyr; l >= 0; l--) {
            if(l != initialPyr){
                actualPoseVector6D = bestPoseVector6D;
            }
            int level = pow(2, l);
            int rows = refGray.rows/level;
            int cols = refGray.cols/level;

            resize(refGray, tempRefGray, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(refDepth, tempRefDepth, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(actGray, tempActGray, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(actDepth, tempActDepth, Size(cols, rows), 0, 0, INTER_NEAREST);

            this->lastGradientNorm = DBL_MAX;
            this->sum = DBL_MAX;
            cerr << "******Initialized with******" << endl;
            cerr << TransformVector6dToMatrix4d(actualPoseVector6D) << endl;
            cerr << "****************************" << endl;
            bool minimized = false;                        
            double m = 1;            
            double a = 1.1;
            double b = 1.1;
            int countMin = 0;
            int countNotMin = 0;
            bool color = true;
            bool depth = true;
            bool robust = false;
            double weight = 5;
            for (int i = 0; i < iteratLevel[l]; ++i) {
                MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);
                MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);                
                computeResidualsAndJacobians(tempRefGray, tempRefDepth,
                                             tempActGray, tempActDepth,
                                             pyrWeights[l], pyrNormalMap[l], residuals, jacobians,
                                             weight, l, color, depth);

                minimized = doSingleIteration(residuals, jacobians, m*lambdas[l], 1, robust);
                if (!minimized){
                    //cerr << "Not\n";
                    m *= 1/a;
                    actualPoseVector6D = bestPoseVector6D;
                    if(++countNotMin >= 3)
                        break;
                }
                else{
                    //cerr << "Yes\n";
                    m *= b;
                    countMin++;
                    countNotMin = 0;
                }                
            }
            cerr << "\nMinimized " << countMin << " times\n" << endl;
        }

//        MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);
//        MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);
//        double outlierRatio = computeResidualsAndJacobians(tempRefGray, tempRefDepth,
//                                                           tempActGray, tempActDepth,
//                                                           pyrWeights[0], normalMap, residuals, jacobians, 10, 0);
        return make_pair(bestPoseVector6D, 1);
    }
};
#endif
