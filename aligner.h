#ifndef ALIGNER_H
#define ALIGNER_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Open3D/Core/Camera/PinholeCameraIntrinsic.h>

#include "util.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace open3d;

class Aligner{

public:

    VectorXd actualPoseVector6D;
    VectorXd bestPoseVector6D;

    PinholeCameraIntrinsic intrinsecs;

    Mat sourceIntensityImage;
    Mat sourceDepthImage;
    Mat targetIntensityImage;
    Mat targetDepthImage;

    MatrixXd jacobianRt = MatrixXd(3,6);
    MatrixXd jacobianProj = MatrixXd(2,3);
    MatrixXd jacobianIntensity = MatrixXd(1,6);
    MatrixXd jacobianDepth = MatrixXd(1,6);
    MatrixXd gradPixIntensity = MatrixXd(1,2);
    MatrixXd gradPixDepth = MatrixXd(1,2);

    double lastGradientNorm = INT_MAX;
    double sum = INT_MAX;

    double gradIntensityScale = 0.0001;
    double gradDepthScale = 0.000001;

    Aligner(PinholeCameraIntrinsic intrinsecs){
        actualPoseVector6D.setZero(6);
        bestPoseVector6D.setZero(6);
        this->intrinsecs = intrinsecs;
    }

    VectorXd getPose6D(){
        return bestPoseVector6D;
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

    void computeResidualsAndJacobians(Mat &refIntensityImage, Mat &refDepthImage,
                                      Mat &actIntensityImage, Mat &actDepthImage,
                                      MatrixXd &residuals, MatrixXd &jacobians,
                                      int level)
    {
        Mat residualImage = Mat::zeros(refIntensityImage.rows*2, refIntensityImage.cols, CV_64FC1);
        Mat actDerivativeX, actDerivativeY, actDepDerivativeX, actDepDerivativeY;
        Scharr(actIntensityImage, actDerivativeX, CV_64F, 1, 0, gradIntensityScale * level, 0.0, cv::BORDER_DEFAULT);
        Scharr(actIntensityImage, actDerivativeY, CV_64F, 0, 1, gradIntensityScale * level, 0.0, cv::BORDER_DEFAULT);
        Scharr(actDepthImage, actDepDerivativeX, CV_64F, 1, 0, gradDepthScale * level, 0.0, cv::BORDER_DEFAULT);
        Scharr(actDepthImage, actDepDerivativeY, CV_64F, 0, 1, gradDepthScale * level, 0.0, cv::BORDER_DEFAULT);

        //Extrinsecs
        double x = actualPoseVector6D(0);
        double y = actualPoseVector6D(1);
        double z = actualPoseVector6D(2);
        double yaw = actualPoseVector6D(3);
        double pitch = actualPoseVector6D(4);
        double roll = actualPoseVector6D(5);

        //Intrinsecs
        double scaleFactor = 1.0/pow(2,level-1);
//        cerr << scaleFactor << endl;
        double cx = scaleFactor * intrinsecs.GetPrincipalPoint().first;
        double cy = scaleFactor * intrinsecs.GetPrincipalPoint().second;
        double fx = scaleFactor * intrinsecs.GetFocalLength().first;
        double fy = scaleFactor * intrinsecs.GetFocalLength().second;

        //Matrix |R|t| from 6D vector
        Matrix4d Rt = getMatrixRtFromPose6D(actualPoseVector6D);

        //Calculation of jacobians and residuals
        int nCols = refIntensityImage.cols;
        int nRows = refIntensityImage.rows;
        double thisSum = 0;

        for (int y = 0; y < nRows; ++y) {
            for (int x = 0; x < nCols; ++x) {
                //******* BEGIN Unprojection of DepthMap ********//
                Vector4d refPoint3D;
                Vector4d actPoint3D;
                refPoint3D(2) = *refDepthImage.ptr<ushort>(y, x)/5000.f;
                actPoint3D(2) = *actDepthImage.ptr<ushort>(y, x)/5000.f;

                refPoint3D(0) = (x - cx) * refPoint3D(2) * 1/fx;
                refPoint3D(1) = (y - cy) * refPoint3D(2) * 1/fy;
                refPoint3D(3) = 1;
                //                actPoint3D(0) = (x - cx) * actPoint3D(2) * 1/fx;
                //                actPoint3D(1) = (y - cy) * actPoint3D(2) * 1/fy;
                //                actPoint3D(3) = 1;

                //******* END Unprojection of DepthMap ********//
                //******* BEGIN Transformation of PointCloud ********//
                Vector4d trfPoint3D = Rt * refPoint3D;
                double invTransfZ = 1.0/trfPoint3D(2);
                //******* END Transformation of PointCloud ********//

                gradPixIntensity(0,0) = *actDerivativeX.ptr<double>(y, x);
                gradPixIntensity(0,1) = *actDerivativeY.ptr<double>(y, x);
                gradPixDepth(0,0) = *actDepDerivativeX.ptr<double>(y, x);
                gradPixDepth(0,1) = *actDepDerivativeY.ptr<double>(y, x);
                double px = refPoint3D(0);
                double py = refPoint3D(1);
                double pz = refPoint3D(2);

                //Derivative with respect to x
                jacobianRt(0,0) = 1.;
                jacobianRt(1,0) = 0.;
                jacobianRt(2,0) = 0.;

                //******* BEGIN Residual and Jacobians computation ********//
                //Derivative with respect to y
                jacobianRt(0,1) = 0.;
                jacobianRt(1,1) = 1.;
                jacobianRt(2,1) = 0.;

                //Derivative with respect to z
                jacobianRt(0,2) = 0.;
                jacobianRt(1,2) = 0.;
                jacobianRt(2,2) = 1.;

                //Derivative with respect to yaw
                jacobianRt(0,3) = py*(-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw))+pz*(sin(roll)*cos(yaw)-sin(pitch)*cos(roll)*sin(yaw))-cos(pitch)*px*sin(yaw);
                jacobianRt(1,3) = pz*(sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw))+py*(sin(pitch)*sin(roll)*cos(yaw)-cos(roll)*sin(yaw))+cos(pitch)*px*cos(yaw);
                jacobianRt(2,3) = 0.;

                //Derivative with respect to pitch
                jacobianRt(0,4) = cos(pitch)*py*sin(roll)*cos(yaw)+cos(pitch)*pz*cos(roll)*cos(yaw)-sin(pitch)*px*cos(yaw);
                jacobianRt(1,4) = cos(pitch)*py*sin(roll)*sin(yaw)+cos(pitch)*pz*cos(roll)*sin(yaw)-sin(pitch)*px*sin(yaw);
                jacobianRt(2,4) = -sin(pitch)*py*sin(roll)-sin(pitch)*pz*cos(roll)-cos(pitch)*px;

                //Derivative with respect to roll
                jacobianRt(0,5) = py*(sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw))+pz*(cos(roll)*sin(yaw)-sin(pitch)*sin(roll)*cos(yaw));
                jacobianRt(1,5) = pz*(-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw))+py*(sin(pitch)*cos(roll)*sin(yaw)-sin(roll)*cos(yaw));
                jacobianRt(2,5) = cos(pitch)*py*cos(roll)-cos(pitch)*pz*sin(roll);

                //Derivative with respect to x
                jacobianProj(0,0) = fx*invTransfZ;
                jacobianProj(1,0) = 0.;

                //Derivative with respect to y
                jacobianProj(0,1) = 0.;
                jacobianProj(1,1) = fy*invTransfZ;

                //Derivative with respect to z
                jacobianProj(0,2) = -(fx*trfPoint3D(0))*invTransfZ*invTransfZ;
                jacobianProj(1,2) = -(fy*trfPoint3D(1))*invTransfZ*invTransfZ;

                jacobianDepth = gradPixDepth * jacobianProj * jacobianRt;
                jacobianIntensity = gradPixIntensity * jacobianProj * jacobianRt;

                //******* BEGIN Projection of PointCloud on the image plane ********//
                double transfC = (trfPoint3D(0) * fx) * invTransfZ + cx;
                double transfR = (trfPoint3D(1) * fy) * invTransfZ + cy;
                int transfR_int = static_cast<int>(round(transfR));
                int transfC_int = static_cast<int>(round(transfC));
                //******* END Projection of PointCloud on the image plane ********//

                //Checks if this pixel projects inside of the source image
                if((transfR_int >= 0 && transfR_int < nRows) & (transfC_int >= 0 && transfC_int < nCols)) {
                    double pixInt1 = *(refIntensityImage.ptr<uchar>(y, x))/255.f;
                    double pixInt2 = *(actIntensityImage.ptr<uchar>(transfR_int, transfC_int))/255.f;
                    double pixDep1 = *(refDepthImage.ptr<ushort>(y, x))/5000.f;
                    double pixDep2 = *(actDepthImage.ptr<ushort>(transfR_int, transfC_int))/5000.f;
                    //Assign the pixel residual and jacobian to its corresponding row
                    uint i = nCols * y + x;

                    double wInt = 1;
                    double wDep = 1;
                    jacobians(i,0)  = wDep * jacobianDepth(0,0);
                    jacobians(i,1)  = wDep * jacobianDepth(0,1);
                    jacobians(i,2)  = wDep * jacobianDepth(0,2);
                    jacobians(i,3)  = wDep * jacobianDepth(0,3);
                    jacobians(i,4)  = wDep * jacobianDepth(0,4);
                    jacobians(i,5)  = wDep * jacobianDepth(0,5);
                    jacobians(i*2,0) = wInt * jacobianIntensity(0,0);
                    jacobians(i*2,1) = wInt * jacobianIntensity(0,1);
                    jacobians(i*2,2) = wInt * jacobianIntensity(0,2);
                    jacobians(i*2,3) = wInt * jacobianIntensity(0,3);
                    jacobians(i*2,4) = wInt * jacobianIntensity(0,4);
                    jacobians(i*2,5) = wInt * jacobianIntensity(0,5);

                    //Residual of the pixel
                    double dInt = pixInt2 - pixInt1;
                    double dDep = pixDep2 - pixDep1;
                    dDep = fabs(dDep) < 0.2 ? 0 : dDep;
                    thisSum += fabs(dDep) + fabs(dInt);
                    residuals(nCols * transfR_int + transfC_int, 0) = wDep * dDep;                    
                    residuals(nCols * 2 * transfR_int + 2 * transfC_int, 0) = wInt * dInt;
                    residualImage.at<double>(transfR_int, transfC_int) = wDep * dDep;
                    residualImage.at<double>(nRows-1 + transfR_int, nCols-1 + transfC_int) = dInt;
                }
            }
        }
        imshow("Residual", residualImage);
        waitKey(0);

        if(thisSum < sum){
            sum = thisSum;
            cerr << "best pose by sum: " << sum << endl;
            bestPoseVector6D = actualPoseVector6D;
        }
        //        cerr << thisSum << endl;
    }

    //*********** Compute the rigid transformation matrix from the poseVector6D ************//
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

    bool doSingleIteration(MatrixXd &residuals, MatrixXd &jacobians, int level){

        MatrixXd gradients = jacobians.transpose() * residuals;
        actualPoseVector6D -= 1 * ((jacobians.transpose()*jacobians).inverse() * gradients);
        //Gets best transform until now
        //        double gradientNorm = gradients.norm();
        //        if(gradientNorm < lastGradientNorm){
        //            lastGradientNorm = gradientNorm;
        //            cerr << "best pose by grad: " << lastGradientNorm << endl;
        //            bestPoseVector6D = actualPoseVector6D;
        //        }
        //        if(gradients.norm() < 300){
        //            bestPoseVector6D = actualPoseVector6D;
        //            return true;
        //        }
        return false;
    }

    void getPoseTransform(Mat &refGray, Mat &refDepth, Mat &actGray, Mat &actDepth){

        Mat tempRefGray, tempActGray;
        Mat tempRefDepth, tempActDepth;

        this->actualPoseVector6D = this->bestPoseVector6D;
        int iterMult = 1;
        for (int level = 2; level >= 1; level/=2) {
            int rows = refGray.rows/level;
            int cols = refGray.cols/level;

            resize(refGray, tempRefGray, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(refDepth, tempRefDepth, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(actGray, tempActGray, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(actDepth, tempActDepth, Size(cols, rows), 0, 0, INTER_NEAREST);

            Mat residualImage;
            MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);

            this->lastGradientNorm = INT_MAX;
            this->sum = INT_MAX;
            cerr << "Iniciando de:" << endl;
            cerr << getMatrixRtFromPose6D(actualPoseVector6D);
            for (int i = 0; i < iterMult * 5; ++i) {
                MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);
                this->computeResidualsAndJacobians(tempRefGray, tempRefDepth, tempActGray, tempActDepth, residuals, jacobians, level);
                bool converged = doSingleIteration(residuals, jacobians, level);
                if(converged) break;
            }
//            iterMult++;
            //cerr << this->getMatrixRtFromPose6D(bestPoseVector6D) << "\n\n";
        }
    }
};
#endif
