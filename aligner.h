#ifndef ALIGNER_H
#define ALIGNER_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui.hpp>
#include <Open3D/Core/Camera/PinholeCameraIntrinsic.h>

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

    PinholeCameraIntrinsic intrinsecs;

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

    Aligner(PinholeCameraIntrinsic intrinsecs){
        actualPoseVector6D.setZero(6);
        bestPoseVector6D.setZero(6);
        this->intrinsecs = intrinsecs;
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

    void computeResidualsAndJacobians(Mat &refIntImage, Mat &refDepImage,
                                      Mat &actIntImage, Mat &actDepImage,
                                      MatrixXd &residuals, MatrixXd &jacobians,                                      
                                      int level, bool color = true, bool depth = true)
    {        
        double scaleFactor = 1.0 / pow(2, level);
        //level comes in reverse order 2 > 1 > 0
        double intScales[] = { 0.0001, 0.0001, 0.0001 };
        double depScales[] = { 0.001, 0.001, 0.001 };
        double gradLevels[] = {0.0004, 0.0004, 0.0009};
        Mat residualImage = Mat::zeros(refIntImage.rows * 2, refIntImage.cols, CV_64FC1);
        Mat actIntDerivX, actIntDerivY, actDepDerivX, actDepDerivY;
        Scharr(actIntImage, actIntDerivX, CV_64F, 1, 0, intScales[level], 0.0, cv::BORDER_DEFAULT);
        Scharr(actIntImage, actIntDerivY, CV_64F, 0, 1, intScales[level], 0.0, cv::BORDER_DEFAULT);
        Scharr(actDepImage, actDepDerivX, CV_64F, 1, 0, depScales[level], 0.0, cv::BORDER_DEFAULT);
        Scharr(actDepImage, actDepDerivY, CV_64F, 0, 1, depScales[level], 0.0, cv::BORDER_DEFAULT);

        Mat filteredNormals = getMaskOfNormalsFromDepth(actDepImage, intrinsecs, level);
        imshow("wDep", filteredNormals);

        //Extrinsecs
//        double x = bestPoseVector6D(0);
//        double y = bestPoseVector6D(1);
//        double z = bestPoseVector6D(2);
        double yaw = bestPoseVector6D(3);
        double pitch = bestPoseVector6D(4);
        double roll = bestPoseVector6D(5);

        //Intrinsecs
        double cx = scaleFactor * intrinsecs.GetPrincipalPoint().first;
        double cy = scaleFactor * intrinsecs.GetPrincipalPoint().second;
        double fx = scaleFactor * intrinsecs.GetFocalLength().first;
        double fy = scaleFactor * intrinsecs.GetFocalLength().second;

        //Matrix |R|t| from 6D vector
        Matrix4d Rt = getMatrixRtFromPose6D(actualPoseVector6D);

        //Calculation of jacobians and residuals
        int nCols = refIntImage.cols;
        int nRows = refIntImage.rows;
        double thisSum = 0;
        int count = 0;

        int nthreads = 4;
        cv::setNumThreads(nthreads);
        parallel_for_(Range(0, nCols*nRows), [&](const Range& range)
        {
            MatrixXd jacobianRt = MatrixXd(3,6);
            MatrixXd jacobianRt_z = MatrixXd(1,6);
            MatrixXd jacobianProj = MatrixXd(2,3);
            MatrixXd jacobianIntensity = MatrixXd(1,6);
            MatrixXd jacobianDepth = MatrixXd(1,6);
            MatrixXd gradPixIntensity = MatrixXd(1,2);
            MatrixXd gradPixDepth = MatrixXd(1,2);

            for (int r = range.start; r < range.end; r++) {

                int y = r / refDepImage.cols;
                int x = r % refDepImage.cols;
                if(*refDepImage.ptr<double>(y, x) == 0)
                    continue;

                gradPixIntensity(0,0) = *actIntDerivX.ptr<double>(y, x);
                gradPixIntensity(0,1) = *actIntDerivY.ptr<double>(y, x);
                gradPixDepth(0,0) = *actDepDerivX.ptr<double>(y, x);
                gradPixDepth(0,1) = *actDepDerivY.ptr<double>(y, x);

                double gradX = gradPixIntensity(0,0);
                double gradY = gradPixIntensity(0,1);

                double wInt = 1;
//                double wInt = (gradX * gradX + gradY * gradY) > gradLevels[level] ? 1 : 0;
//                if( wInt == 0 ){
//                    count++;
//                    continue;
//                }

                //******* BEGIN Unprojection of DepthMap ********
                Vector4d refPoint3D;
                refPoint3D(2) = *refDepImage.ptr<double>(y, x);
                refPoint3D(0) = (x - cx) * refPoint3D(2) * 1/fx;
                refPoint3D(1) = (y - cy) * refPoint3D(2) * 1/fy;
                refPoint3D(3) = 1;
                //******* END Unprojection of DepthMap ********

                //******* BEGIN Transformation of PointCloud ********
                Vector4d trfPoint3D = Rt * refPoint3D;
                double invTransfZ = 1.0 / trfPoint3D(2);
                //******* END Transformation of PointCloud ********

                double px = refPoint3D(0);
                double py = refPoint3D(1);
                double pz = refPoint3D(2);

                //Derivative with respect to x
                jacobianRt(0,0) = 1.;
                jacobianRt(1,0) = 0.;
                jacobianRt(2,0) = 0.;

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

                jacobianRt_z(0,0) = jacobianRt(2,0);
                jacobianRt_z(0,1) = jacobianRt(2,1);
                jacobianRt_z(0,2) = jacobianRt(2,2);
                jacobianRt_z(0,3) = jacobianRt(2,3);
                jacobianRt_z(0,4) = jacobianRt(2,4);
                jacobianRt_z(0,5) = jacobianRt(2,5);

                jacobianDepth = gradPixDepth * jacobianProj * jacobianRt - jacobianRt_z;
                jacobianIntensity = gradPixIntensity * jacobianProj * jacobianRt;

                //******* BEGIN Projection of PointCloud on the image plane ********
                double transfC = (trfPoint3D(0) * fx) * invTransfZ + cx;
                double transfR = (trfPoint3D(1) * fy) * invTransfZ + cy;
                int transfR_int = static_cast<int>(round(transfR));
                int transfC_int = static_cast<int>(round(transfC));
                //******* END Projection of PointCloud on the image plane ********
                //Checks if this pixel projects inside of the source image
                if((transfR_int >= 0 && transfR_int < nRows) && (transfC_int >= 0 && transfC_int < nCols)
                        && refPoint3D(2) > minDist && refPoint3D(2) < maxDist) {
                    double pixInt1 = *(refIntImage.ptr<uchar>(y, x))/255.f;
                    double pixInt2 = *(actIntImage.ptr<uchar>(transfR_int, transfC_int))/255.f;
                    double pixDep1 = trfPoint3D(2) * 0.1;
                    double pixDep2 = *(actDepImage.ptr<double>(transfR_int, transfC_int)) * 0.1;

                    //Assign the pixel residual and jacobian to itsa corresponding row
                    uint i = nCols * y + x;

                    //Residual of the pixel
                    double dInt = pixInt2 - pixInt1;
                    double dDep = pixDep2 - pixDep1;
                    dDep = abs(dDep) > 0.6 ? 0 : dDep;
                    dDep = pixDep1 == 0 ? 0 : dDep;
                    dDep = pixDep2 == 0 ? 0 : dDep;
                    double wDep = *filteredNormals.ptr<double>(transfR_int, transfC_int);

                    double v = 0.2f;
                    double wd = (v + 0.f) / (0.05 + dDep*dDep);
                    wDep *= sqrt(wd);

                    jacobians(i,0)   = wDep * jacobianDepth(0,0);
                    jacobians(i,1)   = wDep * jacobianDepth(0,1);
                    jacobians(i,2)   = wDep * jacobianDepth(0,2);
                    jacobians(i,3)   = wDep * jacobianDepth(0,3);
                    jacobians(i,4)   = wDep * jacobianDepth(0,4);
                    jacobians(i,5)   = wDep * jacobianDepth(0,5);
                    jacobians(i*2,0) = wInt * jacobianIntensity(0,0);
                    jacobians(i*2,1) = wInt * jacobianIntensity(0,1);
                    jacobians(i*2,2) = wInt * jacobianIntensity(0,2);
                    jacobians(i*2,3) = wInt * jacobianIntensity(0,3);
                    jacobians(i*2,4) = wInt * jacobianIntensity(0,4);
                    jacobians(i*2,5) = wInt * jacobianIntensity(0,5);

                    residuals(nCols * transfR_int + transfC_int, 0) = wDep * dDep * depth;
                    residuals(nCols * 2 * transfR_int + 2 * transfC_int, 0) = wInt * dInt * color;

                    residualImage.at<double>(transfR_int, transfC_int) = wInt * dInt * color;
                    residualImage.at<double>(nRows-1 + transfR_int, nCols-1 + transfC_int) = 500 * wDep * dDep * depth;
                }
            }
        }, nthreads
        );

        if (level == 0){
            resize(residualImage, residualImage, Size(residualImage.cols/2, residualImage.rows/2));
//            Matrix4d Id = Matrix4d::Identity();
//            Mat m1 = transfAndProject(refDepImage, 1, Rt, intrinsecs);
//            Mat m2 = transfAndProject(actDepImage, 1, Id, intrinsecs);
//            Mat m3;
//            m3 = m2 - m1;
//            threshold(m3, m3, 0.002, 0, THRESH_TOZERO_INV);
//            Scalar mean, stddev;
//            meanStdDev(residualImage, mean, stddev);
//            cerr << "Mean " << mean[0] << " Stddev " << stddev[0] << endl;
//            m3 *= 500;
//            imshow("difference", m3);
//            waitKey(1);
        }

        imshow("Residual", residualImage);
        waitKey(1);

        thisSum = residuals.squaredNorm();
        if(thisSum < sum){
            sum = thisSum;
            cerr << "best pose summation: " << sum << endl;
            bestPoseVector6D = actualPoseVector6D;
        }
//        cerr << "Count " << count / (double)(nCols * nRows) << " %" << endl;
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
                           double lambda, double threshold){

        double chi = residuals.squaredNorm();        
        if(chi > threshold){
            cerr << "Escalando o erro " << endl;
            residuals *= sqrt(threshold/chi);
        }
        MatrixXd gradients = jacobians.transpose() * residuals;
        actualPoseVector6D -= lambda * ((jacobians.transpose()*jacobians).inverse() * gradients);
        //Gets best transform until now
        double gradientNorm = gradients.norm();
        if(gradientNorm < lastGradientNorm){
            lastGradientNorm = gradientNorm;
            cerr << "best pose by grad: " << lastGradientNorm << endl;
            bestPoseVector6D = actualPoseVector6D;
        }
        return false;
    }

    void getPoseTransform(Mat &refGray, Mat &refDepth, Mat &actGray, Mat &actDepth,
                          bool refinement){

        refDepth.convertTo(refDepth, CV_64FC1, 0.0002);
        actDepth.convertTo(actDepth, CV_64FC1, 0.0002);
        Mat tempRefGray, tempActGray;
        Mat tempRefDepth, tempActDepth;

        this->actualPoseVector6D.setZero(6);
        int iteratLevel[] = { 7, 3, 3 };

        for (int l = 2; l >= 0; l--) {
            cerr << "LEVEL " << l << endl;
            int level = pow(2, l);
            int rows = refGray.rows/level;
            int cols = refGray.cols/level;

            resize(refGray, tempRefGray, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(refDepth, tempRefDepth, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(actGray, tempActGray, Size(cols, rows), 0, 0, INTER_NEAREST);
            resize(actDepth, tempActDepth, Size(cols, rows), 0, 0, INTER_NEAREST);

            Mat residualImage;
            this->lastGradientNorm = DBL_MAX;
            this->sum = DBL_MAX;
            cerr << "Iniciando de:" << endl;
            cerr << getMatrixRtFromPose6D(actualPoseVector6D);
            double lambdas[] = { 1, 1, 2 };
            double threshold[] = { 1000, 160, 80 };
            for (int i = 0; i < iteratLevel[l]; ++i) {
                MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);
                MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);
                computeResidualsAndJacobians(tempRefGray, tempRefDepth,
                                             tempActGray, tempActDepth,
                                             residuals, jacobians, l);
                doSingleIteration(residuals, jacobians, lambdas[l], threshold[l]);
            }
        }
        if (refinement){
            cerr << "REFINEMENT\n";
            int rows = refGray.rows;
            int cols = refGray.cols;

            for (int i = 0; i < 5; ++i) {
                MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);
                MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);
                computeResidualsAndJacobians(tempRefGray, tempRefDepth,
                                             tempActGray, tempActDepth,
                                             residuals, jacobians, 0, false, true);
                doSingleIteration(residuals, jacobians, 800, 200);
            }
        }
    }
};
#endif
