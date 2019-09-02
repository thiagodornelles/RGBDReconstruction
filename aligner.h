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
    double depthScale = 5000.0;

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

    double computeResidualsAndJacobians(Mat &refIntImage, Mat &refDepImage,
                                        Mat &actIntImage, Mat &actDepImage,
                                        Mat &weight, Mat &normalMap,
                                        MatrixXd &residuals, MatrixXd &jacobians,
                                        int level, bool color = true, bool depth = true)
    {
        double scaleFactor = 1.0 / pow(2, level);
        //level comes in reverse order 2 > 1 > 0
        double intScales[] = { 0.0001, 0.0001, 0.0001 };
        double depScales[] = { 0.0001, 0.0001, 0.0001 };
        //        double intScales[] = { 0.001, 0.001, 0.001 };
        //        double depScales[] = { 0.001, 0.001, 0.001 };

        Mat residualImage = Mat::zeros(refIntImage.rows * 2, refIntImage.cols, CV_64FC1);
        Mat actIntDerivX = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);
        Mat actIntDerivY = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);
        Mat actDepDerivX = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);
        Mat actDepDerivY = Mat::zeros(refIntImage.rows, refIntImage.cols, CV_64FC1);

        Scharr(actIntImage, actIntDerivX, CV_64F, 1, 0, intScales[level], 0.0, cv::BORDER_DEFAULT);
        Scharr(actIntImage, actIntDerivY, CV_64F, 0, 1, intScales[level], 0.0, cv::BORDER_DEFAULT);
        Scharr(actDepImage, actDepDerivX, CV_64F, 1, 0, depScales[level], 0.0, cv::BORDER_DEFAULT);
        Scharr(actDepImage, actDepDerivY, CV_64F, 0, 1, depScales[level], 0.0, cv::BORDER_DEFAULT);
        //        Sobel(actIntImage, actIntDerivX, CV_64F, 1, 0, 3, intScales[level], 0.0, cv::BORDER_DEFAULT);
        //        Sobel(actIntImage, actIntDerivY, CV_64F, 0, 1, 3, intScales[level], 0.0, cv::BORDER_DEFAULT);
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
        double cx = scaleFactor * intrinsecs.GetPrincipalPoint().first;
        double cy = scaleFactor * intrinsecs.GetPrincipalPoint().second;
        double fx = scaleFactor * intrinsecs.GetFocalLength().first;
        double fy = scaleFactor * intrinsecs.GetFocalLength().second;

        //Matrix |R|t| from 6D vector
        //        Matrix4d Rt = getMatrixRtFromPose6D(actualPoseVector6D);
        Matrix4d Rt = getPoseExponentialMap2(actualPoseVector6D);

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
            MatrixXd jacobianRt_z = MatrixXd(1,6);
            MatrixXd jacobianProj = MatrixXd(2,3);
            MatrixXd jacobianIntensity = MatrixXd(1,6);
            MatrixXd jacobianDepth = MatrixXd(1,6);
            MatrixXd gradPixIntensity = MatrixXd(1,2);
            MatrixXd gradPixDepth = MatrixXd(1,2);

            int nThread = range.start/range.size();
            for (int r = range.start; r < range.end; r++) {

                int y = r / refDepImage.cols;
                int x = r % refDepImage.cols;

                if(*refDepImage.ptr<double>(y, x) < minDist)
                    continue;

                gradPixIntensity(0,0) = *actIntDerivX.ptr<double>(y, x);
                gradPixIntensity(0,1) = *actIntDerivY.ptr<double>(y, x);
                gradPixDepth(0,0) = *actDepDerivX.ptr<double>(y, x);
                gradPixDepth(0,1) = *actDepDerivY.ptr<double>(y, x);

                double wInt = gradPixIntensity(0,0) * gradPixIntensity(0,0);
                wInt += gradPixIntensity(0,1) * gradPixIntensity(0,1);
                wInt = sqrt(wInt);

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
                double w1 = -ny*pz + nz*py;
                double w2 =  nx*pz - nz*px;
                double w3 = -nx*py + ny*px;

                jacobianDepth(0) = nx;
                jacobianDepth(1) = ny;
                jacobianDepth(2) = nz;
                jacobianDepth(3) = w1;
                jacobianDepth(4) = w2;
                jacobianDepth(5) = w3;

                //                jacobianDepth = gradPixDepth * jacobianProj * jacobianRt - jacobianRt_z;
                jacobianIntensity = gradPixIntensity * jacobianProj * jacobianRt;

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
                    double pixInt1 = (double)*(refIntImage.ptr<uchar>(y, x))/255.0;
                    double pixInt2 = (double)*(actIntImage.ptr<uchar>(transfR_int, transfC_int))/255.0;
                    double pixDep1 = trfPoint3D(2);
                    double pixDep2 = *(actDepImage.ptr<double>(transfR_int, transfC_int));

                    //Assign the pixel residual and jacobian to its corresponding row
                    uint i = nCols * y + x;

                    //Residual of the pixel
                    double dInt = pixInt2 - pixInt1;
                    //                    dInt = abs(dInt) > 0.3 ? 0 : dInt;
                    double dDep = pixDep2 - pixDep1;
                    dDep = pixDep1 == 0 ? 0 : dDep;
                    dDep = pixDep2 * dDep == 0 ? 0 : dDep;
                    double diff = abs(dDep);
                    dDep = diff > 0.0005 ? 0 : dDep;
                    double wDep = *weight.ptr<double>(transfR_int, transfC_int);
                    //                    double maxCurv = 0.5;
                    //                    double minCurv = 0.2;
                    //                    wDep = wDep >= minCurv && wDep <= maxCurv ? wDep : 0;

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

                    residualImage.at<double>(transfR_int, transfC_int) = dInt * color;
                    residualImage.at<double>(nRows-1 + transfR_int, nCols-1 + transfC_int) = 1000 * dDep * depth;
                    if(diff > 0.0003){
                        count[nThread] += 1;
                    }
                }
            }
        }, nthreads
        );

        if (level == 0){
            resize(residualImage, residualImage, Size(residualImage.cols/2, residualImage.rows/2));
            Matrix4d Id = Matrix4d::Identity();
            Mat m1 = transfAndProject(refDepImage, 1, Rt, intrinsecs);
            Mat m2 = transfAndProject(actDepImage, 1, Id, intrinsecs);
            Mat m3;
            m3 = m2 - m1;
            threshold(m3, m3, 0.003, 0, THRESH_TOZERO_INV);
            //            Scalar mean, stddev;
            //            meanStdDev(residualImage, mean, stddev);
            //            cerr << "Mean " << mean[0] << " Stddev " << stddev[0] << endl;
            m3 *= 200;
            imshow("difference", m3);
        }

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

    VectorXd getPoseMatExpToVector(Eigen::Matrix4d_u matrix){
        VectorXd retVec;
        retVec.setZero(6);
        matrix = matrix.log();
        retVec(0) = matrix(0,3)/5000.;
        retVec(1) = matrix(1,3)/5000.;
        retVec(2) = matrix(2,3)/5000.;
        retVec(3) = matrix(2,1);
        retVec(4) = matrix(0,2);
        retVec(5) = matrix(1,0);
        return retVec;
    }

    Matrix4d getPoseExponentialMap(VectorXd poseVector6D){
        Matrix4d pose = Matrix4d::Identity();
        double x = poseVector6D(0);
        double y = poseVector6D(1);
        double z = poseVector6D(2);
        double yaw = poseVector6D(3);
        double pitch = poseVector6D(4);
        double roll = poseVector6D(5);

        Vector3d w, t;
        w << yaw, pitch, roll;
        t << x, y, z;

        double phi = w.norm();

        //        if( phi > 100. * std::numeric_limits<double>::epsilon()){
        Matrix3d hat;
        hat(0,0) = 0;
        hat(1,0) = roll;
        hat(2,0) = -pitch;
        hat(0,1) = -roll;
        hat(1,1) = 0;
        hat(2,1) = yaw;
        hat(0,2) = pitch;
        hat(1,2) = -yaw;
        hat(2,2) = 0;
        pose.block( 0, 0, 3, 3 ) = hat;
        pose.block( 0, 3, 3, 1 ) = t;

        //RODRIGUES
        //            Matrix3d R = Matrix3d::Identity();
        //            if( phi > 100. * std::numeric_limits<double>::epsilon() )
        //            {
        //                double inv_phi = 1. / phi;
        //                R += inv_phi * hat * sin(phi) + (inv_phi * inv_phi) *
        //                        hat * hat * (1. - cos(phi));
        //            }
        //            pose.block( 0, 0, 3, 3 ) = R;
        //            double inv_phi = 1. / phi;
        //            Matrix3d V = Matrix3d::Identity() +
        //                    ( inv_phi * inv_phi ) * ( 1. - cos( phi )  ) * hat +
        //                    ( inv_phi * inv_phi * inv_phi ) * ( phi - sin( phi ) ) * hat * hat;
        //            pose.block( 0, 3, 3, 1 ) = V * t;
        //        }
        //        else{
        //            pose.block( 0, 3, 3, 1 ) = t;
        //        }
        return pose;
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
        cerr << "chi squared " << chi << endl;
        if(chi > 10){
            cerr << "Escalando o erro " << endl;
            residuals *= sqrt(10.f/chi);
        }

        MatrixXd gradients = jacobians.transpose() * residuals;
        MatrixXd hessian = jacobians.transpose() * jacobians;
        MatrixXd identity;
        identity.setZero(6,6);
        identity(0,0) = hessian(0,0);
        identity(1,1) = hessian(1,1);
        identity(2,2) = hessian(2,2);
        identity(3,3) = hessian(3,3);
        identity(4,4) = hessian(4,4);
        identity(5,5) = hessian(5,5);
        hessian += lambda * identity;
        //        actualPoseVector6D -= hessian.ldlt().solve(gradients);
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

    std::pair<VectorXd, bool> getPoseTransform(Mat &refGray, Mat &refDepth, Mat &actGray, Mat &actDepth,
                                               bool refinement){

        //Generate weight based on normal map
        actDepth.convertTo(actDepth, CV_64FC1, 1.f/depthScale);
        refDepth.convertTo(refDepth, CV_64FC1, 1.f/depthScale);
        Mat normalMap = getNormalMapFromDepth(refDepth, intrinsecs, 0, depthScale);
        Mat normalsWeight = getNormalWeight(normalMap, refDepth, intrinsecs);

        Mat pyrWNormals[3];
        pyrWNormals[0] = normalsWeight;
        int rows = refGray.rows;
        int cols = refGray.cols;
        resize(normalsWeight, pyrWNormals[1], Size(cols/2, rows/2), 0, 0, INTER_NEAREST);
        resize(normalsWeight, pyrWNormals[2], Size(cols/4, rows/4), 0, 0, INTER_NEAREST);

        Mat tempRefGray, tempActGray;
        Mat tempRefDepth, tempActDepth;

        int iteratLevel[] = { 10, 5, 5 };
        double lambdas[] = { 0.0002, 0.0002, 0.0002 };
        double threshold[] = { 80, 160, 160 };

        for (int l = 2; l >= 0; l--) {
            if(l != 2){
                actualPoseVector6D = bestPoseVector6D;
            }
            else{
                actualPoseVector6D.setZero(6);
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
            cerr << "Iniciando de:" << endl;
            cerr << getPoseExponentialMap2(actualPoseVector6D);
            bool minimized = true;
            double m = 1;
            double a = 1.1;
            double b = 1.1;
            for (int i = 0; i < iteratLevel[l]; ++i) {
                MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);
                MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);
                computeResidualsAndJacobians(tempRefGray, tempRefDepth,
                                             tempActGray, tempActDepth,
                                             pyrWNormals[l], normalMap, residuals, jacobians, l);

                minimized = doSingleIteration(residuals, jacobians, m*lambdas[l], threshold[l]);
                if (!minimized){
                    m *= 1/a;
                    cerr << m*lambdas[l] << endl;
                    actualPoseVector6D = bestPoseVector6D;
                }
                else{
                    m *= b;
                    cerr << "minimized " << m*lambdas[l] << endl;
                }
            }
        }
        if (refinement){
            cerr << "REFINEMENT\n";
            int rows = refGray.rows;
            int cols = refGray.cols;
            bool minimized = false;
            double m = 1;
            double a = 1.1;
            double b = 1.5;
            for (int i = 0; i < 5; ++i) {
                MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);
                MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);
                computeResidualsAndJacobians(tempRefGray, tempRefDepth,
                                             tempActGray, tempActDepth,
                                             pyrWNormals[0], normalMap,
                        residuals, jacobians, 0, false, true);
                minimized = doSingleIteration(residuals, jacobians, m*lambdas[0], threshold[0]);
                if (!minimized){
                    m *= 1/a;
                    cerr << m*lambdas[0] << endl;
                    //                    actualPoseVector6D = bestPoseVector6D;
                }
                else{
                    m *= b;
                    cerr << "minimized " << m*lambdas[0] << endl;
                }
            }
        }

        MatrixXd jacobians = MatrixXd::Zero(rows * cols * 2, 6);
        MatrixXd residuals = MatrixXd::Zero(rows * cols * 2, 1);
        double outlierRatio = computeResidualsAndJacobians(tempRefGray, tempRefDepth,
                                                           tempActGray, tempActDepth,
                                                           pyrWNormals[0], normalMap, residuals, jacobians, 0);
        if(outlierRatio > 3){
            return make_pair(bestPoseVector6D, false);
        }
        return make_pair(bestPoseVector6D, true);
    }
};
#endif
