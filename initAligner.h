#include <Open3D/Open3D.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/IO/ClassIO/ImageIO.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace open3d;
using namespace open3d::geometry;


//returns estimated scale
double getMatchingsBetween2Frames(Mat &depth1, Mat &rgb1, Mat &rgb2,
                                vector< vector<DMatch> > &matchings,
                                vector<KeyPoint> &keyPoints1, vector<KeyPoint> &keyPoints2,
                                vector<Point2f> &points1,
                                vector<Point2f> &points2,
                                vector<Point3f> &points1_3d,
                                double focalLenght, double cx, double cy){
    Mat desc1, desc2;
    vector< vector<DMatch> > matches;

    Ptr<Feature2D> orb = ORB::create(3000);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    orb->detectAndCompute(rgb1, noArray(), keyPoints1, desc1, false);
    orb->detectAndCompute(rgb2, noArray(), keyPoints2, desc2, false);

    matcher->knnMatch(desc1, desc2, matches, 2);

    //Pegando somente matchings mais confiáveis
    double meanDist = 0;
    for (int i = 0; i < matches.size(); ++i) {
        DMatch d1  = matches[i][0];
        DMatch d2  = matches[i][1];

        KeyPoint kt1 = keyPoints1.at(d1.queryIdx);
        KeyPoint kt2 = keyPoints2.at(d1.trainIdx);
        double z1 = *depth1.ptr<double>(kt2.pt.y, kt2.pt.x);
        double z2 = *depth1.ptr<double>(kt2.pt.y, kt2.pt.x);
        double z3 = *depth1.ptr<double>(kt2.pt.y, kt2.pt.x);
        double z4 = *depth1.ptr<double>(kt2.pt.y, kt2.pt.x);

        if (d1.distance < 0.8 * d2.distance
                && z1 > 0 && z2 > 0 && z3 > 0 && z4 > 0)
        {
            double dx = kt1.pt.x - kt2.pt.x;
            double dy = kt1.pt.y - kt2.pt.y;
            double dist = sqrt(dx*dx + dy*dy);
            if (dist < 50){
                //cerr << kt1.pt << " " << kt2.pt << endl;
                meanDist += dist;
                vector<DMatch> v;
                v.push_back(d1);
                matchings.push_back(v);
                points1.push_back(kt1.pt);
                points2.push_back(kt2.pt);

                Vector4d refPoint3D;
                refPoint3D(2) = *depth1.ptr<double>(kt2.pt.y, kt2.pt.x);
                refPoint3D(0) = (kt2.pt.x - cx) * refPoint3D(2) * 1/focalLenght;
                refPoint3D(1) = (kt2.pt.y - cy) * refPoint3D(2) * 1/focalLenght;
                refPoint3D(3) = 1;
                Point3f v3d(refPoint3D(0), refPoint3D(1), refPoint3D(2));
                points1_3d.push_back(v3d);
            }
        }
    }

    Mat image1 = rgb1.clone();
    Mat image2 = rgb2.clone();
    for (int i = 0; i < points1.size(); ++i) {
//        DMatch d1  = matchings[i][0];
        Point2f pt1 = points1.at(i);
        Point2f pt2 = points2.at(i);
        circle(image1, pt1, 3, cvScalar(255,0,0));
        line(image1, pt1, pt2, cvScalar(0,255,0));
        circle(image2, pt2, 3, cvScalar(0,0,255));
    }
    imshow("matchings1", image1);
    imshow("matchings2", image2);
    return meanDist/points1.size();
}

VectorXd coarseRegistration(Mat &depth2, Mat &rgb2, Mat &depth1, Mat &rgb1,
                                   double focalLenght, double cx, double cy){


    Eigen::Matrix4d transf = Eigen::Matrix4d::Identity();

    //ORB Matching
    vector< vector<DMatch> > matchings;
    vector<KeyPoint> keyPoints1;
    vector<KeyPoint> keyPoints2;
    vector<Point2f> points1;
    vector<Point2f> points2;

    vector<Point3f> points1_3d;

    double scale = getMatchingsBetween2Frames(depth1, rgb1, rgb2, matchings,
                                              keyPoints1, keyPoints2,
                                              points1, points2, points1_3d,
                                              focalLenght, cx, cy);
    cerr << "Scale" << scale << endl;
    //Recovering pose from essential matrix
    //Mat t, R;
    //Mat E = findEssentialMat(points2, points1, focalLenght, Point2d(cx, cy), RANSAC, 0.9999999);
    //recoverPose(E, points2, points1, R, t, focalLenght, Point2d(cx, cy));

    Mat camMatrix = Mat::zeros(cv::Size(3, 3), CV_64FC1);
    camMatrix.at<double>(0,0) = focalLenght;
    camMatrix.at<double>(1,1) = focalLenght;
    camMatrix.at<double>(0,2) = cx;
    camMatrix.at<double>(1,2) = cy;
    camMatrix.at<double>(2,2) = 1;

    Mat rvec = Mat::zeros(3, 1, CV_64FC1);
    Mat tvec = Mat::zeros(3, 1, CV_64FC1);
    Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1);
    solvePnPRansac(points1_3d, points1, camMatrix, distCoeffs, rvec, tvec, false, 100, 0.1, 0.9999, noArray(), SOLVEPNP_EPNP);
//    Mat R;
//    Rodrigues(rvec, R);
//    transf <<                   0, -rvec.at<double>(2,0),  rvec.at<double>(1,0), tvec.at<double>(0,0),
//             rvec.at<double>(2,0),                     0, -rvec.at<double>(0,0), tvec.at<double>(1,0),
//            -rvec.at<double>(1,0),  rvec.at<double>(0,0),  0,                    tvec.at<double>(2,0),
//            0,                  0,                     0,  1;

    VectorXd initVector;
    initVector.setZero(6);
    initVector(0) = tvec.at<double>(0,0);
    initVector(1) = tvec.at<double>(1,0);
    initVector(2) = tvec.at<double>(2,0);
    initVector(3) = rvec.at<double>(0,0);
    initVector(4) = rvec.at<double>(1,0);
    initVector(5) = rvec.at<double>(2,0);

    return initVector;
}
