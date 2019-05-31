#include <iostream>
#include <memory>
#include <thread>
#include <string>
#include <chrono>

#include <Core/Core.h>
#include <Core/Utility/Helper.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Integration/ScalableTSDFVolume.h>
#include <Geometry/Geometry.h>
#include <Geometry/PointCloud.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include "filereader.h"
#include "util.h"
#include "aligner.h"

using namespace open3d;
using namespace std;
using namespace cv;
using namespace std::chrono;

bool generateMesh = false;

int main(int argc, char *argv[]){

    cerr << CV_VERSION << endl;

    //    SetVerbosityLevel(VerbosityLevel::VerboseAlways);
    ScalableTSDFVolume tsdf(0.0002, 0.001, TSDFVolumeColorType::RGB8);
    Eigen::Matrix4d transf = Eigen::Matrix4d::Identity();
    transf <<  1,  0,  0,  0,
            0, -1,  0,  0,
            0,  0, -1,  0,
            0,  0,  0,  1;

    Eigen::Matrix4d initCam = Eigen::Matrix4d::Identity();

    //Kinect 360 unprojection
    PinholeCameraIntrinsic cameraIntrinsics = PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3);
    //Mickey Dataset
    //PinholeCameraIntrinsic cameraIntrinsics = PinholeCameraIntrinsic(640, 480, 525, 525, 319.5, 239.5);
    //CORBS
    //        PinholeCameraIntrinsic cameraIntrinsics = PinholeCameraIntrinsic(640, 480, 468.60, 468.61, 318.27, 243.99);
    //Kinect v2
    //    PinholeCameraIntrinsic cameraIntrinsics = PinholeCameraIntrinsic(512, 424, 363.491, 363.491, 256.496, 207.778);

    Aligner aligner(cameraIntrinsics);
    aligner.setDist(0.1, 0.3);
    double maxDistProjection = 0.3;

    double fovX = 2 * atan(640 / (2 * cameraIntrinsics.GetFocalLength().first)) * 180.0 / CV_PI;
    double fovY = 2 * atan(480 / (2 * cameraIntrinsics.GetFocalLength().second)) * 180.0 / CV_PI;
    cerr << "FOVy " << fovY << endl;
    cerr << "FOVx " << fovX << endl;
        string datasetFolder = "/media/thiago/BigStorage/kinectdata/";
    //    string datasetFolder = "/Users/thiago/Datasets/rgbd_dataset_freiburg1_plant/";
    //    string datasetFolder = "/media/thiago/BigStorage/Datasets/mickey/";
    //    string datasetFolder = "/media/thiago/BigStorage/Datasets/mickey/";
    //    string datasetFolder = "/Users/thiago/Datasets/car/";
    //    string datasetFolder = "/Users/thiago/Datasets/desk/";
//    string datasetFolder = "/media/thiago/BigStorage/Datasets/sculp/";
    vector<string> depthFiles, rgbFiles;
    readFilenames(depthFiles, datasetFolder + "depth/");
    readFilenames(rgbFiles, datasetFolder + "rgb/");

    int initFrame = 30;
    int finalFrame = 6000;

    shared_ptr<PointCloudExtended> pointCloud = std::make_shared<PointCloudExtended>();
    shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();

    int step = 1;
    for (int i = initFrame; i < depthFiles.size()-1; i+=step) {

        Visualizer vis;
        vis.CreateVisualizerWindow("Visualization", 640, 480);
        vis.GetRenderOption().point_size_ = 1;
        vis.GetViewControl().SetViewMatrices(initCam);

        if (generateMesh){
            vis.AddGeometry({mesh});
        }
        else{
            vis.AddGeometry({pointCloud});
        }
        vis.UpdateGeometry();
        vis.UpdateRender();
        vis.PollEvents();

        int i1 = i;
        int i2 = i1 + step;
        string depthPath1 = datasetFolder + "depth/" + depthFiles[i1];
        string rgbPath1 = datasetFolder + "rgb/" + rgbFiles[i1];
        string depthPath2 = datasetFolder + "depth/" + depthFiles[i2];
        string rgbPath2 = datasetFolder + "rgb/" + rgbFiles[i2];
        Mat rgb1 = imread(rgbPath1);
        Mat rgb2 = imread(rgbPath2);
        Mat depth2, depth1;
        Mat index2;
        if (i == initFrame){
            depth2 = imread(depthPath2, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
            depth1 = imread(depthPath1, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
        }
        else{
            depth1 = imread(depthPath1, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

            if (!generateMesh){
                projectPointCloud(*pointCloud, maxDistProjection, transf.inverse(),
                                  cameraIntrinsics, depth2, index2);
            }
            else{
                projectPointCloud(*tsdf.ExtractPointCloud(), maxDistProjection,
                                  transf.inverse(), cameraIntrinsics, depth2, index2);
            }
            imshow("projection", depth2 * 10);
        }
        Mat gray1;
        Mat gray2;
        cvtColor(rgb1, gray1, CV_BGR2GRAY);
        cvtColor(rgb2, gray2, CV_BGR2GRAY);

        threshold(depth1, depth1, 2000, 65535, THRESH_TOZERO_INV);
//        erode(depth1, depth1, getStructuringElement(MORPH_RECT, Size(5, 5)));
//        erode(depth2, depth2, getStructuringElement(MORPH_RECT, Size(5, 5)));

        Mat grayOut, depthOut;
        hconcat(gray1, gray2, grayOut);
        hconcat(depth1, depth2, depthOut);
        resize(grayOut, grayOut, Size(grayOut.cols/2, grayOut.rows/2));
        resize(depthOut, depthOut, Size(depthOut.cols/2, depthOut.rows/2));
        imshow("rgb", grayOut);
        imshow("depth", depthOut * 10);

        //Visualization
        shared_ptr<RGBDImage> rgbdImage;
        //Mat maskNormals = getNormalMap(depth1, true);
        Mat depthTmp;
        depth1.convertTo(depthTmp, CV_64FC1, 0.0002);
        Mat maskNormals = getMaskOfNormalsFromDepth(depthTmp, cameraIntrinsics, 0, false);
        imshow("maskNormals", maskNormals);

        rgbdImage = ReadRGBDImage(rgbPath1.c_str(), depthPath1.c_str(),
                                  cameraIntrinsics, maxDistProjection, &maskNormals);
        //        rgbdImage = ReadRGBDImage(rgbPath1.c_str(), depthPath1.c_str(),
        //                                  cameraIntrinsics, maxDistProjection);

        shared_ptr<PointCloud> pcd = CreatePointCloudFromRGBDImage(*rgbdImage, cameraIntrinsics, initCam);
        transf = transf * aligner.getMatrixRtFromPose6D(aligner.getPose6D()).inverse();
        if (!generateMesh){
            pcd->Transform(transf);
            merge(pointCloud, pcd, transf.inverse(), cameraIntrinsics);
        }

        //************ ALIGNMENT ************//
        auto start = high_resolution_clock::now();
        aligner.getPoseTransform(gray1, depth1, gray2, depth2, true);
        auto stop = high_resolution_clock::now();
        cerr << "Obtida transf:\n" << aligner.getMatrixRtFromPose6D(aligner.getPose6D()) << endl;

        if (generateMesh){
            tsdf.Integrate(*rgbdImage, cameraIntrinsics, transf.inverse());
            mesh = tsdf.ExtractTriangleMesh();
            mesh->ComputeTriangleNormals();
        }

        auto duration = duration_cast<microseconds>(stop - start);
        cerr << "Time elapsed: " << duration.count()/1000000.f << endl;
        cerr << "Frame : " << i << endl;

        char key = waitKey(100);
        if(key == '1') imwrite("teste.png", depthOut);
        if(i == finalFrame || key == 'z'){
            removeUnstablePoints(pointCloud);
            vis.Run();
            vis.UpdateGeometry();
            vis.UpdateRender();
            vis.PollEvents();
            vis.DestroyVisualizerWindow();

            do {
                key = waitKey(100);
                if(key == 'o')
                    break;
                vis.UpdateGeometry();
            }
            while(key == 's');
        }
        if(key == 27){
            break;
        }
    }
    if (generateMesh)
        WriteTriangleMeshToPLY("result.ply", *mesh);
    else
        WritePointCloudToPLY("result.ply", *pointCloud);
    return 0;
}
