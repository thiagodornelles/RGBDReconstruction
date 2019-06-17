#include <iostream>
#include <memory>
#include <thread>
#include <string>
#include <chrono>

#include <Open3D/Open3D.h>
#include <Open3D/Utility/Helper.h>
#include <Open3D/IO/ClassIO/ImageIO.h>
#include <Open3D/Visualization/Visualizer/Visualizer.h>
#include <Open3D/Integration/ScalableTSDFVolume.h>
#include <Open3D/Geometry/Geometry.h>
#include <Open3D/Geometry/PointCloud.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include "filereader.h"
#include "util.h"
#include "aligner.h"

using namespace open3d;
using namespace open3d::integration;
using namespace open3d::visualization;
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
    //PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 589.322232303107740, 589.849429472609130, 321.140896612950880, 235.563195335248370);
    PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3);
    //Mickey Dataset
    //PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 525, 525, 319.5, 239.5);
    //CORBS
    //PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 468.60, 468.61, 318.27, 243.99);
    //Kinect v2
    //PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(512, 424, 363.491, 363.491, 256.496, 207.778);

    Aligner aligner(intrinsics);
    aligner.setDist(0.1, 0.3);
    double maxDistProjection = 0.3;

//    string datasetFolder = "/media/thiago/BigStorage/gabrielaGAP/";
//    string datasetFolder = "/media/thiago/BigStorage/gabrielaFar/";
//    string datasetFolder = "/Users/thiago/Datasets/gabrielaXYZ/";
    string datasetFolder = "/Users/thiago/Datasets/kinectdata/";
    //    string datasetFolder = "/media/thiago/BigStorage/Datasets/rgbd_dataset_freiburg1_plant/";
    //    string datasetFolder = "/media/thiago/BigStorage/Datasets/mickey/";
    //    string datasetFolder = "/media/thiago/BigStorage/Datasets/mickey/";
    //    string datasetFolder = "/Users/thiago/Datasets/car/";
    //    string datasetFolder = "/Users/thiago/Datasets/desk/";
    //    string datasetFolder = "/media/thiago/BigStorage/Datasets/sculp/";
    vector<string> depthFiles, rgbFiles;
    readFilenames(depthFiles, datasetFolder + "depth/");
    readFilenames(rgbFiles, datasetFolder + "rgb/");

    int initFrame = 0;
    int finalFrame = 250;

    shared_ptr<PointCloudExtended> pointCloud = std::make_shared<PointCloudExtended>();
    shared_ptr<PointCloud> pcdVisualizer = std::make_shared<PointCloud>();
    shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();

    int step = 1;
    VectorXd lastPose;
    lastPose.setZero(6);
    for (int i = initFrame; i < depthFiles.size()-1; i+=step) {

        Visualizer vis;
        vis.CreateVisualizerWindow("Visualization", 640, 480);
        vis.GetRenderOption().point_size_ = 1;
        vis.GetViewControl().SetViewMatrices(initCam);

        if (generateMesh){
            vis.AddGeometry({mesh});
        }
        else{
            vis.AddGeometry({pcdVisualizer});
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
                projectPointCloudExtended(*pointCloud, maxDistProjection, transf.inverse(),
                                  intrinsics, depth2, index2);
            }
            else{
                projectPointCloud(*tsdf.ExtractPointCloud(), maxDistProjection,
                                  transf.inverse(), intrinsics, depth2, index2);
            }
            imshow("projection", depth2 * 10);
        }
        Mat gray1;
        Mat gray2;
        cvtColor(rgb1, gray1, CV_BGR2GRAY);
        cvtColor(rgb2, gray2, CV_BGR2GRAY);

//        threshold(depth1, depth1, 1000, 65535, THRESH_TOZERO_INV);
//        erode(depth1, depth1, getStructuringElement(MORPH_CROSS, Size(5, 5)));
//        erode(depth2, depth2, getStructuringElement(MORPH_CROSS, Size(5, 5)));

        Mat grayOut, depthOut;
        hconcat(gray1, gray2, grayOut);
        hconcat(depth1, depth2, depthOut);
        resize(grayOut, grayOut, Size(grayOut.cols/2, grayOut.rows/2));
        resize(depthOut, depthOut, Size(depthOut.cols/2, depthOut.rows/2));
        imshow("rgb", grayOut);
        imshow("depth", depthOut * 10);

        //Visualization
        shared_ptr<RGBDImage> rgbdImage;
        Mat depthTmp;
        depth1.convertTo(depthTmp, CV_64FC1, 0.0002);
        Mat normalMap1 = getNormalMapFromDepth(depthTmp, intrinsics, 0);
        Mat maskNormals = getNormalWeight(normalMap1, depthTmp, intrinsics, false, 1.3);
        imshow("normals", maskNormals);
        rgbdImage = ReadRGBDImage(rgbPath1.c_str(), depthPath1.c_str(),
                                  intrinsics, maxDistProjection, &maskNormals);
//        rgbdImage = ReadRGBDImage(rgbPath1.c_str(), depthPath1.c_str(),
//                                  intrinsics, maxDistProjection);

        Mat depthTmp2;
        depth2.convertTo(depthTmp2, CV_64FC1, 0.0002);
        Mat normalMap2 = getNormalMapFromDepth(depthTmp2, intrinsics, 0);
        Mat model = getNormalWeight(normalMap2, depthTmp2, intrinsics);
        imshow("model", model);
        shared_ptr<PointCloud> pcd = CreatePointCloudFromRGBDImage(*rgbdImage, intrinsics, initCam);

        transf = transf * aligner.getMatrixRtFromPose6D(aligner.getPose6D()).inverse();

        if (!generateMesh){
            pcd->Transform(transf);
            auto start = high_resolution_clock::now();
            merge(pointCloud, pcd, transf.inverse(), intrinsics, normalMap1, maskNormals);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            cerr << "Merge time: " << duration.count()/1000000.f << endl;            
        }

        //************ ALIGNMENT ************//
        auto start = high_resolution_clock::now();
        VectorXd pose = aligner.getPoseTransform(gray1, depth1, gray2, depth2, false);
        auto stop = high_resolution_clock::now();                
        cerr << "Obtida transf:\n" << aligner.getMatrixRtFromPose6D(pose) << endl;

        if (generateMesh){
            tsdf.Integrate(*rgbdImage, intrinsics, transf.inverse());
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
            pcdVisualizer = std::get<0>(geometry::RemoveStatisticalOutliers(*pointCloud, 10, 0.5));
            vis.AddGeometry({pcdVisualizer});
            vis.Run();
            vis.UpdateGeometry();
            vis.UpdateRender();
            vis.PollEvents();
            vis.DestroyVisualizerWindow();
            break;

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
