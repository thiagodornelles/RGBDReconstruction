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

#include <opencv2/highgui/highgui.hpp>

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

    ScalableTSDFVolume tsdf(0.001, 0.002, TSDFVolumeColorType::RGB8);
    Eigen::Matrix4d transf = Eigen::Matrix4d::Identity();
    transf <<  1,  0,  0,  0,
            0, -1,  0,  0,
            0,  0, -1,  0,
            0,  0,  0,  1;

    Eigen::Matrix4d initCam = Eigen::Matrix4d::Identity();

    //Kinect 360 unprojection
//    PinholeCameraIntrinsic cameraIntrisecs = PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3);
    //Mickey Dataset
    PinholeCameraIntrinsic cameraIntrisecs = PinholeCameraIntrinsic(640, 480, 525, 525, 319.5, 239.5);
    //CORBS
//    PinholeCameraIntrinsic cameraIntrisecs = PinholeCameraIntrinsic(640, 480, 468.60, 468.61, 318.27, 243.99);

    Aligner aligner(cameraIntrisecs);
    aligner.setDist(0.1, 0.3);

//    string datasetFolder = "/Users/thiago/Datasets/rgbd_dataset_freiburg1_plant/";
//    string datasetFolder = "/Users/thiago/Datasets/mickey/";
//    string datasetFolder = "/Users/thiago/Datasets/car/";
//    string datasetFolder = "/Users/thiago/Datasets/desk/";
    string datasetFolder = "/Users/thiago/Datasets/sculp/";
    vector<string> depthFiles, rgbFiles;
    readFilenames(depthFiles, datasetFolder + "depth/");
    readFilenames(rgbFiles, datasetFolder + "rgb/");

    int initFrame = 100;
    int finalFrame = 6000;

    shared_ptr<PointCloud> pointCloud = std::make_shared<PointCloud>();
    shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();

    for (int i = initFrame; i < depthFiles.size()-1; i++) {

//        Visualizer visImg;
//        visImg.CreateVisualizerWindow("Depth buffer", 640, 480);

        Visualizer vis;
        vis.CreateVisualizerWindow("Visualization", 640, 480);
        vis.GetRenderOption().point_size_ = 1;
        vis.GetViewControl().SetViewMatrices(initCam);

        if (generateMesh)
            vis.AddGeometry({mesh});
        else
            vis.AddGeometry({pointCloud});
        vis.UpdateGeometry();
        vis.UpdateRender();
        vis.PollEvents();

//        visImg.AddGeometry({vis.CaptureDepthFloatBuffer(true)});
//        visImg.UpdateGeometry();
//        visImg.UpdateRender();
//        visImg.PollEvents();

        int i1 = i;
        int i2 = i + 1;
        string depthPath1 = datasetFolder + "depth/" + depthFiles[i];
        string rgbPath1 = datasetFolder + "rgb/" + rgbFiles[i];
        string depthPath2 = datasetFolder + "depth/" + depthFiles[i+1];
        string rgbPath2 = datasetFolder + "rgb/" + rgbFiles[i+1];
        Mat depth1 = imread(depthPath1, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
        Mat rgb1 = imread(rgbPath1);
        Mat depth2 = imread(depthPath2, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
        Mat rgb2 = imread(rgbPath2);

        Mat gray1;
        Mat gray2;
        cvtColor(rgb1, gray1, CV_BGR2GRAY);
        cvtColor(rgb2, gray2, CV_BGR2GRAY);

//        medianBlur(depth1, depth1, 7);
//        medianBlur(depth2, depth2, 7);
        Mat grayOut, depthOut;
//        putText(gray1, to_string(i1), Point(5,30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), 2);
//        putText(gray2, to_string(i+1), Point(5,30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), 2);        
        hconcat(gray1, gray2, grayOut);
        hconcat(depth1, depth2, depthOut);
        resize(grayOut, grayOut, Size(grayOut.cols/2, grayOut.rows/2));
        resize(depthOut, depthOut, Size(depthOut.cols/2, depthOut.rows/2));
        imshow("rgb", grayOut);
        imshow("depth", depthOut);

//        Mat laplacImg;
//        Laplacian(gray2, laplacImg, CV_64F);
//        Scalar mean, stdDev;
//        meanStdDev(laplacImg, mean, stdDev);
//        cerr << "###### Variance #####" << endl;
//        cerr << "Variance: " << sqrt(stdDev[0]) << endl;

//        if (sqrt(stdDev[0]) < 3.2){
//            i++;
//            string depthPath2 = datasetFolder + "depth/" + depthFiles[i+1];
//            string rgbPath2 = datasetFolder + "rgb/" + rgbFiles[i+1];
//            depth2 = imread(depthPath2, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
//            rgb2 = imread(rgbPath2);
//        }

//        imshow("rgb2", gray2);
//        imshow("depth1", depth2);
//        imshow("depth1", depth1);

        //Visualization
        shared_ptr<RGBDImage> rgbdImage;
        Mat tmpDepth;
        depth1.convertTo(tmpDepth, CV_64FC1, 0.0002);
        Mat maskNormals = getMaskOfNormalsFromDepth(tmpDepth, cameraIntrisecs, 0);

        rgbdImage = ReadRGBDImage(rgbPath1.c_str(), depthPath1.c_str(),
                                  cameraIntrisecs, aligner.maxDist, &maskNormals);
//        rgbdImage = ReadRGBDImage(rgbPath1.c_str(), depthPath1.c_str(),
//                                  cameraIntrisecs, aligner.maxDist);

        shared_ptr<PointCloud> pcd = CreatePointCloudFromRGBDImage(*rgbdImage, cameraIntrisecs, initCam);
        if (!generateMesh){
            pcd->Transform(transf);
            *pointCloud = *pointCloud + *pcd;
            pointCloud = VoxelDownSample(*pointCloud, 0.0004);
    //        pointCloud = get<0>(RemoveRadiusOutliers(*pointCloud, 1, 0.002));
        }


        auto start = high_resolution_clock::now();
        aligner.getPoseTransform(gray1, depth1, gray2, depth2, true);
        auto stop = high_resolution_clock::now();
        cerr << "Obtida transf:\n" << aligner.getMatrixRtFromPose6D(aligner.getPose6D()) << endl;
        transf = transf * aligner.getMatrixRtFromPose6D(aligner.getPose6D()).inverse();

        if (generateMesh){
            tsdf.Integrate(*rgbdImage, cameraIntrisecs, transf.inverse());
            mesh = tsdf.ExtractTriangleMesh();
            mesh->ComputeTriangleNormals();
        }

        auto duration = duration_cast<microseconds>(stop - start);
        cerr << "Time elapsed: " << duration.count()/1000000.f << endl;
        cerr << "Frame : " << i << endl;

        char key = waitKey(100);
        if(i == finalFrame || key == 's'){

            vis.Run();
            vis.PollEvents();
            vis.DestroyVisualizerWindow();

//            visImg.Run();
//            visImg.PollEvents();
//            visImg.DestroyVisualizerWindow();

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
