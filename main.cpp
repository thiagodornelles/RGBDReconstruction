#include <iostream>
#include <memory>
#include <thread>

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

int main(int argc, char *argv[]){

    //    SetVerbosityLevel(VerbosityLevel::VerboseAlways);

    ScalableTSDFVolume tsdf(0.001, 0.01, TSDFVolumeColorType::RGB8);
    Eigen::Matrix4d transf = Eigen::Matrix4d::Identity();
    transf <<  1,  0,  0,  0,
               0, -1,  0,  0,
               0,  0, -1,  0,
               0,  0,  0,  1;

    Eigen::Matrix4d initCam = Eigen::Matrix4d::Identity();

    //Kinect 360 unprojection
    PinholeCameraIntrinsic cameraIntrisecs = PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3);

    Aligner aligner(cameraIntrisecs);

    string datasetFolder = "/Users/thiago/RGBDOdometry/build/rgbd_dataset_freiburg1_plant/";
    vector<string> depthFiles, rgbFiles;
    readFilenames(depthFiles, datasetFolder + "depth/");
    readFilenames(rgbFiles, datasetFolder + "rgb/");

    Visualizer imgVis;
    imgVis.CreateVisualizerWindow("Image", 640, 480);
    Visualizer vis;
    vis.CreateVisualizerWindow("Visualization", 640, 480);
    vis.GetRenderOption().point_size_ = 1;
    vis.GetViewControl().SetViewMatrices(initCam);

    for (int i = 60; i < depthFiles.size()-1; i++) {

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

        imshow("rgb1", rgb1);
        imshow("rgb2", rgb2);
        imshow("depth1", depth1);
        imshow("depth2", depth2);

        //Visualization
        shared_ptr<RGBDImage> rgbdImage;
        rgbdImage = ReadRGBDImage(rgbPath1.c_str(), depthPath1.c_str(), cameraIntrisecs);

        shared_ptr<PointCloud> pcd = CreatePointCloudFromRGBDImage(*rgbdImage, cameraIntrisecs, initCam);
        pcd->Transform(transf);
//        pcd = VoxelDownSample(*pcd, 0.005);
//        pcd = get<0>(RemoveRadiusOutliers(*pcd, 1, 0.01));

        aligner.getPoseTransform(gray2, depth2, gray1, depth1);
        transf = transf * aligner.getMatrixRtFromPose6D(aligner.getPose6D());

//        tsdf.Integrate(*rgbdImage, cameraIntrisecs, transf.inverse());
//        shared_ptr<TriangleMesh> pcd = tsdf.ExtractTriangleMesh();
//        pcd->ComputeTriangleNormals();

        vis.AddGeometry({pcd});        
        vis.UpdateGeometry();
        vis.UpdateRender();
        vis.PollEvents();        

//        shared_ptr<Image> depth = vis.CaptureDepthFloatBuffer();
//        imgVis.AddGeometry({depth});
//        imgVis.UpdateGeometry();
//        imgVis.UpdateRender();
//        imgVis.PollEvents();
        cerr << "Frame : " << i << endl;
        char key = waitKey(0);
        if(i == 65 || key == 'p'){
            vis.Run();
        }
        if(key == 27){
            break;
        }        
    }
    vis.DestroyVisualizerWindow();

    return 0;
}
