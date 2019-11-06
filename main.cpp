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
#include <Open3D/Registration/GlobalOptimization.h>
#include <Open3D/Registration/PoseGraph.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include "filereader.h"
#include "fastbilateral.h"
#include "util.h"
#include "initAligner.h"
#include "aligner.h"
#include "groundTruthReader.h"

using namespace open3d;
using namespace open3d::integration;
using namespace open3d::visualization;
using namespace open3d::registration;
using namespace std;
using namespace cv;
using namespace std::chrono;

bool generateMesh = false;
bool groundTruth = false;
bool isOpen3D = false;
double totalAngle = 0;
double totalTransl = 0;
double voxelDSAngle = 0;
double radius = 0.25;
double errorDepth = 0;
PoseGraph poseGraph;

int main(int argc, char *argv[]){

    int initFrame = 0;
    int finalFrame = 1000;
    int step = 1;

    if(argc > 1){
        string genMesh = argv[1];
        if(genMesh == "mesh"){
            generateMesh = true;
            initFrame = atoi(argv[2]);
            finalFrame = atoi(argv[3]);
            step = atoi(argv[4]);
            if(argc == 6){
                string open3dArg = argv[5];
                isOpen3D = open3dArg == "open3d" ? true : false;
            }            
        }
        else if(genMesh == "gt"){
            groundTruth = true;
            generateMesh = true;
            if(argc == 5){
                initFrame = atoi(argv[2]);
                finalFrame = atoi(argv[3]);
                step = atoi(argv[4]);
            }
        }
        else{
            initFrame = atoi(argv[1]);
            finalFrame = atoi(argv[2]);
            step = atoi(argv[3]);
        }
    }

    cerr << CV_VERSION << endl;

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d transf = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d actualPose = Eigen::Matrix4d::Identity();
    transf <<   1,  0,  0,  0,
                0, -1,  0,  0,
                0,  0, -1,  0,
                0,  0,  0,  1;

    Eigen::Matrix4d initCam = Eigen::Matrix4d::Identity();

    //Siemens Dataset
    PinholeCameraIntrinsic intrinsics =
            PinholeCameraIntrinsic(640, 480, 538.925221, 538.925221, 316.473843, 243.262082);
    //Kinect 360 unprojection
    //    PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 589.322232303107740, 589.849429472609130, 321.140896612950880, 235.563195335248370);
    //    PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3);
    //InHand Dataset
    //    PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 474.31524658203125, 474.31524658203125, 317.4243469238281, 244.9296875);
    //Mickey Dataset
    //    PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 525, 525, 319.5, 239.5);
    //CORBS
    //PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(640, 480, 468.60, 468.61, 318.27, 243.99);
    //Kinect v2
    //PinholeCameraIntrinsic intrinsics = PinholeCameraIntrinsic(512, 424, 363.491, 363.491, 256.496, 207.778);

    double depthScale = 1000;
    double maxDistProjection = 2;
    Aligner aligner(intrinsics);
    aligner.setDist(0.1, 2);
    aligner.depthScale = depthScale;

    //0.000125
    ScalableTSDFVolume tsdf(1.f/300, 0.02, TSDFVolumeColorType::RGB8);
    ScalableTSDFVolume tsdfParcial(1.f/depthScale, 0.015, TSDFVolumeColorType::RGB8);
    ScalableTSDFVolume tsdfFinal(1.f/depthScale, 0.015, TSDFVolumeColorType::RGB8);

//    string datasetFolder = "/Users/thiago/Datasets/3d-printed-dataset/Kinect_Leopard_Turntable/";
    string datasetFolder = "/media/thiago/BigStorage/3d-printed-dataset/Kinect_Teddy_Turntable/";

    //Output file with poses
    ofstream posesFile, diffFile, poseGraphFile;
    if(!groundTruth){
        posesFile.open(datasetFolder + "poses.txt");
        diffFile.open(datasetFolder + "difference.txt");
        poseGraphFile.open(datasetFolder + "poseGraph.txt");
    }
    //groundtruth reader
    vector<Matrix4d> posesRefined = readGTFile(datasetFolder + "poseGraph.txt");
    vector<Matrix4d> gt_poses = readGTFile(datasetFolder + "markerboard_poses.txt");

    vector<string> depthFiles, rgbFiles, omaskFiles, tmaskFiles;
    readFilenames(depthFiles, datasetFolder + "depth/");
    readFilenames(rgbFiles, datasetFolder + "rgb/");
    readFilenames(omaskFiles, datasetFolder + "omask/");
    readFilenames(tmaskFiles, datasetFolder + "tmask/");

    shared_ptr<PointCloudExtended> pcdExtended = std::make_shared<PointCloudExtended>();
    shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();

    vector< shared_ptr<PointCloud> > parcialPcds;
    vector<Matrix4d> parcialPcdsPoses;

    VectorXd lastPose;
    bool lastValid = true;
    lastPose.setZero(6);
    Matrix4d prevTransf = transf;
    Matrix3d prevRotation;
    Matrix3d rotation;

    for (int i = initFrame; i < depthFiles.size()-1; i+=step) {
        int i1 = i;
        int i2 = i1 + step;

        if(!groundTruth){
            posesFile << i1 << endl;
            posesFile << actualPose << endl;
        }

        cerr << "diff Estimated - GT" << endl;
        cerr << (actualPose - gt_poses[i]).cwiseAbs() << endl;
        Vector4d position;
        position << 0,0,0,1;
        cerr << "diff camera position Est - GT: " << endl
             << (actualPose * position - gt_poses[i] * position).cwiseAbs() << endl;
        diffFile << i << endl;
        diffFile << (actualPose * position - gt_poses[i] * position).cwiseAbs() << endl;
        cerr << "difference between depths: " << errorDepth << endl;

        string depthPath1 = datasetFolder + "depth/" + depthFiles[i1];
        string depthPath2 = datasetFolder + "depth/" + depthFiles[i2];
        string rgbPath1 = datasetFolder + "rgb/" + rgbFiles[i1];
        string rgbPath2 = datasetFolder + "rgb/" + rgbFiles[i2];
        string omaskPath1 = datasetFolder + "omask/" + omaskFiles[i1];
        string omaskPath2 = datasetFolder + "omask/" + omaskFiles[i2];
        string tmaskPath1 = datasetFolder + "tmask/" + tmaskFiles[i1];
        string tmaskPath2 = datasetFolder + "tmask/" + tmaskFiles[i2];
        cerr << rgbPath1 << endl;
        cerr << rgbPath2 << endl;
        Mat rgb1 = imread(rgbPath1);
        Mat rgb2 = imread(rgbPath2);
        Mat omask1 = imread(omaskPath1);
        Mat omask2 = imread(omaskPath2);
        Mat tmask1 = imread(tmaskPath1);
        Mat tmask2 = imread(tmaskPath2);
        Mat mask1 = omask1 + tmask1;
        Mat mask2 = omask2 + tmask2;
        removeMaskBorders(mask1, 50, 50);
        removeMaskBorders(mask2, 50, 50);
        bitwise_and(rgb1, mask1, rgb1);
        bitwise_and(rgb2, mask2, rgb2);
        Mat depth2, depth1;
        Mat index2;
        if (i == initFrame || isOpen3D){
            depth1 = imread(depthPath1, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
            depth2 = imread(depthPath2, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
            depth2 = applyMaskDepth(depth2, mask2);            
        }
        else{
            depth1 = imread(depthPath1, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

            if (!generateMesh){
                projectPointCloudExtended(*pcdExtended, maxDistProjection, transf.inverse(),
                                          intrinsics, depthScale, radius, initFrame, i1, depth2, index2);
            }
            else{                
                Mat color;
                projectMesh(tsdf.ExtractTriangleMesh(), maxDistProjection, depthScale, transf.inverse(), intrinsics, depth2, color);
                //imshow("previewModel", color);
            }
        }
        depth1 = applyMaskDepth(depth1, mask1);
        threshold(depth1, depth1, 500, 65535, THRESH_TOZERO);
        threshold(depth1, depth1, 1100, 65535, THRESH_TOZERO_INV);
        Mat gray1;
        Mat gray2;
        cvtColor(rgb1, gray1, CV_BGR2GRAY);
        cvtColor(rgb2, gray2, CV_BGR2GRAY);
        cv_extend::bilateralFilter(gray1, gray1, 5, 3);
        cv_extend::bilateralFilter(gray2, gray2, 5, 3);
        cv_extend::bilateralFilter(depth1, depth1, 5, 3);
        cv_extend::bilateralFilter(depth2, depth2, 5, 3);
        //        erode(depth1, depth1, getStructuringElement(MORPH_RECT, Size(3, 3)));
        //        erode(depth2, depth2, getStructuringElement(MORPH_RECT, Size(3, 3)));

        Mat grayOut, depthOut;
        hconcat(gray1, gray2, grayOut);
        hconcat(depth1, depth2, depthOut);
        resize(grayOut, grayOut, Size(grayOut.cols/2, grayOut.rows/2));
        resize(depthOut, depthOut, Size(depthOut.cols/2, depthOut.rows/2));
        imshow("rgb", grayOut);
        imshow("depth", depthOut * 10);

        //Visualization
        shared_ptr<RGBDImage> rgbdImage;
        Mat depthTmp1;
        depth1.convertTo(depthTmp1, CV_64FC1, 1.0/depthScale);

        Mat normalMap1 = getNormalMapFromDepth(depthTmp1, intrinsics, 0, depthScale);
        Mat maskNormals = getNormalWeight(normalMap1, depthTmp1, intrinsics, false);
        imshow("normals", maskNormals);

        Mat depthTmp2;
        depth2.convertTo(depthTmp2, CV_64FC1, 1.0/depthScale);
        Mat normalMap2 = getNormalMapFromDepth(depthTmp2, intrinsics, 0, depthScale);
        Mat model = getNormalWeight(normalMap2, depthTmp2, intrinsics);
        imshow("previewNormals", model);


        Image rgb = CreateRGBImageFromMat(&rgb1);        
        Image depth = CreateDepthImageFromMat(&depth1, NULL);
        rgbdImage = CreateRGBDImageFromColorAndDepth(rgb, depth, depthScale, 1.5, false);

        shared_ptr<PointCloud> actualPointCloud = CreatePointCloudFromRGBDImage(*rgbdImage, intrinsics, initCam);

        //Images to Open3D odometry
        shared_ptr<Image> rgbImage1 = CreateImageFromFile(rgbPath1);
        Image depthImage1 = CreateDepthImageFromMat(&depth1, NULL);
        shared_ptr<RGBDImage> rgbdImage1 = CreateRGBDImageFromColorAndDepth(*rgbImage1, depthImage1, depthScale, 1.5, true);
        shared_ptr<Image> rgbImage2 = CreateImageFromFile(rgbPath2);
        Image depthImage2 = CreateDepthImageFromMat(&depth2, NULL);
        shared_ptr<RGBDImage> rgbdImage2 = CreateRGBDImageFromColorAndDepth(*rgbImage2, depthImage2, depthScale, 1.5, true);

        Vector3d prevTranslation(prevTransf(0,3), prevTransf(1,3), prevTransf(2,3));

        prevRotation << prevTransf(0,0) , prevTransf(0,1), prevTransf(0,2),
                prevTransf(1,0) , prevTransf(1,1), prevTransf(1,2),
                prevTransf(2,0) , prevTransf(2,1), prevTransf(2,2);

        Vector3d translation(transf(0,3), transf(1,3), transf(2,3));

        rotation << transf(0,0) , transf(0,1), transf(0,2),
                transf(1,0) , transf(1,1), transf(1,2),
                transf(2,0) , transf(2,1), transf(2,2);

        double transl = (prevTranslation - translation).norm();
        double angle = acos(((prevRotation.transpose() * rotation).trace()-1)/2);
        angle = isnan(angle) ? 0 : angle;
        angle *= 180/M_PI;

        totalAngle += angle;
        totalTransl += transl;
        voxelDSAngle += angle;
        cerr << "angle: " << totalAngle << "transl:" << totalTransl << endl;

        if (!generateMesh){
            actualPointCloud->Transform(transf);
            auto start = high_resolution_clock::now();
            merge(pcdExtended, actualPointCloud, prevTransf, transf,
                  intrinsics, depthScale, totalAngle, maskNormals);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            cerr << "Merge time: " << duration.count()/1000000.f << endl;
        }

        VectorXd pose;
        if(!groundTruth){            
            //************ INITIAL ALIGNMENT ************//
            double focal = intrinsics.GetFocalLength().first;
            double cx = intrinsics.GetPrincipalPoint().first;
            double cy = intrinsics.GetPrincipalPoint().second;

            pose = coarseRegistration(depthTmp1, rgb1, depthTmp2, rgb2, focal, cx, cy);
            aligner.setInitialPoseVector(pose);
            cerr << TransformVector6dToMatrix4d(pose).inverse();
            cerr << "--------" << endl;
            //************ INITIAL ALIGNMENT ************//

            //************ REFINE ALIGNMENT ************//
            auto start = high_resolution_clock::now();

            std::pair<VectorXd, double> result;
            if(isOpen3D){
                result = aligner.getPoseTransformOpen3d(rgbdImage1, rgbdImage2,
                                   TransformVector6dToMatrix4d(pose).inverse());
            }
            else{
                result = aligner.getPoseTransform(gray1, depth1, gray2, depth2, false);
            }
            pose = result.first;
            errorDepth = result.second;
            auto stop = high_resolution_clock::now();
            cerr << "Obtida transf:\n" << TransformVector6dToMatrix4d(pose).inverse() << endl;
            //************ REFINE ALIGNMENT ************//
        }
        //GroundTruth tests
        else{
            prevTransf = transf;
            transf = posesRefined[i+1];
            actualPose = transf;
        }

        if (generateMesh){
            tsdf.Integrate(*rgbdImage, intrinsics, transf.inverse());
        }

        if(!groundTruth){
            t = TransformVector6dToMatrix4d(pose).inverse();            
            prevTransf = transf;
            transf = transf * t;
            actualPose = actualPose * t;
        }

        //auto duration = duration_cast<microseconds>(stop - start);
        //cerr << "Time elapsed: " << duration.count()/1000000.f << endl;
        //cerr << "Frame : " << i << endl;
        //cerr << "TRANSLATION: " << totalTransl << endl;
        //cerr << "ANGLE: " << totalAngle << endl;

        char key = waitKey(200);
        if(i == finalFrame || key == 'z'){
            Visualizer vis;
            vis.CreateVisualizerWindow("Visualization", 800, 600);
            vis.GetRenderOption().point_size_ = 2;
            vis.GetRenderOption().background_color_ = Eigen::Vector3d(0.2,0.1,0.6);
            vis.GetRenderOption().show_coordinate_frame_ = false;
            vis.GetRenderOption().mesh_show_back_face_ = true;
            vis.GetViewControl().SetViewMatrices(initCam);
            if(generateMesh){                
                mesh = tsdf.ExtractTriangleMesh();                
                vis.AddGeometry({mesh});                
            }
            else{
                vis.AddGeometry({pcdExtended});
            }
            vis.Run();
            vis.UpdateGeometry();
            vis.UpdateRender();
            vis.PollEvents();
            vis.Close();
            vis.DestroyVisualizerWindow();
        }
        if(key == 27){
            break;
        }

        if(!groundTruth){
            //************ POSE GRAPH ************//
            //Final Frame
            if(i + step >= depthFiles.size() - 1){
                poseGraph.nodes_.push_back(PoseGraphNode(actualPose.inverse()));
                poseGraph.edges_.push_back(PoseGraphEdge(initFrame, i, t, Eigen::Matrix6d::Identity(), false));

                GlobalOptimizationConvergenceCriteria criteria;
                GlobalOptimizationOption option;
                option.edge_prune_threshold_ = 0.25;
                option.reference_node_ = initFrame;
                GlobalOptimizationLevenbergMarquardt optimization_method;
                GlobalOptimization(poseGraph, optimization_method, criteria, option);

                cerr << "Writing pose Graph results:" << endl;
                poseGraphFile << "0" << endl;
                poseGraphFile << Matrix4d::Identity() << endl;
                for (int i = 0; i < poseGraph.nodes_.size(); ++i) {
                    poseGraphFile << i+1 << endl;
                    poseGraphFile << poseGraph.nodes_[i].pose_ << endl;
                }
            }
            else{
                poseGraph.nodes_.push_back(PoseGraphNode(actualPose.inverse()));
                poseGraph.edges_.push_back(PoseGraphEdge(i+1, i, t, Eigen::Matrix6d::Identity(), false));
            }
            //************ POSE GRAPH ************//
        }
    }

    posesFile << depthFiles.size()-1 << endl;
    posesFile << actualPose << endl;
    posesFile.close();
    diffFile.close();
    if(groundTruth){
        poseGraphFile.close();
    }

    if (generateMesh){
        mesh->ComputeTriangleNormals();
        mesh = tsdf.ExtractTriangleMesh();
        WriteTriangleMeshToPLY(datasetFolder + "mesh.ply", *mesh);
    }
    else{
        WritePointCloudToPLY(datasetFolder + "pointcloud.ply", *pcdExtended);
    }
    return 0;
}
