#include <vector>
#include <string>
#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

//Groundtruth 3d-printed-dataset
vector<Matrix4d> readGTFile(string filePath){

    vector<Matrix4d> poses;

    ifstream inFile;
    inFile.open(filePath);
    if (!inFile) {
        cerr << "Unable to open file " << filePath;
        exit(1);   // call system to stop
    }
    double value = 0;
    while (inFile >> value) {
//        cerr << value << endl;

        Eigen::Matrix4d pose;
        inFile >> pose(0,0);
        inFile >> pose(0,1);
        inFile >> pose(0,2);
        inFile >> pose(0,3);
        inFile >> pose(1,0);
        inFile >> pose(1,1);
        inFile >> pose(1,2);
        inFile >> pose(1,3);
        inFile >> pose(2,0);
        inFile >> pose(2,1);
        inFile >> pose(2,2);
        inFile >> pose(2,3);
        inFile >> pose(3,0);
        inFile >> pose(3,1);
        inFile >> pose(3,2);
        inFile >> pose(3,3);

//        cerr << pose << endl;
        poses.push_back(pose);
    }
    inFile.close();
    return poses;
}
