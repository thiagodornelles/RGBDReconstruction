#include <Open3D/Open3D.h>
#include <Open3D/Geometry/PointCloud.h>

using namespace open3d;
using namespace geometry;
using namespace std;

class PointCloudExtended : public PointCloud
{
public:
    PointCloudExtended() : PointCloud() {        
    }
    ~PointCloudExtended() override {}
    void reserveAndAssign(int size){
        hitCounter_.reserve(size);
        hitCounter_.assign(size, 0);
        frameCounter_.reserve(size);
        frameCounter_.assign(size, 0);
    }

public:
    vector<int> hitCounter_;
    vector<int> frameCounter_;
};
