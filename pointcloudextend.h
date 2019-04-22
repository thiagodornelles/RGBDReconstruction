#include <Core/Core.h>
#include <Geometry/Geometry.h>

using namespace open3d;

class PointCloudExtended : public PointCloud
{
public:
    PointCloudExtended() : PointCloud() {};
    ~PointCloudExtended() override {};

public:
    std::vector<int> hitCounter;
};
