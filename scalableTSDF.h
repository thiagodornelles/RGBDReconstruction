#include <Open3D/Integration/ScalableTSDFVolume.h>

using namespace open3d;
using namespace open3d::integration;

class ScalableTSDF : public ScalableTSDFVolume {

    void ScalableTSDF::Integrate(
            const geometry::RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic) {
        if ((image.depth_.num_of_channels_ != 1) ||
                (image.depth_.bytes_per_channel_ != 4) ||
                (image.depth_.width_ != intrinsic.width_) ||
                (image.depth_.height_ != intrinsic.height_) ||
                (color_type_ == TSDFVolumeColorType::RGB8 &&
                 image.color_.num_of_channels_ != 3) ||
                (color_type_ == TSDFVolumeColorType::RGB8 &&
                 image.color_.bytes_per_channel_ != 1) ||
                (color_type_ == TSDFVolumeColorType::Gray32 &&
                 image.color_.num_of_channels_ != 1) ||
                (color_type_ == TSDFVolumeColorType::Gray32 &&
                 image.color_.bytes_per_channel_ != 4) ||
                (color_type_ != TSDFVolumeColorType::None &&
                 image.color_.width_ != intrinsic.width_) ||
                (color_type_ != TSDFVolumeColorType::None &&
                 image.color_.height_ != intrinsic.height_)) {
            utility::LogWarning(
                        "[ScalableTSDFVolume::Integrate] Unsupported image format.\n");
            return;
        }
        auto depth2cameradistance =
                geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
                    intrinsic);
        auto pointcloud = geometry::PointCloud::CreateFromDepthImage(
                    image.depth_, intrinsic, extrinsic, 1000.0, 1000.0,
                    depth_sampling_stride_);
        std::unordered_set<Eigen::Vector3i,
                utility::hash_eigen::hash<Eigen::Vector3i>>
                touched_volume_units_;
        for (const auto &point : pointcloud->points_) {
            auto min_bound = LocateVolumeUnit(
                        point - Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
            auto max_bound = LocateVolumeUnit(
                        point + Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
            for (auto x = min_bound(0); x <= max_bound(0); x++) {
                for (auto y = min_bound(1); y <= max_bound(1); y++) {
                    for (auto z = min_bound(2); z <= max_bound(2); z++) {
                        auto loc = Eigen::Vector3i(x, y, z);
                        if (touched_volume_units_.find(loc) == touched_volume_units_.end()) {
                            touched_volume_units_.insert(loc);
                            auto volume = OpenVolumeUnit(Eigen::Vector3i(x, y, z));
                            volume->IntegrateWithDepthToCameraDistanceMultiplier(
                                        image, intrinsic, extrinsic,
                                        *depth2cameradistance);
                        }
                    }
                }
            }
        }
    }
};
