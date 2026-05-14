#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace gmr {

    struct IkTaskEntry {
        std::string robotBodyName;
        std::string humanBodyName;
        double posWeight             = 0.0;
        double rotWeight             = 0.0;
        Eigen::Vector3d posOffset    = Eigen::Vector3d::Zero();
        Eigen::Quaterniond rotOffset = Eigen::Quaterniond::Identity();
    };

    /// Mirrors Python `ik_config["collision_avoidance"]` used with `mink.CollisionAvoidanceLimit`.
    struct CollisionAvoidanceConfig {
        bool enabled = false;
        /// Minimum separation (m). Same as mink `minimum_distance_from_collisions`.
        double minDistance = 0.005;
        /// Outside this distance (m) the inequality is inactive. Same as mink `collision_detection_distance`.
        double detectionDistance = 0.15;
        /// Same as mink `gain` (default 0.85 in mink if omitted in JSON).
        double gain            = 0.85;
        double boundRelaxation = 0.0;
        /// Each entry is two geom groups; each group is a list of MuJoCo geom names.
        std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> selfCollisionPairs;
    };

    struct IkConfig {
        std::string robotRootName;
        std::string humanRootName;
        double groundHeight          = 0.0;
        double humanHeightAssumption = 1.0;
        bool useTable1               = true;
        bool useTable2               = true;
        std::unordered_map<std::string, double> humanScaleTable;
        std::vector<IkTaskEntry> tasksTable1;
        std::vector<IkTaskEntry> tasksTable2;
        CollisionAvoidanceConfig collisionAvoidance;
    };

    IkConfig loadIkConfig(const std::filesystem::path& configPath, double actualHumanHeight);

}  // namespace gmr
