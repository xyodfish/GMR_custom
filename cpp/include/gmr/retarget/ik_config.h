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

    struct PlanarBaseConfig {
        bool enabled = false;
        std::string frameName;
        std::string humanBody;
        double groundZ = 0.0;
        std::string yawFrame;
        std::string positionSource;
    };

    struct MobileArmChainConfig {
        std::string shoulderFrame;
        std::string elbowFrame;
        std::string wristFrame;
        std::string orientationFrame;
        std::string shoulderHumanBody;
        std::string elbowHumanBody;
        std::string wristHumanBody;
        Eigen::Quaterniond elbowRotationOffset = Eigen::Quaterniond::Identity();
        Eigen::Quaterniond wristRotationOffset = Eigen::Quaterniond::Identity();
    };

    struct MobileUpperBodyConfig {
        bool enabled = false;
        std::string torsoFrame;
        std::string torsoHumanBody;
        Eigen::Quaterniond torsoRotationOffset = Eigen::Quaterniond::Identity();
        Eigen::Vector2d torsoLocalXy = Eigen::Vector2d::Zero();
        double torsoHeightScale = 0.75;
        Eigen::Vector2d torsoHeightRange = Eigen::Vector2d(0.72, 1.12);
        Eigen::Vector3d torsoOrientationLimitDeg = Eigen::Vector3d(20.0, 20.0, 30.0);
        double torsoPositionCost = 120.0;
        double torsoOrientationCost = 30.0;
        int torsoIterations = 20;
        int torsoMinIterations = 4;
        int initialTorsoIterations = 60;
        int initialTorsoMinIterations = 30;
        std::string headFrame;
        std::string headHumanBody;
        Eigen::Vector3d headOrientationLimitDeg = Eigen::Vector3d(30.0, 30.0, 60.0);
        double headOrientationCost = 2.0;
        double armPositionCost = 80.0;
        double elbowOrientationCost = 0.2;
        double wristOrientationCost = 0.25;
        int armIterations = 15;
        int armMinIterations = 8;
        int initialArmIterations = 40;
        int initialArmMinIterations = 20;
        int armTargetPasses = 2;
        double jointLimitMarginDeg = 2.0;
        double postureCost = 0.05;
        std::unordered_map<std::string, double> jointPostureCost;
        std::vector<MobileArmChainConfig> armChains;
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
        PlanarBaseConfig planarBase;
        MobileUpperBodyConfig mobileUpperBody;
    };

    IkConfig loadIkConfig(const std::filesystem::path& configPath, double actualHumanHeight);

}  // namespace gmr
