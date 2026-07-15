#pragma once

#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <mujoco/mujoco.h>

#include "gmr/retarget/human_frame_types.h"

namespace gmr {

    struct ContactGroundConfig {
        bool enabled = false;
        std::vector<std::string> footBodies;
        std::string humanRootName;
        double velThreshold = 0.5;
        double heightThreshold = 0.08;
        double heightOffThreshold = 0.12;
        int velWindow = 6;
        double groundZ = 0.0;
        double groundMargin = 0.02;
        double lpfAlpha = 0.3;
        bool enableFootLock = true;
        double footLockEmaAlpha = 0.05;
        bool fixRobotPenetration = true;
        bool footGroundLimitEnabled = false;
        bool penetrationExcludeFeetWhenFootLimit = true;
        double penetrationMargin = 0.01;
        double lyingHipHeightThreshold = 0.45;
        double lowPoseFootHeightThreshold = 0.20;
        double lowPoseMaxHipHeight = 0.65;
        double lyingPenetrationMargin = 0.02;
        int penetrationMaxIterations = 5;
        double airborneHeightThreshold = 0.15;
        double airborneOffsetDecay = 1.0;
        std::string floorGeomName = "floor";
        std::vector<std::string> footCollisionGeoms;
        std::vector<std::string> robotFootBodies;
        std::vector<std::string> robotTrunkBodies;
        std::vector<std::string> robotLegBodies;
        std::vector<std::string> robotArmBodies;
    };

    class ContactGroundPipeline {
       public:
        ContactGroundPipeline(ContactGroundConfig config, const mjModel* model, double fps = 30.0);

        bool enabled() const { return config_.enabled; }
        void setFps(double fps);

        HumanFrame processHumanFrame(const HumanFrame& humanData);
        double fixRobotPenetration(mjModel* model, mjData* data);

        double lastRootLift() const { return lastRootLift_; }

       private:
        ContactGroundConfig config_;
        const mjModel* model_ = nullptr;
        double fps_ = 30.0;

        std::vector<int> footGeomIds_;
        std::vector<int> trunkGeomIds_;
        std::vector<int> legGeomIds_;
        std::vector<int> armGeomIds_;
        std::vector<int> groundGeomIds_;
        std::vector<int> lyingGroundGeomIds_;
        int floorGeomId_ = -1;

        std::deque<std::unordered_map<std::string, Eigen::Vector3d>> footPosBuf_;
        std::unordered_map<std::string, bool> lastContacts_;
        std::unordered_map<std::string, Eigen::Vector3d> lockedFeet_;
        double groundAlignOffset_ = 0.0;
        double lastHumanHipZ_ = std::numeric_limits<double>::infinity();
        double lastMinFootZ_ = std::numeric_limits<double>::infinity();
        double lastRootLift_ = 0.0;

        bool isLowPose() const;
        std::pair<const std::vector<int>&, double> penetrationTargets() const;
    };

}  // namespace gmr
