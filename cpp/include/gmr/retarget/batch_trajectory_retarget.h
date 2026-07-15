#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "gmr/retarget/batch_trajectory_config.h"
#include "gmr/retarget/human_frame_types.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/retargeter.h"

namespace gmr {

    /// Offline sliding-window batch GN trajectory optimization (MuJoCo FK costs).
    class BatchTrajectoryRetargeter {
       public:
        struct TrackEntry {
            int bodyId = -1;
            double posWeight = 0.0;
            double rotWeight = 0.0;
        };
        struct FrameTaskTarget {
            int bodyId = -1;
            double posWeight = 0.0;
            double rotWeight = 0.0;
            Eigen::Vector3d targetPos = Eigen::Vector3d::Zero();
            Eigen::Quaterniond targetRot = Eigen::Quaterniond::Identity();
        };

        BatchTrajectoryRetargeter(const std::filesystem::path& robotModelPath, IkConfig ikConfig, BatchTrajectoryConfig config = {});

        ~BatchTrajectoryRetargeter();

        BatchTrajectoryRetargeter(const BatchTrajectoryRetargeter&)            = delete;
        BatchTrajectoryRetargeter& operator=(const BatchTrajectoryRetargeter&) = delete;

        /// Bootstrap with ``retargeter`` (per-frame IK), then optimize jointly.
        std::vector<Eigen::VectorXd> retargetBatch(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                                  bool offsetToGround = false);

        const BatchTrajectoryProfile& lastProfile() const { return lastProfile_; }
        int modelNq() const { return nq_; }

       private:
        void buildTrackEntries();
        void buildOptIndices();
        void resolveFootBodyIds();

        HumanFrame prepareHumanFrame(const HumanFrame& frame, bool offsetToGround) const;
        std::vector<FrameTaskTarget> targetsForPrepared(const HumanFrame& prepared) const;

        std::vector<Eigen::VectorXd> bootstrapQ(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                                bool offsetToGround);
        std::vector<int> windowStarts(int nFrames) const;
        std::vector<std::vector<bool>> batchContactMask(const std::vector<Eigen::VectorXd>& qRef) const;

        std::vector<Eigen::VectorXd> optimizeSlidingWindows(const std::vector<Eigen::VectorXd>& qInit,
                                                        const std::vector<std::vector<FrameTaskTarget>>& targets);
        std::vector<Eigen::VectorXd> optimizeGnWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                      const std::vector<std::vector<FrameTaskTarget>>& targets,
                                                      const Eigen::VectorXd& anchor, const std::vector<Eigen::VectorXd>& qRef,
                                                      int frameOffset, double anchorWeight);

        void clipHingeQpos(Eigen::VectorXd& q) const;
        Eigen::VectorXd finalizeQpos(const Eigen::VectorXd& qpos, Retargeter& retargeter, const HumanFrame& prepared,
                                     bool offsetToGround);

        BatchTrajectoryConfig config_;
        IkConfig ikConfig_;
        BatchTrajectoryProfile lastProfile_;

        struct Impl;
        std::unique_ptr<Impl> impl_;
        int nq_ = 0;

        std::vector<TrackEntry> trackEntries_;
        std::vector<int> optVidx_;
        std::vector<int> smoothQidx_;
        std::vector<int> footBodyIds_;
        std::unordered_map<std::string, Eigen::Vector3d> table1PosOffsets_;
        std::unordered_map<std::string, Eigen::Quaterniond> table1RotOffsets_;
        std::vector<std::vector<bool>> globalRefContact_;
        double groundZ_ = 0.0;
    };

}  // namespace gmr
