#pragma once

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "gmr/retarget/batch_trajectory_config.h"
#include "gmr/retarget/contact_ground.h"
#include "gmr/retarget/human_frame_types.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/retargeter.h"

namespace gmr {

    class QpSolveError : public std::runtime_error {
       public:
        using std::runtime_error::runtime_error;
    };

    /// IK factory context for optional parallel bootstrap / finalize in batch TO.
    struct BatchIkBootstrapContext {
        RetargetBackend backend = RetargetBackend::kMujoco;
        RetargetOptions options;
        ContactGroundConfig contactGround;
    };

    /// Offline sliding-window batch GN trajectory optimization (MuJoCo FK costs).
    class BatchTrajectoryRetargeter {
       public:
        struct TrackEntry {
            int bodyId       = -1;
            double posWeight = 0.0;
            double rotWeight = 0.0;
        };
        struct FrameTaskTarget {
            int bodyId                   = -1;
            Eigen::Vector3d targetPos    = Eigen::Vector3d::Zero();
            Eigen::Quaterniond targetRot = Eigen::Quaterniond::Identity();
        };

        /// Per-frame IK targets keyed by MuJoCo body id (table2 overwrites table1, matching Python).
        using FrameTargets = std::unordered_map<int, FrameTaskTarget>;

        BatchTrajectoryRetargeter(const std::filesystem::path& robotModelPath, IkConfig ikConfig, BatchTrajectoryConfig config = {});

        ~BatchTrajectoryRetargeter();

        BatchTrajectoryRetargeter(const BatchTrajectoryRetargeter&)            = delete;
        BatchTrajectoryRetargeter& operator=(const BatchTrajectoryRetargeter&) = delete;

        /// Bootstrap with ``retargeter`` (per-frame IK), then optimize jointly.
        std::vector<Eigen::VectorXd> retargetBatch(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                                   bool offsetToGround = false, const BatchIkBootstrapContext* ikBootstrap = nullptr);

        const BatchTrajectoryProfile& lastProfile() const { return lastProfile_; }
        int modelNq() const { return nq_; }

        BatchTrajectoryConfig& config() { return config_; }
        const BatchTrajectoryConfig& config() const { return config_; }

        void applyContactGroundConfig(const ContactGroundConfig& contactGround);

        HumanFrame prepareHumanFrame(const HumanFrame& frame, bool offsetToGround) const;
        FrameTargets targetsForPrepared(const HumanFrame& prepared) const;

        /// One constrained GN/SCP window solve used by Online QP-MPC.
        struct QpWindowOptions {
            const Eigen::VectorXd* qPrev = nullptr;
            int pinFrames                = 0;
            double wGmr                  = 0.0;
            double dqMax                 = 4.0;
            double motionDt              = 1.0 / 30.0;
            bool useJointLimits          = true;
            bool useVelocityLimits       = true;
            double jointLimitMarginDeg   = 0.0;
            std::string qpBackend        = "daqp";
        };

        std::vector<Eigen::VectorXd> optimizeQpWindow(const std::vector<Eigen::VectorXd>& qInit, const std::vector<FrameTargets>& targets,
                                                      const Eigen::VectorXd& anchor, const std::vector<Eigen::VectorXd>& qRef,
                                                      int frameOffset, double anchorWeight, const QpWindowOptions& qpOpts);

        void clearFootContactSchedule();
        void setFootContactFromQRef(const std::vector<Eigen::VectorXd>& qRef);

       private:
        struct PreparedFrameTargets {
            HumanFrame prepared;
            FrameTargets targets;
        };
        struct PreparedBatchTargets {
            std::vector<HumanFrame> prepared;
            std::vector<FrameTargets> targets;
        };

        void buildTrackEntries();
        void buildOptIndices();
        void buildSmoothMappings();
        void buildTorqueLimitJoints();
        void resolveFootBodyIds();
        void ensureGnWorkspace(int nFrames) const;

        double windowTorquePeakRatio(const std::vector<Eigen::VectorXd>& qWin) const;
        double torqueLimitGateFromRatio(double rPeak);
        void updateTorqueLimitGateFromWindow(const std::vector<Eigen::VectorXd>& qWin);

        /// Inverse-dynamics torque-limit barrier (control feasibility). Adds a GN term / cost
        /// penalising torque beyond kappa*tau_max on the configured joint set.
        void accumulateWindowTorqueLimitGn(const std::vector<Eigen::VectorXd>& qWin, int m) const;
        double windowTorqueCost(const std::vector<Eigen::VectorXd>& qWin) const;

        std::vector<Eigen::VectorXd> bootstrapQ(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter, bool offsetToGround,
                                                const BatchIkBootstrapContext* ikBootstrap);
        PreparedFrameTargets prepareFrameTargets(const HumanFrame& humanFrame, Retargeter& retargeter,
                                                 bool offsetToGround) const;
        PreparedBatchTargets prepareBatchTargets(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                                 bool offsetToGround) const;
        std::vector<Eigen::VectorXd> loadQInitFromJson(const std::filesystem::path& path, std::size_t expectedFrames) const;
        std::vector<int> windowStarts(int nFrames) const;
        std::vector<std::vector<bool>> batchContactMask(const std::vector<Eigen::VectorXd>& qRef) const;
        void buildGlobalRefFootPos(const std::vector<Eigen::VectorXd>& qRef);
        void applyJointLimitMargin(std::vector<Eigen::VectorXd>& qFrames) const;

        std::vector<Eigen::VectorXd> optimizeSlidingWindows(const std::vector<Eigen::VectorXd>& qInit,
                                                            const std::vector<FrameTargets>& targets);
        std::vector<Eigen::VectorXd> optimizeGnWindow(const std::vector<Eigen::VectorXd>& qInit, const std::vector<FrameTargets>& targets,
                                                      const Eigen::VectorXd& anchor, const std::vector<Eigen::VectorXd>& qRef,
                                                      int frameOffset, double anchorWeight, const QpWindowOptions* qpOpts = nullptr);

        void clipHingeQpos(Eigen::VectorXd& q) const;

       public:
        /// Clamp limited hinge joints to their range shrunk by ``marginDeg`` (slide joints use the
        /// hard range). Mirrors Python ``_apply_margin_clip`` so committed poses stay off the limits.
        void clipHingeQposMargin(Eigen::VectorXd& q, double marginDeg) const;

        /// Reset torque-limit gate hysteresis (call on sequence reset).
        void resetTorqueLimitGate();
        double lastTorqueGate() const { return lastTorqueGate_; }
        double lastTorquePeakRatio() const { return lastTorquePeakRatio_; }
        double meanTorqueGate() const { return torqueGateUpdates_ > 0 ? torqueGateSum_ / static_cast<double>(torqueGateUpdates_) : 1.0; }
        double maxTorquePeakRatio() const { return maxTorquePeakRatio_; }

       private:
        void applyGnStepToWindow(std::vector<Eigen::VectorXd>& qWin, const Eigen::VectorXd& dqFlat, double alpha) const;
        double windowCost(const std::vector<Eigen::VectorXd>& qWin, const std::vector<FrameTargets>& targets, const Eigen::VectorXd& anchor,
                          const std::vector<Eigen::VectorXd>& qRef, int frameOffset, double anchorWeight, double wGmr) const;
        Eigen::VectorXd finalizeQpos(const Eigen::VectorXd& qpos, Retargeter& retargeter, const HumanFrame& prepared, bool offsetToGround);
        std::vector<Eigen::VectorXd> finalizeTrajectory(std::vector<Eigen::VectorXd> qOpt, Retargeter& retargeter,
                                                        const std::vector<HumanFrame>& prepared, bool offsetToGround,
                                                        const BatchIkBootstrapContext* ikBootstrap);

        BatchTrajectoryConfig config_;
        IkConfig ikConfig_;
        BatchTrajectoryProfile lastProfile_;

        struct Impl;
        struct GnWorkspace;
        std::unique_ptr<Impl> impl_;
        mutable std::unique_ptr<GnWorkspace> gnWs_;
        int nq_ = 0;

        std::filesystem::path robotModelPath_;
        struct TorqueLimitJoint {
            int localv    = -1;  ///< index within optVidx_ (per-frame variable slot)
            int dof       = -1;  ///< global DoF index
            double tauMax = 0.0;
        };
        std::vector<TorqueLimitJoint> torqueLimitJoints_;
        mutable double motionDtForTorque_ = 1.0 / 30.0;
        double torqueLimitGate_           = 1.0;
        bool torqueGateActive_            = false;
        int torqueGateOnStreak_           = 0;
        int torqueGateOffStreak_          = 0;
        double lastTorqueGate_            = 1.0;
        double lastTorquePeakRatio_       = 0.0;
        int torqueGateUpdates_            = 0;
        double torqueGateSum_             = 0.0;
        double maxTorquePeakRatio_        = 0.0;

        std::vector<TrackEntry> trackEntries_;
        std::vector<int> optVidx_;  // 每一帧里参与优化的 MuJoCo velocity dof index。
        std::vector<int> smoothQidx_;
        std::vector<int> smoothV_;
        std::vector<int> smoothQ_;
        std::unordered_map<int, int> qToOptV_;
        std::vector<int> footBodyIds_;
        std::unordered_map<std::string, Eigen::Vector3d> table1PosOffsets_;
        std::unordered_map<std::string, Eigen::Quaterniond> table1RotOffsets_;
        std::vector<std::vector<bool>> globalRefContact_;
        std::vector<std::vector<Eigen::Vector3d>> globalRefFootPos_;
        double groundZ_ = 0.0;
        std::unique_ptr<ContactGroundPipeline> contactGroundPipeline_;
    };

}  // namespace gmr
