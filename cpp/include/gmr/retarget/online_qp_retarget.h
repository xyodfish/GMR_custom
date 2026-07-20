#pragma once

#include <deque>
#include <filesystem>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "gmr/retarget/batch_trajectory_retarget.h"
#include "gmr/retarget/human_frame_types.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/online_qp_config.h"
#include "gmr/retarget/retargeter.h"

namespace gmr {

    /// Online QP-MPC retargeting (Python ``OnlineQpRetargeter``).
    class OnlineQpRetargeter {
       public:
        OnlineQpRetargeter(const std::filesystem::path& robotModelPath, IkConfig ikConfig, OnlineQpConfig config = {});

        void reset();
        void setMotionFps(double fps);
        void applyContactGroundConfig(const ContactGroundConfig& contactGround);

        OnlineQpConfig& config() { return config_; }
        const OnlineQpConfig& config() const { return config_; }

        /// Streaming causal API (one human frame in → one qpos out).
        Eigen::VectorXd retargetFrame(const HumanFrame& humanFrame, Retargeter& retargeter, bool offsetToGround = false);

        /// Full sequence; lookahead MPC when ``config.useLookahead``.
        std::vector<Eigen::VectorXd> retargetSequence(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                                      bool offsetToGround = false);

        /// Start buffered streaming (lookahead peeks future frames when enabled).
        void beginSequence(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter, bool offsetToGround = false);
        bool sequenceActive() const { return sequenceActive_; }
        bool sequenceDone() const { return !sequenceActive_ || sequenceK_ >= sequenceT_; }
        /// Advance one committed frame; throws if sequence not started / already done.
        Eigen::VectorXd stepSequence(Retargeter& retargeter);

        int frameIndex() const { return frameIndex_; }
        double lastFrameMs() const { return lastFrameMs_; }
        double lastTorqueGate() const { return batch_ ? batch_->lastTorqueGate() : 1.0; }
        double meanTorqueGate() const { return batch_ ? batch_->meanTorqueGate() : 1.0; }
        double lastTorquePeakRatio() const { return batch_ ? batch_->lastTorquePeakRatio() : 0.0; }
        double maxTorquePeakRatio() const { return batch_ ? batch_->maxTorquePeakRatio() : 0.0; }

       private:
        BatchTrajectoryConfig makeBatchConfig() const;
        Eigen::VectorXd softSeed(const HumanFrame& humanFrame, const HumanFrame& prepared, Retargeter& retargeter,
                                 bool offsetToGround);
        std::vector<Eigen::VectorXd> solveQpWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                   const std::vector<BatchTrajectoryRetargeter::FrameTargets>& targets,
                                                   const std::vector<Eigen::VectorXd>& qRef,
                                                   const Eigen::VectorXd* qPrev, int pinFrames);
        void ensurePrepared(int i, Retargeter& retargeter);

        OnlineQpConfig config_;
        std::unique_ptr<BatchTrajectoryRetargeter> batch_;
        double motionFps_ = 30.0;

        std::deque<HumanFrame> preparedBuf_;
        std::deque<BatchTrajectoryRetargeter::FrameTargets> targetsBuf_;
        std::deque<Eigen::VectorXd> qBuf_;
        std::deque<Eigen::VectorXd> qRefBuf_;
        int frameIndex_     = 0;
        double lastFrameMs_ = 0.0;

        // Streaming sequence state (lookahead / causal).
        bool sequenceActive_   = false;
        bool sequenceOffset_   = false;
        int sequenceK_         = 0;
        int sequenceT_         = 0;
        bool sequenceHasPrev_  = false;
        Eigen::VectorXd sequenceQPrev_;
        std::vector<HumanFrame> sequenceFrames_;
        std::vector<HumanFrame> sequencePrepared_;
        std::vector<char> sequencePreparedReady_;
        std::vector<BatchTrajectoryRetargeter::FrameTargets> sequenceTargets_;
    };

}  // namespace gmr
