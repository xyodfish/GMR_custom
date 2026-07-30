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

    /// Online QP-MPC retargeting (true streaming only — no full-sequence preload).
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

        /// Live arrival buffer: solver only sees frames that have been pushed.
        /// Lookahead uses a short delay of up to ``horizon-1`` frames. While filling, ``stepArrived``
        /// emits traditional GMR for the newest frame (no pop); once buffered, short-horizon QP pops.
        void pushArrivedFrame(const HumanFrame& humanFrame);
        std::size_t arrivalBufferSize() const { return arrivalBuf_.size(); }
        /// Causal: buffer nonempty. Lookahead: fill-GMR pending or QP-ready / flush.
        bool canStepArrived(bool flush = false) const;
        /// True when the next ``stepArrived`` will use fill-phase GMR (buffer not yet at horizon).
        bool arrivalFillGmrPending() const { return arrivalFillPending_; }
        Eigen::VectorXd stepArrived(Retargeter& retargeter, bool offsetToGround = false, bool flush = false);

        int frameIndex() const { return frameIndex_; }
        double lastFrameMs() const { return lastFrameMs_; }
        double lastTorqueGate() const { return batch_ ? batch_->lastTorqueGate() : 1.0; }
        double meanTorqueGate() const { return batch_ ? batch_->meanTorqueGate() : 1.0; }
        double lastTorquePeakRatio() const { return batch_ ? batch_->lastTorquePeakRatio() : 0.0; }
        double maxTorquePeakRatio() const { return batch_ ? batch_->maxTorquePeakRatio() : 0.0; }

       private:
        struct PreparedFrameTargets {
            HumanFrame prepared;
            BatchTrajectoryRetargeter::FrameTargets targets;
        };

        BatchTrajectoryConfig makeBatchConfig() const;
        BatchTrajectoryRetargeter::QpWindowOptions makeQpWindowOptions(const Eigen::VectorXd* qPrev,
                                                                       int pinFrames) const;
        void syncBatchConfig();
        PreparedFrameTargets prepareFrameTargets(const HumanFrame& humanFrame, Retargeter& retargeter,
                                                 bool offsetToGround);
        std::vector<BatchTrajectoryRetargeter::FrameTargets> prepareWindowTargets(
            const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter, bool offsetToGround);
        Eigen::VectorXd seedCausalFrame(const HumanFrame& humanFrame, Retargeter& retargeter,
                                        bool offsetToGround);
        std::vector<Eigen::VectorXd> seedWindowFromCursor(const std::vector<HumanFrame>& humanFrames,
                                                          const Eigen::VectorXd& qStart, Retargeter& retargeter,
                                                          bool offsetToGround, bool fullIkFirst);
        std::vector<Eigen::VectorXd> solveQpWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                   const std::vector<BatchTrajectoryRetargeter::FrameTargets>& targets,
                                                   const std::vector<Eigen::VectorXd>& qRef,
                                                   const Eigen::VectorXd* qPrev, int pinFrames);
        Eigen::VectorXd stepLookaheadWindow(const std::vector<HumanFrame>& windowFrames, Retargeter& retargeter,
                                            bool offsetToGround);
        /// Write qpos and clear ground penetration (even when ``finalizeContact`` preset is false).
        Eigen::VectorXd commitOutputQpos(Retargeter& retargeter, Eigen::VectorXd q);
        void appendCommittedQpos(const Eigen::VectorXd& q);

        OnlineQpConfig config_;
        std::unique_ptr<BatchTrajectoryRetargeter> batch_;
        double motionFps_ = 30.0;

        std::deque<HumanFrame> preparedBuf_;
        std::deque<BatchTrajectoryRetargeter::FrameTargets> targetsBuf_;
        std::deque<Eigen::VectorXd> qBuf_;
        std::deque<Eigen::VectorXd> qRefBuf_;
        int frameIndex_     = 0;
        double lastFrameMs_ = 0.0;

        // Live arrival buffer (true streaming).
        std::deque<HumanFrame> arrivalBuf_;
        bool arrivalHasPrev_     = false;
        bool arrivalFillPending_ = false;  // one GMR emit per push while buffer < horizon
        Eigen::VectorXd arrivalQPrev_;
        BatchTrajectoryRetargeter::FrameTargets arrivalPrevTargets_;
    };

}  // namespace gmr
