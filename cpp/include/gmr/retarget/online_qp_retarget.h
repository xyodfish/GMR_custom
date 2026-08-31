#pragma once

#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "gmr/retarget/batch_trajectory_retarget.h"
#include "gmr/retarget/human_frame_types.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/online_qp_config.h"
#include "gmr/retarget/retargeter.h"

namespace gmr {

    /// MPC-like short-horizon Online QP retargeting (no dynamics/control feedback or full-sequence preload).
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
        Eigen::VectorXd retargetFrame(
            const HumanFrame& humanFrame,
            const ContactGroundState& contactState,
            Retargeter& retargeter,
            bool offsetToGround = false);

        /// Live arrival buffer: solver only sees frames that have been pushed.
        /// Lookahead waits for ``horizon`` arrived frames, then emits and pops the oldest frame.
        void pushArrivedFrame(const HumanFrame& humanFrame);
        void pushArrivedFrame(
            const HumanFrame& humanFrame,
            const ContactGroundState& contactState);
        std::size_t arrivalBufferSize() const { return arrivalBuf_.size(); }
        /// Causal: buffer nonempty. Lookahead: full window or explicit flush.
        bool canStepArrived(bool flush = false) const;
        Eigen::VectorXd stepArrived(Retargeter& retargeter, bool offsetToGround = false, bool flush = false);

        int frameIndex() const { return frameIndex_; }
        double lastFrameMs() const { return lastFrameMs_; }
        double lastTorqueGate() const { return batch_ ? batch_->lastTorqueGate() : 1.0; }
        double meanTorqueGate() const { return batch_ ? batch_->meanTorqueGate() : 1.0; }
        double lastTorquePeakRatio() const { return batch_ ? batch_->lastTorquePeakRatio() : 0.0; }
        double maxTorquePeakRatio() const { return batch_ ? batch_->maxTorquePeakRatio() : 0.0; }
        bool lastQpFallback() const { return lastQpFallback_; }
        std::size_t qpFallbackCount() const { return qpFallbackCount_; }
        const std::string& lastQpError() const { return lastQpError_; }

       private:
        struct PreparedFrameTargets {
            HumanFrame raw;
            HumanFrame prepared;
            BatchTrajectoryRetargeter::FrameTargets targets;
            ContactGroundState contactState;
        };

        BatchTrajectoryConfig makeBatchConfig() const;
        BatchTrajectoryRetargeter::QpWindowOptions makeQpWindowOptions(const Eigen::VectorXd* qPrev,
                                                                       int pinFrames) const;
        void syncBatchConfig();
        PreparedFrameTargets prepareFrameTargets(const HumanFrame& humanFrame, Retargeter& retargeter,
                                                 bool offsetToGround,
                                                 const ContactGroundState* contactState = nullptr);
        Eigen::VectorXd retargetFrameImpl(
            const HumanFrame& humanFrame,
            const ContactGroundState* contactState,
            Retargeter& retargeter,
            bool offsetToGround);
        Eigen::VectorXd seedCausalFrame(const PreparedFrameTargets& frame, Retargeter& retargeter);
        std::vector<Eigen::VectorXd> seedWindowFromCursor(const std::vector<PreparedFrameTargets>& frames,
                                                          const Eigen::VectorXd& qStart, Retargeter& retargeter,
                                                          bool fullIkFirst);
        std::vector<Eigen::VectorXd> solveQpWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                   const std::vector<BatchTrajectoryRetargeter::FrameTargets>& targets,
                                                   const std::vector<Eigen::VectorXd>& qRef,
                                                   const std::vector<ContactGroundState>& contactStates,
                                                   const Eigen::VectorXd* qPrev, int pinFrames);
        Eigen::VectorXd stepLookaheadWindow(const std::vector<PreparedFrameTargets>& windowFrames,
                                            Retargeter& retargeter);
        /// Write qpos, optionally finalize contact, then apply the configured joint-limit margin.
        Eigen::VectorXd commitOutputQpos(Retargeter& retargeter, Eigen::VectorXd q,
                                         const ContactGroundState& contactState,
                                         const Eigen::VectorXd* qPrevious);
        void appendCommittedQpos(const Eigen::VectorXd& q);

        OnlineQpConfig config_;
        std::unique_ptr<BatchTrajectoryRetargeter> batch_;
        double motionFps_ = 30.0;

        std::deque<HumanFrame> preparedBuf_;
        std::deque<BatchTrajectoryRetargeter::FrameTargets> targetsBuf_;
        std::deque<ContactGroundState> contactBuf_;
        std::deque<Eigen::VectorXd> qBuf_;
        std::deque<Eigen::VectorXd> qRefBuf_;
        int frameIndex_              = 0;
        double lastFrameMs_          = 0.0;
        bool lastQpFallback_         = false;
        std::size_t qpFallbackCount_ = 0;
        std::string lastQpError_;

        // Live arrival buffer (true streaming).
        std::deque<HumanFrame> arrivalBuf_;
        std::deque<std::optional<ContactGroundState>> arrivalContactBuf_;
        std::deque<PreparedFrameTargets> arrivalPreparedBuf_;
        bool arrivalHasPrev_ = false;
        Eigen::VectorXd arrivalQPrev_;
        BatchTrajectoryRetargeter::FrameTargets arrivalPrevTargets_;
        ContactGroundState arrivalPrevContactState_;
    };

}  // namespace gmr
