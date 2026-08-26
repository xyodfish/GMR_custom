#include "gmr/retarget/online_qp_retarget.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace gmr {
    namespace {

        using Clock = std::chrono::steady_clock;

        double elapsedMs(const Clock::time_point& t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }

    }  // namespace

    BatchTrajectoryConfig OnlineQpRetargeter::makeBatchConfig() const {
        BatchTrajectoryConfig bc;
        bc.windowSize                  = config_.horizon;
        bc.windowStride                = 1;
        bc.gnSteps                     = config_.sqpIters;
        bc.gnDamping                   = config_.gnDamping;
        bc.gnMaxStep                   = config_.gnMaxStep;
        bc.wVelocity                   = config_.wVelocity;
        bc.wAcceleration               = config_.wAcceleration;
        bc.wAnchor                     = config_.wAnchor;
        bc.enableFootPenalties         = config_.enableFootPenalties;
        bc.wFootHeight                 = config_.wFootHeight;
        bc.wFootSlip                   = config_.wFootSlip;
        bc.wFootIkAnchor               = config_.wFootIkAnchor;
        bc.wRootXyContact              = config_.wRootXyContact;
        bc.wContactJointAnchor         = config_.wContactJointAnchor;
        bc.footContactFromRef          = true;
        bc.smoothRootXyz               = false;
        bc.useGmrInit                  = false;
        bc.finalizeContact             = false;
        bc.torqueLimitConstraint       = config_.torqueLimitConstraint;
        bc.torqueLimitMargin           = config_.torqueLimitMargin;
        bc.torqueLimitWeight           = config_.torqueLimitWeight;
        bc.torqueLimitScope            = config_.torqueLimitScope;
        bc.torqueLimitGateMode         = config_.torqueLimitGateMode;
        bc.torqueLimitGateROn          = config_.torqueLimitGateROn;
        bc.torqueLimitGateRFull        = config_.torqueLimitGateRFull;
        bc.torqueLimitGateROff         = config_.torqueLimitGateROff;
        bc.torqueLimitGateMinOnFrames  = config_.torqueLimitGateMinOnFrames;
        bc.torqueLimitGateMinOffFrames = config_.torqueLimitGateMinOffFrames;
        bc.torqueLimitGateFloor        = config_.torqueLimitGateFloor;
        bc.gnLineSearchAlphas          = {1.0, 0.5, 0.25, 0.1};
        bc.gnLineSearchMode            = GnLineSearchMode::kBest;
        bc.useBandedSolver             = false;
        bc.verbose                     = config_.verbose;
        return bc;
    }

    OnlineQpRetargeter::OnlineQpRetargeter(const std::filesystem::path& robotModelPath, IkConfig ikConfig, OnlineQpConfig config)
        : config_(std::move(config)),
          batch_(std::make_unique<BatchTrajectoryRetargeter>(robotModelPath, std::move(ikConfig), makeBatchConfig())) {}

    BatchTrajectoryRetargeter::QpWindowOptions OnlineQpRetargeter::makeQpWindowOptions(const Eigen::VectorXd* qPrev,
                                                                                       int pinFrames) const {
        BatchTrajectoryRetargeter::QpWindowOptions opts;
        opts.qPrev               = qPrev;
        opts.pinFrames           = pinFrames;
        opts.wGmr                = config_.wGmr;
        opts.dqMax               = config_.dqMax;
        opts.motionDt            = 1.0 / std::max(motionFps_, 1e-6);
        opts.useJointLimits      = config_.useJointLimits;
        opts.useVelocityLimits   = config_.useVelocityLimits;
        opts.jointLimitMarginDeg = config_.jointLimitMarginDeg;
        opts.qpBackend           = config_.qpBackend;
        return opts;
    }

    void OnlineQpRetargeter::syncBatchConfig() {
        BatchTrajectoryConfig& bc      = batch_->config();
        bc.gnSteps                     = config_.sqpIters;
        bc.gnDamping                   = config_.gnDamping;
        bc.gnMaxStep                   = config_.gnMaxStep;
        bc.wVelocity                   = config_.wVelocity;
        bc.wAcceleration               = config_.wAcceleration;
        bc.wAnchor                     = config_.wAnchor;
        bc.enableFootPenalties         = config_.enableFootPenalties;
        bc.wFootHeight                 = config_.wFootHeight;
        bc.wFootSlip                   = config_.wFootSlip;
        bc.wFootIkAnchor               = config_.wFootIkAnchor;
        bc.wRootXyContact              = config_.wRootXyContact;
        bc.wContactJointAnchor         = config_.wContactJointAnchor;
        bc.torqueLimitConstraint       = config_.torqueLimitConstraint;
        bc.torqueLimitMargin           = config_.torqueLimitMargin;
        bc.torqueLimitWeight           = config_.torqueLimitWeight;
        bc.torqueLimitGateMode         = config_.torqueLimitGateMode;
        bc.torqueLimitGateROn          = config_.torqueLimitGateROn;
        bc.torqueLimitGateRFull        = config_.torqueLimitGateRFull;
        bc.torqueLimitGateROff         = config_.torqueLimitGateROff;
        bc.torqueLimitGateMinOnFrames  = config_.torqueLimitGateMinOnFrames;
        bc.torqueLimitGateMinOffFrames = config_.torqueLimitGateMinOffFrames;
        bc.torqueLimitGateFloor        = config_.torqueLimitGateFloor;
    }

    void OnlineQpRetargeter::reset() {
        preparedBuf_.clear();
        targetsBuf_.clear();
        qBuf_.clear();
        qRefBuf_.clear();
        frameIndex_      = 0;
        lastFrameMs_     = 0.0;
        lastQpFallback_  = false;
        qpFallbackCount_ = 0;
        lastQpError_.clear();
        batch_->clearFootContactSchedule();
        arrivalBuf_.clear();
        arrivalHasPrev_     = false;
        arrivalQPrev_       = Eigen::VectorXd();
        arrivalPrevTargets_ = BatchTrajectoryRetargeter::FrameTargets{};
        batch_->resetTorqueLimitGate();
    }

    void OnlineQpRetargeter::setMotionFps(double fps) {
        if (fps > 0.0) {
            motionFps_ = fps;
        }
    }

    void OnlineQpRetargeter::applyContactGroundConfig(const ContactGroundConfig& contactGround) {
        batch_->applyContactGroundConfig(contactGround);
    }

    Eigen::VectorXd OnlineQpRetargeter::commitOutputQpos(Retargeter& retargeter, Eigen::VectorXd q) {
        retargeter.setQpos(q);
        if (config_.finalizeContact) {
            retargeter.finalizeContact();
            q = retargeter.currentQpos();
        }

        if (config_.jointLimitMarginDeg > 0.0) {
            batch_->clipHingeQposMargin(q, config_.jointLimitMarginDeg);
            retargeter.setQpos(q);
        }
        return q;
    }

    OnlineQpRetargeter::PreparedFrameTargets OnlineQpRetargeter::prepareFrameTargets(
        const HumanFrame& humanFrame, Retargeter& retargeter, bool offsetToGround) {
        PreparedFrameTargets frame;
        frame.prepared = retargeter.prepareRetargetInput(humanFrame, offsetToGround);
        frame.targets  = batch_->targetsForPrepared(frame.prepared);
        return frame;
    }

    std::vector<BatchTrajectoryRetargeter::FrameTargets> OnlineQpRetargeter::prepareWindowTargets(
        const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter, bool offsetToGround) {
        std::vector<BatchTrajectoryRetargeter::FrameTargets> targets;
        targets.reserve(humanFrames.size());
        for (const auto& frame : humanFrames) {
            targets.push_back(prepareFrameTargets(frame, retargeter, offsetToGround).targets);
        }
        return targets;
    }

    Eigen::VectorXd OnlineQpRetargeter::seedCausalFrame(const HumanFrame& humanFrame, Retargeter& retargeter,
                                                        bool offsetToGround) {
        if (frameIndex_ <= config_.bootstrapGmrFrames) {
            return retargeter.retargetFrame(humanFrame, offsetToGround);
        }
        if (!qBuf_.empty()) {
            retargeter.setQpos(qBuf_.back());
        }
        if (config_.lightIkIters > 0) {
            return retargeter.retargetLightIk(humanFrame, offsetToGround, config_.lightIkIters);
        }
        return retargeter.currentQpos();
    }

    std::vector<Eigen::VectorXd> OnlineQpRetargeter::seedWindowFromCursor(const std::vector<HumanFrame>& humanFrames,
                                                                          const Eigen::VectorXd& qStart,
                                                                          Retargeter& retargeter, bool offsetToGround,
                                                                          bool fullIkFirst) {
        std::vector<Eigen::VectorXd> seeds;
        seeds.reserve(humanFrames.size());

        Eigen::VectorXd qCursor = qStart;
        for (std::size_t i = 0; i < humanFrames.size(); ++i) {
            Eigen::VectorXd qSeed;
            if (fullIkFirst && i == 0) {
                qSeed = retargeter.retargetFrame(humanFrames[i], offsetToGround);
            } else if (config_.lightIkIters > 0) {
                retargeter.setQpos(qCursor);
                qSeed = retargeter.retargetLightIk(humanFrames[i], offsetToGround, config_.lightIkIters);
            } else {
                qSeed = qCursor;
            }
            seeds.push_back(qSeed);
            qCursor = qSeed;
        }
        return seeds;
    }

    void OnlineQpRetargeter::appendCommittedQpos(const Eigen::VectorXd& q) {
        qBuf_.push_back(q);
        const std::size_t maxlen = static_cast<std::size_t>(std::max(config_.horizon, 8));
        while (qBuf_.size() > maxlen) {
            qBuf_.pop_front();
        }
    }

    std::vector<Eigen::VectorXd> OnlineQpRetargeter::solveQpWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                                   const std::vector<BatchTrajectoryRetargeter::FrameTargets>& targets,
                                                                   const std::vector<Eigen::VectorXd>& qRef, const Eigen::VectorXd* qPrev,
                                                                   int pinFrames) {
        // Sync batch weights in case CLI overrode config after construction.
        syncBatchConfig();

        // Match Python: contact from seed/ref trajectory for this window.
        batch_->setFootContactFromQRef(qRef);

        const BatchTrajectoryRetargeter::QpWindowOptions opts = makeQpWindowOptions(qPrev, pinFrames);
        return batch_->optimizeQpWindow(qInit, targets, qInit.front(), qRef, /*frameOffset=*/0, config_.wAnchor, opts);
    }

    Eigen::VectorXd OnlineQpRetargeter::retargetFrame(const HumanFrame& humanFrame, Retargeter& retargeter, bool offsetToGround) {
        const auto t0       = Clock::now();
        lastQpFallback_     = false;
        lastQpError_.clear();
        PreparedFrameTargets prepared = prepareFrameTargets(humanFrame, retargeter, offsetToGround);
        ++frameIndex_;

        Eigen::VectorXd qSeed = seedCausalFrame(humanFrame, retargeter, offsetToGround);
        preparedBuf_.push_back(prepared.prepared);
        targetsBuf_.push_back(prepared.targets);
        qRefBuf_.push_back(qSeed);

        const std::size_t maxlen = static_cast<std::size_t>(std::max(config_.horizon, 8));
        while (preparedBuf_.size() > maxlen) {
            preparedBuf_.pop_front();
            targetsBuf_.pop_front();
            qRefBuf_.pop_front();
        }

        Eigen::VectorXd qOut = qSeed;
        if (frameIndex_ > config_.bootstrapGmrFrames && !qBuf_.empty() &&
            static_cast<int>(qBuf_.size()) + 1 >= config_.minFrames) {
            std::vector<Eigen::VectorXd> qList(qBuf_.begin(), qBuf_.end());
            qList.push_back(qSeed);
            const int Hn = std::min(config_.horizon, static_cast<int>(qList.size()));
            std::vector<Eigen::VectorXd> qWin(qList.end() - Hn, qList.end());
            std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin(targetsBuf_.end() - Hn, targetsBuf_.end());
            std::vector<Eigen::VectorXd> refWin(qRefBuf_.end() - Hn, qRefBuf_.end());
            const std::size_t windowStart = qList.size() - static_cast<std::size_t>(Hn);
            const Eigen::VectorXd* qPrev  = windowStart > 0 ? &qBuf_[windowStart - 1] : nullptr;
            const int trail              = std::min(2, Hn);
            const int pin                = Hn - trail;
            try {
                auto qOpt = solveQpWindow(qWin, tgtWin, refWin, qPrev, pin);
                qOut = qOpt.back();
            } catch (const QpSolveError& error) {
                qOut = qBuf_.back();
                lastQpFallback_ = true;
                lastQpError_ = error.what();
                ++qpFallbackCount_;
            }
        }

        qOut = commitOutputQpos(retargeter, std::move(qOut));

        appendCommittedQpos(qOut);
        lastFrameMs_ = elapsedMs(t0);
        return qOut;
    }

    void OnlineQpRetargeter::pushArrivedFrame(const HumanFrame& humanFrame) {
        arrivalBuf_.push_back(humanFrame);
    }

    bool OnlineQpRetargeter::canStepArrived(bool flush) const {
        if (arrivalBuf_.empty()) {
            return false;
        }
        if (!config_.useLookahead || flush) {
            return true;
        }
        return static_cast<int>(arrivalBuf_.size()) >= std::max(1, config_.horizon);
    }

    Eigen::VectorXd OnlineQpRetargeter::stepLookaheadWindow(const std::vector<HumanFrame>& windowFrames, Retargeter& retargeter,
                                                            bool offsetToGround) {
        if (windowFrames.empty()) {
            throw std::runtime_error("stepLookaheadWindow: empty window");
        }
        const auto t0 = Clock::now();
        lastQpFallback_ = false;
        lastQpError_.clear();
        ++frameIndex_;
        std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin =
            prepareWindowTargets(windowFrames, retargeter, offsetToGround);

        const Eigen::VectorXd qStart = arrivalHasPrev_ ? arrivalQPrev_ : retargeter.currentQpos();
        std::vector<Eigen::VectorXd> seeds =
            seedWindowFromCursor(windowFrames, qStart, retargeter, offsetToGround,
                                 frameIndex_ <= config_.bootstrapGmrFrames);

        Eigen::VectorXd qCmd = seeds.front();
        // Commit-frame targets before optional pin insert (always index 0 of tgtWin here).
        const BatchTrajectoryRetargeter::FrameTargets commitTargets = tgtWin.front();
        if (frameIndex_ > config_.bootstrapGmrFrames) {
            std::vector<Eigen::VectorXd> qWin   = seeds;
            std::vector<Eigen::VectorXd> refWin = seeds;
            int pin                             = 0;
            const Eigen::VectorXd* qPrevPtr     = arrivalHasPrev_ ? &arrivalQPrev_ : nullptr;
            if (arrivalHasPrev_) {
                qWin.insert(qWin.begin(), arrivalQPrev_);
                refWin.insert(refWin.begin(), arrivalQPrev_);
                tgtWin.insert(tgtWin.begin(), arrivalPrevTargets_);
                pin = 1;
            }
            try {
                auto qOpt = solveQpWindow(qWin, tgtWin, refWin, qPrevPtr, pin);
                qCmd = qOpt[static_cast<std::size_t>(pin)];
            } catch (const QpSolveError& error) {
                if (!arrivalHasPrev_) {
                    throw;
                }

                qCmd = arrivalQPrev_;
                lastQpFallback_ = true;
                lastQpError_ = error.what();
                ++qpFallbackCount_;
            }
        }

        qCmd = commitOutputQpos(retargeter, std::move(qCmd));

        if (!lastQpFallback_) {
            arrivalPrevTargets_ = commitTargets;
        }

        arrivalQPrev_       = qCmd;
        arrivalHasPrev_     = true;
        appendCommittedQpos(qCmd);
        lastFrameMs_ = elapsedMs(t0);
        return qCmd;
    }

    Eigen::VectorXd OnlineQpRetargeter::stepArrived(Retargeter& retargeter, bool offsetToGround, bool flush) {
        if (!canStepArrived(flush)) {
            throw std::runtime_error("OnlineQpRetargeter::stepArrived: arrival buffer not ready");
        }
        if (!config_.useLookahead) {
            HumanFrame frame = arrivalBuf_.front();
            arrivalBuf_.pop_front();
            return retargetFrame(frame, retargeter, offsetToGround);
        }

        const int Hn = flush ? static_cast<int>(arrivalBuf_.size()) : std::min(config_.horizon, static_cast<int>(arrivalBuf_.size()));
        std::vector<HumanFrame> window(arrivalBuf_.begin(), arrivalBuf_.begin() + Hn);
        Eigen::VectorXd qCmd = stepLookaheadWindow(window, retargeter, offsetToGround);
        arrivalBuf_.pop_front();
        return qCmd;
    }

}  // namespace gmr
