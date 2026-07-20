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
        bc.windowSize             = config_.horizon;
        bc.windowStride           = 1;
        bc.gnSteps                = config_.sqpIters;
        bc.gnDamping              = config_.gnDamping;
        bc.gnMaxStep              = config_.gnMaxStep;
        bc.wVelocity              = config_.wVelocity;
        bc.wAcceleration          = config_.wAcceleration;
        bc.wAnchor                = config_.wAnchor;
        bc.enableFootPenalties    = config_.enableFootPenalties;
        bc.wFootHeight            = config_.wFootHeight;
        bc.wFootSlip              = config_.wFootSlip;
        bc.wFootIkAnchor          = config_.wFootIkAnchor;
        bc.wRootXyContact         = config_.wRootXyContact;
        bc.wContactJointAnchor    = config_.wContactJointAnchor;
        bc.footContactFromRef     = true;
        bc.smoothRootXyz          = false;
        bc.useGmrInit             = false;
        bc.finalizeContact        = false;
        bc.torqueLimitConstraint  = config_.torqueLimitConstraint;
        bc.torqueLimitMargin      = config_.torqueLimitMargin;
        bc.torqueLimitWeight      = config_.torqueLimitWeight;
        bc.torqueLimitScope       = config_.torqueLimitScope;
        bc.torqueLimitGateMode    = config_.torqueLimitGateMode;
        bc.torqueLimitGateROn     = config_.torqueLimitGateROn;
        bc.torqueLimitGateRFull   = config_.torqueLimitGateRFull;
        bc.torqueLimitGateROff    = config_.torqueLimitGateROff;
        bc.torqueLimitGateMinOnFrames  = config_.torqueLimitGateMinOnFrames;
        bc.torqueLimitGateMinOffFrames = config_.torqueLimitGateMinOffFrames;
        bc.torqueLimitGateFloor   = config_.torqueLimitGateFloor;
        bc.gnLineSearchAlphas     = {1.0, 0.5, 0.25, 0.1};
        bc.gnLineSearchMode       = GnLineSearchMode::kBest;
        bc.useBandedSolver        = false;
        bc.verbose                = config_.verbose;
        return bc;
    }

    OnlineQpRetargeter::OnlineQpRetargeter(const std::filesystem::path& robotModelPath, IkConfig ikConfig,
                                           OnlineQpConfig config)
        : config_(std::move(config)),
          batch_(std::make_unique<BatchTrajectoryRetargeter>(robotModelPath, std::move(ikConfig), makeBatchConfig())) {}

    void OnlineQpRetargeter::reset() {
        preparedBuf_.clear();
        targetsBuf_.clear();
        qBuf_.clear();
        qRefBuf_.clear();
        frameIndex_   = 0;
        lastFrameMs_  = 0.0;
        batch_->clearFootContactSchedule();
        sequenceActive_  = false;
        sequenceK_       = 0;
        sequenceT_       = 0;
        sequenceHasPrev_ = false;
        sequenceFrames_.clear();
        sequencePrepared_.clear();
        sequencePreparedReady_.clear();
        sequenceTargets_.clear();
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

    Eigen::VectorXd OnlineQpRetargeter::softSeed(const HumanFrame& humanFrame, const HumanFrame& prepared,
                                                 Retargeter& retargeter, bool offsetToGround) {
        (void)prepared;
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

    std::vector<Eigen::VectorXd> OnlineQpRetargeter::solveQpWindow(
        const std::vector<Eigen::VectorXd>& qInit,
        const std::vector<BatchTrajectoryRetargeter::FrameTargets>& targets, const std::vector<Eigen::VectorXd>& qRef,
        const Eigen::VectorXd* qPrev, int pinFrames) {
        BatchTrajectoryRetargeter::QpWindowOptions opts;
        opts.qPrev             = qPrev;
        opts.pinFrames         = pinFrames;
        opts.wGmr              = config_.wGmr;
        opts.dqMax             = config_.dqMax;
        opts.motionDt          = 1.0 / std::max(motionFps_, 1e-6);
        opts.useJointLimits      = config_.useJointLimits;
        opts.useVelocityLimits   = config_.useVelocityLimits;
        opts.jointLimitMarginDeg = config_.jointLimitMarginDeg;
        opts.qpBackend           = config_.qpBackend;

        // Sync batch weights in case CLI overrode config after construction.
        BatchTrajectoryConfig& bc = batch_->config();
        bc.gnSteps             = config_.sqpIters;
        bc.gnDamping           = config_.gnDamping;
        bc.gnMaxStep           = config_.gnMaxStep;
        bc.wVelocity           = config_.wVelocity;
        bc.wAcceleration       = config_.wAcceleration;
        bc.wAnchor             = config_.wAnchor;
        bc.enableFootPenalties = config_.enableFootPenalties;
        bc.wFootHeight         = config_.wFootHeight;
        bc.wFootSlip           = config_.wFootSlip;
        bc.wFootIkAnchor       = config_.wFootIkAnchor;
        bc.wRootXyContact      = config_.wRootXyContact;
        bc.wContactJointAnchor = config_.wContactJointAnchor;
        bc.torqueLimitConstraint = config_.torqueLimitConstraint;
        bc.torqueLimitMargin   = config_.torqueLimitMargin;
        bc.torqueLimitWeight   = config_.torqueLimitWeight;
        bc.torqueLimitGateMode = config_.torqueLimitGateMode;
        bc.torqueLimitGateROn  = config_.torqueLimitGateROn;
        bc.torqueLimitGateRFull = config_.torqueLimitGateRFull;
        bc.torqueLimitGateROff = config_.torqueLimitGateROff;
        bc.torqueLimitGateMinOnFrames  = config_.torqueLimitGateMinOnFrames;
        bc.torqueLimitGateMinOffFrames = config_.torqueLimitGateMinOffFrames;
        bc.torqueLimitGateFloor = config_.torqueLimitGateFloor;

        // Match Python: contact from seed/ref trajectory for this window.
        batch_->setFootContactFromQRef(qRef);

        return batch_->optimizeQpWindow(qInit, targets, qInit.front(), qRef, /*frameOffset=*/0, config_.wAnchor, opts);
    }

    Eigen::VectorXd OnlineQpRetargeter::retargetFrame(const HumanFrame& humanFrame, Retargeter& retargeter,
                                                      bool offsetToGround) {
        const auto t0 = Clock::now();
        HumanFrame prepared = retargeter.prepareRetargetInput(humanFrame, offsetToGround);
        auto targets        = batch_->targetsForPrepared(prepared);
        ++frameIndex_;

        Eigen::VectorXd qSeed = softSeed(humanFrame, prepared, retargeter, offsetToGround);
        preparedBuf_.push_back(prepared);
        targetsBuf_.push_back(targets);
        qRefBuf_.push_back(qSeed);

        const std::size_t maxlen = static_cast<std::size_t>(std::max(config_.horizon, 8));
        while (preparedBuf_.size() > maxlen) {
            preparedBuf_.pop_front();
            targetsBuf_.pop_front();
            qRefBuf_.pop_front();
        }
        while (qBuf_.size() > maxlen) {
            qBuf_.pop_front();
        }

        Eigen::VectorXd qOut = qSeed;
        if (frameIndex_ > config_.bootstrapGmrFrames &&
            static_cast<int>(qBuf_.size()) + 1 >= config_.minFrames) {
            std::vector<Eigen::VectorXd> qList(qBuf_.begin(), qBuf_.end());
            qList.push_back(qSeed);
            const int Hn = std::min(config_.horizon, static_cast<int>(qList.size()));
            std::vector<Eigen::VectorXd> qWin(qList.end() - Hn, qList.end());
            std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin(targetsBuf_.end() - Hn, targetsBuf_.end());
            std::vector<Eigen::VectorXd> refWin(qRefBuf_.end() - Hn, qRefBuf_.end());
            const Eigen::VectorXd* qPrev = qBuf_.empty() ? nullptr : &qBuf_.back();
            const int trail = std::min(2, Hn);
            const int pin   = Hn - trail;
            auto qOpt       = solveQpWindow(qWin, tgtWin, refWin, qPrev, pin);
            qOut            = qOpt.back();
        }

        if (config_.finalizeContact) {
            retargeter.setQpos(qOut);
            retargeter.finalizeContact();
            qOut = retargeter.currentQpos();
        } else {
            retargeter.setQpos(qOut);
        }

        if (config_.jointLimitMarginDeg > 0.0) {
            batch_->clipHingeQposMargin(qOut, config_.jointLimitMarginDeg);
            retargeter.setQpos(qOut);
        }

        qBuf_.push_back(qOut);
        lastFrameMs_ = elapsedMs(t0);
        return qOut;
    }

    std::vector<Eigen::VectorXd> OnlineQpRetargeter::retargetSequence(const std::vector<HumanFrame>& humanFrames,
                                                                      Retargeter& retargeter, bool offsetToGround) {
        reset();
        if (!config_.useLookahead) {
            std::vector<Eigen::VectorXd> out;
            out.reserve(humanFrames.size());
            for (const auto& f : humanFrames) {
                out.push_back(retargetFrame(f, retargeter, offsetToGround));
            }
            return out;
        }

        const int T  = static_cast<int>(humanFrames.size());
        const int Hn = config_.horizon;
        std::vector<HumanFrame> preparedAll;
        std::vector<BatchTrajectoryRetargeter::FrameTargets> targetsAll;
        preparedAll.reserve(static_cast<std::size_t>(T));
        targetsAll.reserve(static_cast<std::size_t>(T));
        for (const auto& f : humanFrames) {
            HumanFrame prepared = retargeter.prepareRetargetInput(f, offsetToGround);
            targetsAll.push_back(batch_->targetsForPrepared(prepared));
            preparedAll.push_back(std::move(prepared));
        }

        std::vector<Eigen::VectorXd> out;
        out.reserve(static_cast<std::size_t>(T));
        Eigen::VectorXd qPrev;
        bool hasPrev = false;

        for (int k = 0; k < T; ++k) {
            const auto t0 = Clock::now();
            frameIndex_   = k + 1;
            const int end = std::min(k + Hn, T);

            std::vector<Eigen::VectorXd> seeds;
            seeds.reserve(static_cast<std::size_t>(end - k));
            Eigen::VectorXd qCursor = hasPrev ? qPrev : retargeter.currentQpos();
            for (int i = 0; i < end - k; ++i) {
                const int fi = k + i;
                Eigen::VectorXd qS;
                if (k < config_.bootstrapGmrFrames && i == 0) {
                    qS = retargeter.retargetFrame(humanFrames[static_cast<std::size_t>(fi)], offsetToGround);
                } else if (config_.lightIkIters > 0) {
                    retargeter.setQpos(qCursor);
                    qS = retargeter.retargetLightIk(humanFrames[static_cast<std::size_t>(fi)], offsetToGround,
                                                    config_.lightIkIters);
                } else {
                    qS = qCursor;
                }
                seeds.push_back(qS);
                qCursor = qS;
            }

            Eigen::VectorXd qCmd = seeds.front();
            if (k >= config_.bootstrapGmrFrames) {
                std::vector<Eigen::VectorXd> qWin = seeds;
                std::vector<Eigen::VectorXd> refWin = seeds;
                std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin(
                    targetsAll.begin() + k, targetsAll.begin() + end);
                int pin = 0;
                const Eigen::VectorXd* qPrevPtr = hasPrev ? &qPrev : nullptr;
                if (hasPrev && k > 0) {
                    qWin.insert(qWin.begin(), qPrev);
                    refWin.insert(refWin.begin(), qPrev);
                    tgtWin.insert(tgtWin.begin(), targetsAll[static_cast<std::size_t>(k - 1)]);
                    pin = 1;
                }
                auto qOpt = solveQpWindow(qWin, tgtWin, refWin, qPrevPtr, pin);
                qCmd      = qOpt[static_cast<std::size_t>(pin)];
            }

            if (config_.finalizeContact) {
                retargeter.setQpos(qCmd);
                retargeter.finalizeContact();
                qCmd = retargeter.currentQpos();
            } else {
                retargeter.setQpos(qCmd);
            }

            if (config_.jointLimitMarginDeg > 0.0) {
                batch_->clipHingeQposMargin(qCmd, config_.jointLimitMarginDeg);
                retargeter.setQpos(qCmd);
            }

            out.push_back(qCmd);
            qPrev     = qCmd;
            hasPrev   = true;
            qBuf_.push_back(qCmd);
            lastFrameMs_ = elapsedMs(t0);
        }
        return out;
    }

    void OnlineQpRetargeter::ensurePrepared(int i, Retargeter& retargeter) {
        if (sequencePreparedReady_[static_cast<std::size_t>(i)]) {
            return;
        }
        sequencePrepared_[static_cast<std::size_t>(i)] =
            retargeter.prepareRetargetInput(sequenceFrames_[static_cast<std::size_t>(i)], sequenceOffset_);
        sequenceTargets_[static_cast<std::size_t>(i)] =
            batch_->targetsForPrepared(sequencePrepared_[static_cast<std::size_t>(i)]);
        sequencePreparedReady_[static_cast<std::size_t>(i)] = 1;
    }

    void OnlineQpRetargeter::beginSequence(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                           bool offsetToGround) {
        (void)retargeter;
        reset();
        sequenceFrames_ = humanFrames;
        sequenceT_      = static_cast<int>(humanFrames.size());
        sequenceK_      = 0;
        sequenceOffset_ = offsetToGround;
        sequenceHasPrev_ = false;
        sequenceQPrev_   = Eigen::VectorXd();
        sequencePrepared_.assign(static_cast<std::size_t>(sequenceT_), HumanFrame{});
        sequenceTargets_.assign(static_cast<std::size_t>(sequenceT_), BatchTrajectoryRetargeter::FrameTargets{});
        sequencePreparedReady_.assign(static_cast<std::size_t>(sequenceT_), 0);
        sequenceActive_ = sequenceT_ > 0;
    }

    Eigen::VectorXd OnlineQpRetargeter::stepSequence(Retargeter& retargeter) {
        if (!sequenceActive_ || sequenceK_ >= sequenceT_) {
            throw std::runtime_error("OnlineQpRetargeter::stepSequence called with no active frames.");
        }
        if (!config_.useLookahead) {
            Eigen::VectorXd q =
                retargetFrame(sequenceFrames_[static_cast<std::size_t>(sequenceK_)], retargeter, sequenceOffset_);
            ++sequenceK_;
            return q;
        }

        const auto t0 = Clock::now();
        const int k   = sequenceK_;
        const int Hn  = config_.horizon;
        const int T   = sequenceT_;
        frameIndex_   = k + 1;
        const int end = std::min(k + Hn, T);
        for (int i = k; i < end; ++i) {
            ensurePrepared(i, retargeter);
        }
        if (k > 0) {
            ensurePrepared(k - 1, retargeter);
        }

        std::vector<Eigen::VectorXd> seeds;
        seeds.reserve(static_cast<std::size_t>(end - k));
        Eigen::VectorXd qCursor = sequenceHasPrev_ ? sequenceQPrev_ : retargeter.currentQpos();
        for (int i = 0; i < end - k; ++i) {
            const int fi = k + i;
            Eigen::VectorXd qS;
            if (k < config_.bootstrapGmrFrames && i == 0) {
                qS = retargeter.retargetFrame(sequenceFrames_[static_cast<std::size_t>(fi)], sequenceOffset_);
            } else if (config_.lightIkIters > 0) {
                retargeter.setQpos(qCursor);
                qS = retargeter.retargetLightIk(sequenceFrames_[static_cast<std::size_t>(fi)], sequenceOffset_,
                                                config_.lightIkIters);
            } else {
                qS = qCursor;
            }
            seeds.push_back(qS);
            qCursor = qS;
        }

        Eigen::VectorXd qCmd = seeds.front();
        if (k >= config_.bootstrapGmrFrames) {
            std::vector<Eigen::VectorXd> qWin   = seeds;
            std::vector<Eigen::VectorXd> refWin = seeds;
            std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin(
                sequenceTargets_.begin() + k, sequenceTargets_.begin() + end);
            int pin                              = 0;
            const Eigen::VectorXd* qPrevPtr      = sequenceHasPrev_ ? &sequenceQPrev_ : nullptr;
            if (sequenceHasPrev_ && k > 0) {
                qWin.insert(qWin.begin(), sequenceQPrev_);
                refWin.insert(refWin.begin(), sequenceQPrev_);
                tgtWin.insert(tgtWin.begin(), sequenceTargets_[static_cast<std::size_t>(k - 1)]);
                pin = 1;
            }
            auto qOpt = solveQpWindow(qWin, tgtWin, refWin, qPrevPtr, pin);
            qCmd      = qOpt[static_cast<std::size_t>(pin)];
        }

        if (config_.finalizeContact) {
            retargeter.setQpos(qCmd);
            retargeter.finalizeContact();
            qCmd = retargeter.currentQpos();
        } else {
            retargeter.setQpos(qCmd);
        }

        if (config_.jointLimitMarginDeg > 0.0) {
            batch_->clipHingeQposMargin(qCmd, config_.jointLimitMarginDeg);
            retargeter.setQpos(qCmd);
        }

        sequenceQPrev_   = qCmd;
        sequenceHasPrev_ = true;
        qBuf_.push_back(qCmd);
        lastFrameMs_ = elapsedMs(t0);
        ++sequenceK_;
        return qCmd;
    }

}  // namespace gmr
