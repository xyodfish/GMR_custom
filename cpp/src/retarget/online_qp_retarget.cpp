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
        bc.wFootOrientation            = config_.wFootOrientation;
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
        bc.wFootOrientation            = config_.wFootOrientation;
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
        contactBuf_.clear();
        qBuf_.clear();
        qRefBuf_.clear();
        frameIndex_      = 0;
        lastFrameMs_     = 0.0;
        lastQpFallback_  = false;
        qpFallbackCount_ = 0;
        lastQpError_.clear();
        batch_->clearFootContactSchedule();
        arrivalBuf_.clear();
        arrivalContactBuf_.clear();
        arrivalPreparedBuf_.clear();
        arrivalHasPrev_     = false;
        arrivalQPrev_       = Eigen::VectorXd();
        arrivalPrevTargets_ = BatchTrajectoryRetargeter::FrameTargets{};
        arrivalPrevContactState_ = ContactGroundState{};
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

    Eigen::VectorXd OnlineQpRetargeter::commitOutputQpos(
        Retargeter& retargeter,
        Eigen::VectorXd q,
        const ContactGroundState& contactState,
        const Eigen::VectorXd* qPrevious) {
        if (config_.jointLimitMarginDeg > 0.0) {
            batch_->clipHingeQposMargin(q, config_.jointLimitMarginDeg);
        }

        if (config_.useVelocityLimits && qPrevious != nullptr) {
            q = batch_->limitVelocityStep(
                *qPrevious,
                q,
                config_.dqMax,
                1.0 / std::max(motionFps_, 1e-6));
        }

        retargeter.setQpos(q);
        if (config_.finalizeContact) {
            retargeter.finalizeContact(contactState);
            q = retargeter.currentQpos();
        }

        batch_->commitFootContactAnchors(contactState, q);

        return q;
    }

    OnlineQpRetargeter::PreparedFrameTargets OnlineQpRetargeter::prepareFrameTargets(
        const HumanFrame& humanFrame,
        Retargeter& retargeter,
        bool offsetToGround,
        const ContactGroundState* contactState) {
        PreparedFrameTargets frame;
        frame.raw      = humanFrame;
        frame.prepared = contactState == nullptr
            ? retargeter.prepareRetargetInput(humanFrame, offsetToGround)
            : retargeter.prepareRetargetInput(humanFrame, *contactState, offsetToGround);
        frame.targets  = batch_->targetsForPrepared(frame.prepared);
        frame.contactState = retargeter.contactGroundState();
        if (contactState != nullptr) {
            frame.contactState.footContacts = contactState->footContacts;
        }

        return frame;
    }

    Eigen::VectorXd OnlineQpRetargeter::seedCausalFrame(const PreparedFrameTargets& frame, Retargeter& retargeter) {
        if (frameIndex_ <= config_.bootstrapGmrFrames) {
            return retargeter.retargetPreparedFrame(frame.raw, frame.prepared);
        }
        if (!qBuf_.empty()) {
            retargeter.setQpos(qBuf_.back());
        }
        if (config_.lightIkIters > 0) {
            return retargeter.retargetPreparedLightIk(frame.raw, frame.prepared, config_.lightIkIters);
        }
        return retargeter.currentQpos();
    }

    std::vector<Eigen::VectorXd> OnlineQpRetargeter::seedWindowFromCursor(const std::vector<PreparedFrameTargets>& frames,
                                                                          const Eigen::VectorXd& qStart,
                                                                          Retargeter& retargeter, bool fullIkFirst) {
        std::vector<Eigen::VectorXd> seeds;
        seeds.reserve(frames.size());

        Eigen::VectorXd qCursor = qStart;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            Eigen::VectorXd qSeed;
            if (fullIkFirst && i == 0) {
                qSeed = retargeter.retargetPreparedFrame(frames[i].raw, frames[i].prepared);
            } else if (config_.lightIkIters > 0) {
                retargeter.setQpos(qCursor);
                qSeed = retargeter.retargetPreparedLightIk(
                    frames[i].raw,
                    frames[i].prepared,
                    config_.lightIkIters);
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
                                                                   const std::vector<Eigen::VectorXd>& qRef,
                                                                   const std::vector<ContactGroundState>& contactStates,
                                                                   const Eigen::VectorXd* qPrev,
                                                                   int pinFrames) {
        // Sync batch weights in case CLI overrode config after construction.
        syncBatchConfig();

        BatchTrajectoryRetargeter::FootContactSchedule contacts;
        contacts.reserve(contactStates.size());
        for (const ContactGroundState& state : contactStates) {
            contacts.push_back(state.footContacts);
        }

        batch_->setFootContactSchedule(contacts, qRef);

        const BatchTrajectoryRetargeter::QpWindowOptions opts = makeQpWindowOptions(qPrev, pinFrames);
        return batch_->optimizeQpWindow(qInit, targets, qInit.front(), qRef, /*frameOffset=*/0, config_.wAnchor, opts);
    }

    Eigen::VectorXd OnlineQpRetargeter::retargetFrame(const HumanFrame& humanFrame, Retargeter& retargeter, bool offsetToGround) {
        return retargetFrameImpl(humanFrame, nullptr, retargeter, offsetToGround);
    }

    Eigen::VectorXd OnlineQpRetargeter::retargetFrame(
        const HumanFrame& humanFrame,
        const ContactGroundState& contactState,
        Retargeter& retargeter,
        bool offsetToGround) {
        return retargetFrameImpl(humanFrame, &contactState, retargeter, offsetToGround);
    }

    Eigen::VectorXd OnlineQpRetargeter::retargetFrameImpl(
        const HumanFrame& humanFrame,
        const ContactGroundState* contactState,
        Retargeter& retargeter,
        bool offsetToGround) {
        const auto t0       = Clock::now();
        lastQpFallback_     = false;
        lastQpError_.clear();
        PreparedFrameTargets prepared = prepareFrameTargets(
            humanFrame,
            retargeter,
            offsetToGround,
            contactState);
        ++frameIndex_;

        Eigen::VectorXd qSeed = seedCausalFrame(prepared, retargeter);
        preparedBuf_.push_back(prepared.prepared);
        targetsBuf_.push_back(prepared.targets);
        contactBuf_.push_back(prepared.contactState);
        qRefBuf_.push_back(qSeed);

        const std::size_t maxlen = static_cast<std::size_t>(std::max(config_.horizon, 8));
        while (preparedBuf_.size() > maxlen) {
            preparedBuf_.pop_front();
            targetsBuf_.pop_front();
            contactBuf_.pop_front();
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
            std::vector<ContactGroundState> contactWin(contactBuf_.end() - Hn, contactBuf_.end());
            const std::size_t windowStart = qList.size() - static_cast<std::size_t>(Hn);
            const Eigen::VectorXd* qPrev  = windowStart > 0 ? &qBuf_[windowStart - 1] : nullptr;
            // Every frame before the newest one has already been emitted. Keeping those
            // frames variable would optimize against a history that the caller never saw.
            const int pin = Hn - 1;
            try {
                auto qOpt = solveQpWindow(qWin, tgtWin, refWin, contactWin, qPrev, pin);
                qOut = qOpt.back();
            } catch (const QpSolveError& error) {
                qOut = qBuf_.back();
                lastQpFallback_ = true;
                lastQpError_ = error.what();
                ++qpFallbackCount_;
            }
        }

        const Eigen::VectorXd* committedPrevious = qBuf_.empty() ? nullptr : &qBuf_.back();
        qOut = commitOutputQpos(retargeter, std::move(qOut), prepared.contactState, committedPrevious);

        appendCommittedQpos(qOut);
        lastFrameMs_ = elapsedMs(t0);
        return qOut;
    }

    void OnlineQpRetargeter::pushArrivedFrame(const HumanFrame& humanFrame) {
        arrivalBuf_.push_back(humanFrame);
        arrivalContactBuf_.push_back(std::nullopt);
    }

    void OnlineQpRetargeter::pushArrivedFrame(
        const HumanFrame& humanFrame,
        const ContactGroundState& contactState) {
        arrivalBuf_.push_back(humanFrame);
        arrivalContactBuf_.push_back(contactState);
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

    Eigen::VectorXd OnlineQpRetargeter::stepLookaheadWindow(
        const std::vector<PreparedFrameTargets>& windowFrames,
        Retargeter& retargeter) {
        if (windowFrames.empty()) {
            throw std::runtime_error("stepLookaheadWindow: empty window");
        }
        const auto t0 = Clock::now();
        lastQpFallback_ = false;
        lastQpError_.clear();
        ++frameIndex_;
        std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin;
        std::vector<ContactGroundState> contactWin;
        tgtWin.reserve(windowFrames.size());
        contactWin.reserve(windowFrames.size());
        for (const PreparedFrameTargets& frame : windowFrames) {
            tgtWin.push_back(frame.targets);
            contactWin.push_back(frame.contactState);
        }

        const Eigen::VectorXd qStart = arrivalHasPrev_ ? arrivalQPrev_ : retargeter.currentQpos();
        std::vector<Eigen::VectorXd> seeds =
            seedWindowFromCursor(windowFrames, qStart, retargeter, frameIndex_ <= config_.bootstrapGmrFrames);

        Eigen::VectorXd qCmd = seeds.front();
        // Commit-frame targets before optional pin insert (always index 0 of tgtWin here).
        const BatchTrajectoryRetargeter::FrameTargets commitTargets = tgtWin.front();
        if (frameIndex_ > config_.bootstrapGmrFrames) {
            std::vector<Eigen::VectorXd> qWin;
            std::vector<Eigen::VectorXd> refWin;
            const int pin = std::min(2, static_cast<int>(qBuf_.size()));
            qWin.reserve(static_cast<std::size_t>(pin) + seeds.size());
            refWin.reserve(static_cast<std::size_t>(pin) + seeds.size());
            for (int i = pin; i > 0; --i) {
                const Eigen::VectorXd& committed = qBuf_[qBuf_.size() - static_cast<std::size_t>(i)];
                qWin.push_back(committed);
                refWin.push_back(committed);
            }

            qWin.insert(qWin.end(), seeds.begin(), seeds.end());
            refWin.insert(refWin.end(), seeds.begin(), seeds.end());
            tgtWin.insert(tgtWin.begin(), static_cast<std::size_t>(pin), arrivalPrevTargets_);
            contactWin.insert(contactWin.begin(), static_cast<std::size_t>(pin), arrivalPrevContactState_);

            try {
                auto qOpt = solveQpWindow(qWin, tgtWin, refWin, contactWin, nullptr, pin);
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

        const Eigen::VectorXd* committedPrevious = arrivalHasPrev_ ? &arrivalQPrev_ : nullptr;
        qCmd = commitOutputQpos(
            retargeter,
            std::move(qCmd),
            windowFrames.front().contactState,
            committedPrevious);

        arrivalPrevTargets_ = commitTargets;
        arrivalPrevContactState_ = windowFrames.front().contactState;
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
            const std::optional<ContactGroundState> contactState = arrivalContactBuf_.front();
            arrivalBuf_.pop_front();
            arrivalContactBuf_.pop_front();
            return contactState.has_value()
                ? retargetFrame(frame, contactState.value(), retargeter, offsetToGround)
                : retargetFrame(frame, retargeter, offsetToGround);
        }

        while (arrivalPreparedBuf_.size() < arrivalBuf_.size()) {
            const std::size_t index = arrivalPreparedBuf_.size();
            const ContactGroundState* contactState = arrivalContactBuf_[index].has_value()
                ? &arrivalContactBuf_[index].value()
                : nullptr;
            arrivalPreparedBuf_.push_back(prepareFrameTargets(
                arrivalBuf_[index],
                retargeter,
                offsetToGround,
                contactState));
        }

        const int Hn = flush ? static_cast<int>(arrivalBuf_.size()) : std::min(config_.horizon, static_cast<int>(arrivalBuf_.size()));
        std::vector<PreparedFrameTargets> window(arrivalPreparedBuf_.begin(), arrivalPreparedBuf_.begin() + Hn);
        Eigen::VectorXd qCmd = stepLookaheadWindow(window, retargeter);
        arrivalBuf_.pop_front();
        arrivalContactBuf_.pop_front();
        arrivalPreparedBuf_.pop_front();
        return qCmd;
    }

}  // namespace gmr
