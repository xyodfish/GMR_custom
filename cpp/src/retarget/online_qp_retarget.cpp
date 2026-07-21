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

    void OnlineQpRetargeter::reset() {
        preparedBuf_.clear();
        targetsBuf_.clear();
        qBuf_.clear();
        qRefBuf_.clear();
        frameIndex_  = 0;
        lastFrameMs_ = 0.0;
        batch_->clearFootContactSchedule();
        arrivalBuf_.clear();
        arrivalHasPrev_     = false;
        arrivalFillPending_ = false;
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
        // anti_slip sets finalizeContact=false for historical speed reasons, but Python still runs
        // _apply_penetration_fix. Always lift root-Z here so feet/trunk clear the floor.
        retargeter.finalizeContact();
        q = retargeter.currentQpos();
        if (config_.jointLimitMarginDeg > 0.0) {
            batch_->clipHingeQposMargin(q, config_.jointLimitMarginDeg);
            retargeter.setQpos(q);
        }
        return q;
    }

    Eigen::VectorXd OnlineQpRetargeter::softSeed(const HumanFrame& humanFrame, const HumanFrame& prepared, Retargeter& retargeter,
                                                 bool offsetToGround) {
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

    std::vector<Eigen::VectorXd> OnlineQpRetargeter::solveQpWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                                   const std::vector<BatchTrajectoryRetargeter::FrameTargets>& targets,
                                                                   const std::vector<Eigen::VectorXd>& qRef, const Eigen::VectorXd* qPrev,
                                                                   int pinFrames) {
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

        // Sync batch weights in case CLI overrode config after construction.
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

        // Match Python: contact from seed/ref trajectory for this window.
        batch_->setFootContactFromQRef(qRef);

        return batch_->optimizeQpWindow(qInit, targets, qInit.front(), qRef, /*frameOffset=*/0, config_.wAnchor, opts);
    }

    Eigen::VectorXd OnlineQpRetargeter::retargetFrame(const HumanFrame& humanFrame, Retargeter& retargeter, bool offsetToGround) {
        const auto t0       = Clock::now();
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
        if (frameIndex_ > config_.bootstrapGmrFrames && static_cast<int>(qBuf_.size()) + 1 >= config_.minFrames) {
            std::vector<Eigen::VectorXd> qList(qBuf_.begin(), qBuf_.end());
            qList.push_back(qSeed);
            const int Hn = std::min(config_.horizon, static_cast<int>(qList.size()));
            std::vector<Eigen::VectorXd> qWin(qList.end() - Hn, qList.end());
            std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin(targetsBuf_.end() - Hn, targetsBuf_.end());
            std::vector<Eigen::VectorXd> refWin(qRefBuf_.end() - Hn, qRefBuf_.end());
            const Eigen::VectorXd* qPrev = qBuf_.empty() ? nullptr : &qBuf_.back();
            const int trail              = std::min(2, Hn);
            const int pin                = Hn - trail;
            auto qOpt                    = solveQpWindow(qWin, tgtWin, refWin, qPrev, pin);
            qOut                         = qOpt.back();
        }

        qOut = commitOutputQpos(retargeter, std::move(qOut));

        qBuf_.push_back(qOut);
        lastFrameMs_ = elapsedMs(t0);
        return qOut;
    }

    std::vector<Eigen::VectorXd> OnlineQpRetargeter::retargetSequence(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                                                      bool offsetToGround) {
        // Convenience only: same live arrival path as the viewer (no future-frame peek).
        reset();
        std::vector<Eigen::VectorXd> out;
        out.reserve(humanFrames.size());
        for (const auto& f : humanFrames) {
            pushArrivedFrame(f);
            while (canStepArrived(/*flush=*/false)) {
                out.push_back(stepArrived(retargeter, offsetToGround, /*flush=*/false));
            }
        }
        while (canStepArrived(/*flush=*/true)) {
            out.push_back(stepArrived(retargeter, offsetToGround, /*flush=*/true));
        }
        return out;
    }

    void OnlineQpRetargeter::pushArrivedFrame(const HumanFrame& humanFrame) {
        arrivalBuf_.push_back(humanFrame);
        if (config_.useLookahead && static_cast<int>(arrivalBuf_.size()) < std::max(1, config_.horizon)) {
            // One traditional-GMR output per push while the lookahead buffer fills.
            arrivalFillPending_ = true;
        }
    }

    bool OnlineQpRetargeter::canStepArrived(bool flush) const {
        if (arrivalBuf_.empty()) {
            return false;
        }
        if (!config_.useLookahead || flush) {
            return true;
        }
        if (arrivalFillPending_) {
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
        ++frameIndex_;
        const int Hn = static_cast<int>(windowFrames.size());

        std::vector<BatchTrajectoryRetargeter::FrameTargets> tgtWin;
        tgtWin.reserve(static_cast<std::size_t>(Hn));
        for (const auto& f : windowFrames) {
            // human frame 预处理 包括 root offset ground 相关处理
            // targetsForPrepared 则 根据ik config 把human target 映射成 robot body target
            HumanFrame prepared = retargeter.prepareRetargetInput(f, offsetToGround);
            tgtWin.push_back(batch_->targetsForPrepared(prepared));
        }

        // 生成seed
        std::vector<Eigen::VectorXd> seeds;
        seeds.reserve(static_cast<std::size_t>(Hn));

        // 当前滚动到的机器人姿态 一个在窗口帧上滚动前进的 qpos 指针
        Eigen::VectorXd qCursor = arrivalHasPrev_ ? arrivalQPrev_ : retargeter.currentQpos();

        // 每个窗口帧生成一个qSeed
        for (int i = 0; i < Hn; ++i) {
            Eigen::VectorXd qSeed;
            if (frameIndex_ <= config_.bootstrapGmrFrames && i == 0) {
                qSeed = retargeter.retargetFrame(windowFrames[static_cast<std::size_t>(i)], offsetToGround);
            } else if (config_.lightIkIters > 0) {
                // 把 retargeter 内部状态设置成 qCursor
                // 通过传入 ik iters 来控制 ik的量级  内部调用的是 重定向用的ik接口 只是 迭代次数少了
                retargeter.setQpos(qCursor);
                qSeed = retargeter.retargetLightIk(windowFrames[static_cast<std::size_t>(i)], offsetToGround, config_.lightIkIters);
            } else {
                qSeed = qCursor;
            }
            seeds.push_back(qSeed);
            qCursor = qSeed;
        }

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
            auto qOpt = solveQpWindow(qWin, tgtWin, refWin, qPrevPtr, pin);
            qCmd      = qOpt[static_cast<std::size_t>(pin)];
        }

        qCmd = commitOutputQpos(retargeter, std::move(qCmd));

        arrivalPrevTargets_ = commitTargets;
        arrivalQPrev_       = qCmd;
        arrivalHasPrev_     = true;
        qBuf_.push_back(qCmd);
        while (qBuf_.size() > static_cast<std::size_t>(std::max(config_.horizon, 8))) {
            qBuf_.pop_front();
        }
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

        const int Hneed = std::max(1, config_.horizon);
        // Fill phase: traditional GMR for the newest arrived frame (no pop). Keeps the robot
        // moving during the H-1 frame lookahead delay; seeds arrivalQPrev for the first QP.
        // When the buffer reaches horizon, subsequent steps switch to delayed QP (commit front).
        if (!flush && arrivalFillPending_ && static_cast<int>(arrivalBuf_.size()) < Hneed) {
            const auto t0            = Clock::now();
            arrivalFillPending_      = false;
            const HumanFrame& latest = arrivalBuf_.back();
            Eigen::VectorXd qCmd     = retargeter.retargetFrame(latest, offsetToGround);
            HumanFrame prepared      = retargeter.prepareRetargetInput(latest, offsetToGround);
            arrivalPrevTargets_      = batch_->targetsForPrepared(prepared);
            arrivalQPrev_            = qCmd;
            arrivalHasPrev_          = true;
            ++frameIndex_;
            qBuf_.push_back(qCmd);
            while (qBuf_.size() > static_cast<std::size_t>(std::max(config_.horizon, 8))) {
                qBuf_.pop_front();
            }
            lastFrameMs_ = elapsedMs(t0);
            return qCmd;
        }

        arrivalFillPending_ = false;
        const int Hn = flush ? static_cast<int>(arrivalBuf_.size()) : std::min(config_.horizon, static_cast<int>(arrivalBuf_.size()));
        std::vector<HumanFrame> window(arrivalBuf_.begin(), arrivalBuf_.begin() + Hn);
        Eigen::VectorXd qCmd = stepLookaheadWindow(window, retargeter, offsetToGround);
        arrivalBuf_.pop_front();
        return qCmd;
    }

}  // namespace gmr
