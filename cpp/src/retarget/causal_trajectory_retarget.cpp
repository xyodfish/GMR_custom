#include "gmr/retarget/causal_trajectory_retarget.h"

#include <chrono>
#include <stdexcept>

namespace gmr {
    namespace {

        using Clock = std::chrono::steady_clock;

        double elapsedMs(const Clock::time_point& t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }

    }  // namespace

    CausalTrajectoryRetargeter::CausalTrajectoryRetargeter(CausalTrajectoryConfig config) : config_(std::move(config)) {}

    void CausalTrajectoryRetargeter::reset() {
        hasHistory_  = false;
        frameIndex_  = 0;
        lastFrameMs_ = 0.0;
        qPrev_.resize(0);
        qPrev2_.resize(0);
    }

    void CausalTrajectoryRetargeter::setMotionFps(double fps) {
        if (fps > 0.0) {
            motionFps_ = fps;
        }
    }

    CausalRefineParams CausalTrajectoryRetargeter::buildRefineParams() const {
        CausalRefineParams params;
        params.solver           = config_.solver;
        params.gnSteps          = config_.gnSteps;
        params.gnDamping        = config_.gnDamping;
        params.gnMaxStep        = config_.gnMaxStep;
        params.wVelocity        = config_.wVelocity;
        params.wAcceleration    = config_.wAcceleration;
        params.dt               = 1.0 / std::max(motionFps_, 1e-6);
        params.smoothRootXyz    = config_.smoothRootXyz;
        params.enforceDqDdq     = config_.enforceDqDdq;
        params.dqMax            = config_.dqMax;
        params.ddqMax           = config_.ddqMax;
        params.fastOptIter      = config_.fastOptIter;
        params.optTol           = config_.optTol;
        return params;
    }

    Eigen::VectorXd CausalTrajectoryRetargeter::finalizeQpos(const Eigen::VectorXd& qpos, Retargeter& retargeter) const {
        retargeter.setQpos(qpos);
        if (config_.finalizeContact) {
            retargeter.finalizeContact();
        }
        return retargeter.currentQpos();
    }

    Eigen::VectorXd CausalTrajectoryRetargeter::retargetFrame(const HumanFrame& humanFrame, Retargeter& retargeter,
                                                              bool offsetToGround) {
        const auto t0 = Clock::now();
        frameIndex_++;

        const HumanFrame prepared = retargeter.prepareRetargetInput(humanFrame, offsetToGround);
        const CausalRefineParams refineParams = buildRefineParams();

        Eigen::VectorXd qOut;

        if (!hasHistory_) {
            if (config_.useGmrInit) {
                qOut = retargeter.retargetFrame(humanFrame, offsetToGround);
            } else {
                retargeter.setQpos(retargeter.currentQpos());
                Eigen::VectorXd qInit = retargeter.retargetLightIk(humanFrame, offsetToGround, config_.lightIkWarmstartIters);
                const Eigen::VectorXd qSeed = qInit;
                qOut = retargeter.optimizeCausalRefine(prepared, qInit, qSeed, qSeed, refineParams);
                qOut = finalizeQpos(qOut, retargeter);
            }
        } else {
            retargeter.setQpos(qPrev_);
            Eigen::VectorXd qInit = retargeter.retargetLightIk(humanFrame, offsetToGround, config_.lightIkWarmstartIters);
            if (refineParams.solver == CausalSolver::kNone) {
                qOut = qInit;
            } else {
                qOut = retargeter.optimizeCausalRefine(prepared, qInit, qPrev_, qPrev2_, refineParams);
            }
            qOut = finalizeQpos(qOut, retargeter);
        }

        if (qPrev_.size() != qOut.size()) {
            qPrev_.resize(qOut.size());
            qPrev2_.resize(qOut.size());
        }
        qPrev2_     = qPrev_;
        qPrev_      = qOut;
        hasHistory_ = true;
        lastFrameMs_ = elapsedMs(t0);
        return qOut;
    }

}  // namespace gmr
