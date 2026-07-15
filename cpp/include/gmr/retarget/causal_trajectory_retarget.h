#pragma once

#include <memory>

#include <Eigen/Core>

#include "gmr/retarget/causal_trajectory_config.h"
#include "gmr/retarget/human_frame_types.h"
#include "gmr/retarget/retargeter.h"

namespace gmr {

    /// Causal (online) trajectory optimization: one frame in, one ``q_t`` out.
    ///
    /// Fast pipeline per frame (after bootstrap):
    ///   q_seed = light IK from q_{t-1}  →  L-BFGS / GN temporal refine vs q_{t-1}, q_{t-2}
    class CausalTrajectoryRetargeter {
       public:
        explicit CausalTrajectoryRetargeter(CausalTrajectoryConfig config = {});

        void reset();
        void setMotionFps(double fps);

        /// Process one human frame; returns optimized robot ``qpos`` for this timestep.
        Eigen::VectorXd retargetFrame(const HumanFrame& humanFrame, Retargeter& retargeter, bool offsetToGround = false);

        int frameIndex() const { return frameIndex_; }
        double lastFrameMs() const { return lastFrameMs_; }

       private:
        CausalRefineParams buildRefineParams() const;
        Eigen::VectorXd finalizeQpos(const Eigen::VectorXd& qpos, Retargeter& retargeter) const;

        CausalTrajectoryConfig config_;
        double motionFps_ = 30.0;

        bool hasHistory_ = false;
        Eigen::VectorXd qPrev_;
        Eigen::VectorXd qPrev2_;
        int frameIndex_ = 0;
        double lastFrameMs_ = 0.0;
    };

}  // namespace gmr
