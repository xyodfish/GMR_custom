#pragma once

namespace gmr {

    enum class CausalSolver {
        kNone,
        kGn,
        kLbfgs,
    };

    /// Parameters for single-frame causal refine (GN or L-BFGS).
    struct CausalRefineParams {
        CausalSolver solver = CausalSolver::kLbfgs;
        int gnSteps          = 3;
        double gnDamping     = 0.05;
        double gnMaxStep     = 0.08;
        double wVelocity     = 2.0;
        double wAcceleration = 10.0;
        double dt            = 1.0 / 30.0;
        bool smoothRootXyz   = false;
        bool enforceDqDdq    = true;
        double dqMax         = 8.0;
        double ddqMax        = 80.0;
        int fastOptIter      = 5;
        double optTol        = 1e-4;
    };

    /// Backward-compatible alias.
    using CausalGnParams = CausalRefineParams;

    /// Online causal TO (fast mode): light IK warm start + temporal refine.
    struct CausalTrajectoryConfig {
        CausalSolver solver  = CausalSolver::kLbfgs;
        int gnSteps          = 3;
        double gnDamping     = 0.05;
        double gnMaxStep     = 0.08;
        double wVelocity     = 2.0;
        double wAcceleration = 10.0;
        int lightIkWarmstartIters = 5;
        int fastOptIter      = 5;
        double optTol        = 1e-4;
        bool useGmrInit      = true;
        bool finalizeContact = true;
        bool smoothRootXyz   = false;
        bool enforceDqDdq    = true;
        double dqMax         = 8.0;
        double ddqMax        = 80.0;
    };

}  // namespace gmr
