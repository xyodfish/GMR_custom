#pragma once

#include <filesystem>
#include <vector>

#include "gmr/retarget/contact_ground.h"

namespace gmr {

    enum class GnLineSearchMode {
        kBest,    ///< Evaluate all alphas, pick lowest cost (Python parity).
        kArmijo,  ///< Accept first alpha that decreases cost (faster).
    };

    struct BatchTrajectoryConfig {
        int windowSize              = 16;
        int windowStride            = 8;
        int gnSteps                 = 3;
        double gnDamping            = 0.1;
        double gnMaxStep            = 0.05;
        double wVelocity            = 2.0;
        double wAcceleration        = 10.0;
        double wAnchor              = 0.0;
        double windowAnchorWeight   = 2.0;
        bool useGmrInit             = true;
        bool finalizeContact        = true;
        bool enableFootPenalties    = true;
        double wFootHeight          = 50.0;
        double wFootSlip            = 2000.0;
        double wFootIkAnchor        = 200.0;
        double wRootXyContact       = 100.0;
        double wContactJointAnchor  = 400.0;
        double footContactMargin    = 0.02;
        // Keep committed hinge joints this many degrees away from hard limits (0 disables).
        double jointLimitMarginDeg  = 0.0;
        bool footContactFromRef     = true;
        bool smoothRootXyz          = false;
        std::vector<double> gnLineSearchAlphas = {1.0, 0.5, 0.25, 0.125};
        GnLineSearchMode gnLineSearchMode      = GnLineSearchMode::kBest;
        bool useBandedSolver                   = false;
        std::filesystem::path qInitJsonPath;
        bool parallelBootstrap                 = false;
        bool parallelFinalize                  = false;
        int parallelThreads                    = 0;  ///< 0 = OpenMP default
        bool verbose                = false;
    };

    struct BatchTrajectoryProfile {
        double prepareMs    = 0.0;
        double bootstrapMs  = 0.0;
        double optimizeMs   = 0.0;
        double finalizeMs   = 0.0;
        double totalMs      = 0.0;
        int nFrames         = 0;

        double msPerFrame() const {
            return nFrames > 0 ? totalMs / static_cast<double>(nFrames) : 0.0;
        }
        double effectiveFps() const {
            return totalMs > 0.0 ? 1000.0 * static_cast<double>(nFrames) / totalMs : 0.0;
        }
    };

}  // namespace gmr
