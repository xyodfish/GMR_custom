#pragma once

#include <string>

namespace gmr {

    enum class OnlineQpPreset {
        kDefault,
        kSmooth,
        kAntiSlip,
    };

    struct OnlineQpConfig {
        OnlineQpPreset preset = OnlineQpPreset::kDefault;
        int horizon           = 3;
        int sqpIters          = 3;
        int minFrames         = 2;
        double wVelocity      = 2.0;
        double wAcceleration  = 8.0;
        double wAnchor        = 1.0;
        double wGmr           = 0.35;
        double gnDamping      = 1e-2;
        double gnMaxStep      = 0.08;
        int lightIkIters      = 4;
        bool enableFootPenalties = true;
        double wFootHeight       = 40.0;
        double wFootSlip         = 900.0;
        double wFootIkAnchor     = 30.0;
        double wRootXyContact    = 20.0;
        double wContactJointAnchor = 40.0;
        double dqMax             = 4.0;
        bool useJointLimits      = true;
        bool useVelocityLimits   = true;
        // Keep committed hinge joints this many degrees away from hard limits (0 disables).
        double jointLimitMarginDeg = 0.0;
        bool useLookahead        = true;
        std::string qpBackend    = "daqp";
        bool finalizeContact     = true;
        int bootstrapGmrFrames   = 2;
        bool verbose             = false;

        static OnlineQpConfig fromPreset(OnlineQpPreset preset);
        static OnlineQpConfig fromPresetName(const std::string& name);
    };

}  // namespace gmr
