#include "gmr/retarget/online_qp_config.h"

#include <stdexcept>

namespace gmr {

    OnlineQpConfig OnlineQpConfig::fromPreset(OnlineQpPreset preset) {
        OnlineQpConfig cfg;
        cfg.preset = preset;
        if (preset == OnlineQpPreset::kSmooth) {
            cfg.wVelocity     = 3.0;
            cfg.wAcceleration = 12.0;
            cfg.wGmr          = 0.25;
            cfg.wFootSlip     = 700.0;
            cfg.sqpIters      = 3;
            return cfg;
        }
        if (preset == OnlineQpPreset::kAntiSlip) {
            cfg.wVelocity          = 1.5;
            cfg.wAcceleration      = 6.0;
            cfg.wGmr               = 0.4;
            cfg.wFootSlip          = 2000.0;
            cfg.wFootHeight        = 60.0;
            cfg.wFootIkAnchor      = 40.0;
            cfg.sqpIters           = 3;
            cfg.finalizeContact    = false;
            return cfg;
        }
        return cfg;
    }

    OnlineQpConfig OnlineQpConfig::fromPresetName(const std::string& name) {
        if (name == "default") {
            return fromPreset(OnlineQpPreset::kDefault);
        }
        if (name == "smooth") {
            return fromPreset(OnlineQpPreset::kSmooth);
        }
        if (name == "anti_slip") {
            return fromPreset(OnlineQpPreset::kAntiSlip);
        }
        throw std::runtime_error("Unknown online QP preset: " + name + " (use default|smooth|anti_slip)");
    }

}  // namespace gmr
