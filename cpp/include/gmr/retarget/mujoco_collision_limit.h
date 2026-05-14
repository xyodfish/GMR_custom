#pragma once

#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>

#include "gmr/retarget/ik_config.h"

namespace gmr {

using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /// Builds geom pairs and fills QP rows matching `mink.CollisionAvoidanceLimit` (inequality on Δq).
    class MujocoCollisionLimit {
       public:
        /// If `cfg.enabled` is false or no valid pairs remain, `maxRows()` is zero.
        MujocoCollisionLimit(const mjModel* model, const CollisionAvoidanceConfig& cfg);

        int maxRows() const { return static_cast<int>(geomIdPairs_.size()); }

        /// For each geom pair, write one row into `CI` / bounds starting at `rowOffset`.
        /// `inequalityScale`: use `1.0` when QP variable is **Δq** (same as mink). Use `1.0/dt` when variable is **qvel**
        /// (legacy backend).
        void fillRows(mjData* data, double dt, double inequalityScale, Eigen::Ref<RowMajorMatrixXd> CI,
                      Eigen::Ref<Eigen::VectorXd> ciLb, Eigen::Ref<Eigen::VectorXd> ciUb, int rowOffset) const;

       private:
        const mjModel* model_ = nullptr;
        CollisionAvoidanceConfig cfg_{};
        std::vector<std::pair<int, int>> geomIdPairs_;
        int nv_ = 0;
        mutable std::vector<mjtNum> jac1_;
        mutable std::vector<mjtNum> jac2_;
    };

}  // namespace gmr
