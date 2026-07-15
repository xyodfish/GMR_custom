#pragma once

#include <algorithm>
#include <vector>

#include <mujoco/mujoco.h>

#include <Eigen/Core>

namespace gmr {
    namespace mink_limits {

        /// Mink ``ConfigurationLimit``: ``mj_differentiatePos`` + gain on limited hinges/slides.
        inline void applyConfigurationLimit(const mjModel* model, const double* qpos, double gain, Eigen::VectorXd* ciLb,
                                            Eigen::VectorXd* ciUb) {
            if (model == nullptr || qpos == nullptr || ciLb == nullptr || ciUb == nullptr) {
                return;
            }

            std::vector<double> qUpper(static_cast<std::size_t>(model->nq));
            std::vector<double> qLower(static_cast<std::size_t>(model->nq));
            std::fill(qUpper.begin(), qUpper.end(), mjMAXVAL);
            std::fill(qLower.begin(), qLower.end(), -mjMAXVAL);

            for (int j = 0; j < model->njnt; ++j) {
                const int jointType = model->jnt_type[j];
                if (jointType == mjJNT_FREE || model->jnt_limited[j] <= 0) {
                    continue;
                }
                if (jointType != mjJNT_HINGE && jointType != mjJNT_SLIDE) {
                    continue;
                }
                const int qadr    = model->jnt_qposadr[j];
                qLower[static_cast<std::size_t>(qadr)] = model->jnt_range[2 * j + 0];
                qUpper[static_cast<std::size_t>(qadr)] = model->jnt_range[2 * j + 1];
            }

            std::vector<double> deltaQMax(static_cast<std::size_t>(model->nv), 0.0);
            std::vector<double> deltaQMin(static_cast<std::size_t>(model->nv), 0.0);
            mj_differentiatePos(model, deltaQMax.data(), 1.0, qpos, qUpper.data());
            mj_differentiatePos(model, deltaQMin.data(), 1.0, qLower.data(), qpos);

            for (int j = 0; j < model->njnt; ++j) {
                const int jointType = model->jnt_type[j];
                if (jointType == mjJNT_FREE || model->jnt_limited[j] <= 0) {
                    continue;
                }
                if (jointType != mjJNT_HINGE && jointType != mjJNT_SLIDE) {
                    continue;
                }
                const int vadr = model->jnt_dofadr[j];
                const double pMax = gain * deltaQMax[static_cast<std::size_t>(vadr)];
                const double pMin = gain * deltaQMin[static_cast<std::size_t>(vadr)];
                (*ciUb)[vadr]     = std::min((*ciUb)[vadr], pMax);
                (*ciLb)[vadr]     = std::max((*ciLb)[vadr], -pMin);
            }
        }

    }  // namespace mink_limits
}  // namespace gmr
