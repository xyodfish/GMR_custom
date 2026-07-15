#pragma once

#include <vector>

#include <mujoco/mujoco.h>
#include <pinocchio/spatial/explog.hpp>

#include <Eigen/Core>

namespace gmr {
    namespace mujoco_task_internal {

        /// Mink-style FrameTask: body-frame SE3 log error and 6 x nv Jacobian (unweighted).
        inline void evalFrameTask(const mjModel* model, mjData* data, int bodyId, const Eigen::Vector3d& targetPos,
                                  const Eigen::Quaterniond& targetRot, Eigen::Matrix<double, 6, 1>* error,
                                  Eigen::Matrix<double, 6, Eigen::Dynamic>* jacobianNv, std::vector<double>* jacpBuf,
                                  std::vector<double>* jacrBuf) {
            std::fill(jacpBuf->begin(), jacpBuf->end(), 0.0);
            std::fill(jacrBuf->begin(), jacrBuf->end(), 0.0);
            mj_jacBody(model, data, jacpBuf->data(), jacrBuf->data(), bodyId);

            const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> JpWorld(jacpBuf->data(), 3,
                                                                                                        model->nv);
            const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> JrWorld(jacrBuf->data(), 3,
                                                                                                        model->nv);

            const double* xpos  = &data->xpos[3 * bodyId];
            const double* xquat = &data->xquat[4 * bodyId];
            const Eigen::Vector3d currPos(xpos[0], xpos[1], xpos[2]);
            Eigen::Quaterniond currRot(xquat[0], xquat[1], xquat[2], xquat[3]);
            currRot.normalize();
            const Eigen::Matrix3d Rwb = currRot.toRotationMatrix();
            const Eigen::Matrix3d Rbw = Rwb.transpose();

            Eigen::MatrixXd Jlocal(6, model->nv);
            Jlocal.topRows(3)    = Rbw * JpWorld;
            Jlocal.bottomRows(3) = Rbw * JrWorld;

            Eigen::Quaterniond tgtRot = targetRot;
            tgtRot.normalize();
            const pinocchio::SE3 T_wb(Rwb, currPos);
            const pinocchio::SE3 T_wt(tgtRot.toRotationMatrix(), targetPos);
            const pinocchio::SE3 T_bt               = T_wb.inverse() * T_wt;
            const pinocchio::SE3 T_tb               = T_wt.inverse() * T_wb;
            const Eigen::Matrix<double, 6, 1> err6 = pinocchio::log6(T_bt).toVector();
            const Eigen::Matrix<double, 6, 6> jlog = pinocchio::Jlog6(T_tb);
            const Eigen::MatrixXd Jtask            = -jlog * Jlocal;

            *error       = err6;
            *jacobianNv  = Jtask;
        }

        /// Map tangent-space gradient (nv) to qpos gradient (nq), MuJoCo free-joint convention.
        inline void scatterNvGradToQpos(const mjModel* model, const double* qpos, const Eigen::VectorXd& gradNv,
                                        Eigen::VectorXd* gradQ) {
            gradQ->setZero(model->nq);
            for (int v = 0; v < model->nv; ++v) {
                const int j    = model->dof_jntid[v];
                const int jtyp = model->jnt_type[j];
                const int qadr = model->jnt_qposadr[j];
                if (jtyp == mjJNT_HINGE || jtyp == mjJNT_SLIDE) {
                    (*gradQ)[qadr] += gradNv[v];
                } else if (jtyp == mjJNT_FREE) {
                    if (v < 3) {
                        (*gradQ)[qadr + v] += gradNv[v];
                    }
                }
            }

            // Free-joint quaternion: map angular-velocity components through mj_integratePos.
            for (int j = 0; j < model->njnt; ++j) {
                if (model->jnt_type[j] != mjJNT_FREE) {
                    continue;
                }
                const int qadr = model->jnt_qposadr[j];
                const int vadr = model->jnt_dofadr[j];
                if (vadr + 5 >= model->nv) {
                    continue;
                }

                constexpr double eps = 1e-7;
                std::vector<double> v(model->nv, 0.0);
                std::vector<double> qplus(model->nq, 0.0);
                std::vector<double> qminus(model->nq, 0.0);
                for (int r = 0; r < 3; ++r) {
                    std::fill(v.begin(), v.end(), 0.0);
                    v[vadr + 3 + r] = eps;
                    mju_copy(qplus.data(), qpos, model->nq);
                    mj_integratePos(model, qplus.data(), v.data(), 1.0);
                    v[vadr + 3 + r] = -eps;
                    mju_copy(qminus.data(), qpos, model->nq);
                    mj_integratePos(model, qminus.data(), v.data(), 1.0);
                    for (int k = 0; k < 4; ++k) {
                        const double jac = (qplus[qadr + 3 + k] - qminus[qadr + 3 + k]) / (2.0 * eps);
                        (*gradQ)[qadr + 3 + k] += jac * gradNv[vadr + 3 + r];
                    }
                }
            }
        }

    }  // namespace mujoco_task_internal
}  // namespace gmr
