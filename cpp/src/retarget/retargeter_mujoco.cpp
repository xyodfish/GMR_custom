#include "gmr/retarget/retargeter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <mujoco/mujoco.h>

#include <pinocchio/spatial/explog.hpp>

#include "gmr/retarget/mujoco_collision_limit.h"
#include "gmr/retarget/contact_ground.h"
#include "gmr/solver/qp_solver.h"
#include "retargeter_internal_utils.h"
#include "causal_lbfgs_b.h"
#include "mujoco_task_kinematics.h"

#include "glog/logging.h"

namespace gmr {

    struct MujocoTaskRuntime {
        int bodyId = -1;
        std::string humanBodyName;
        double posWeight             = 0.0;
        double rotWeight             = 0.0;
        Eigen::Vector3d posOffset    = Eigen::Vector3d::Zero();
        Eigen::Quaterniond rotOffset = Eigen::Quaterniond::Identity();

        Eigen::Vector3d targetPos    = Eigen::Vector3d::Zero();
        Eigen::Quaterniond targetRot = Eigen::Quaterniond::Identity();
    };

    struct MujocoRetargetBackend::Impl {
        struct ModelDeleter {
            void operator()(mjModel* p) const {
                if (p != nullptr) {
                    mj_deleteModel(p);
                }
            }
        };

        struct DataDeleter {
            void operator()(mjData* p) const {
                if (p != nullptr) {
                    mj_deleteData(p);
                }
            }
        };

        std::unique_ptr<mjModel, ModelDeleter> model;
        std::unique_ptr<mjData, DataDeleter> data;

        IkConfig ikConfig;
        RetargetOptions options;

        std::vector<MujocoTaskRuntime> tasks1;
        std::vector<MujocoTaskRuntime> tasks2;
        std::unordered_map<std::string, Eigen::Vector3d> table1PosOffsets;
        std::unordered_map<std::string, Eigen::Quaterniond> table1RotOffsets;

        bool hasRootFreeFlyer = false;
        std::vector<ScalarJointCoordinate> scalarJointCoordinates;
        Eigen::VectorXd qpos;
        Eigen::VectorXd qvel;

        std::unique_ptr<MujocoCollisionLimit> collisionLimit;
        std::unique_ptr<ContactGroundPipeline> contactGround;

        Impl(const std::filesystem::path& robotModelPath, IkConfig ikConfigIn, RetargetOptions optionsIn)
            : ikConfig(std::move(ikConfigIn)), options(std::move(optionsIn)) {
            const std::string extension = retarget_internal::toLower(robotModelPath.extension().string());
            if (extension != ".xml" && extension != ".mjcf") {
                throw std::runtime_error("MuJoCo retarget backend requires XML/MJCF model: " + robotModelPath.string());
            }

            std::array<char, 1024> error{};
            mjModel* rawModel = mj_loadXML(robotModelPath.string().c_str(), nullptr, error.data(), error.size());
            if (rawModel == nullptr) {
                throw std::runtime_error("Failed to load MuJoCo model: " + robotModelPath.string() + " error=" + std::string(error.data()));
            }

            model.reset(rawModel);
            data.reset(mj_makeData(model.get()));
            if (!data) {
                throw std::runtime_error("Failed to allocate MuJoCo data.");
            }

            hasRootFreeFlyer = false;
            for (int j = 0; j < model->njnt; ++j) {
                if (model->jnt_type[j] == mjJNT_FREE) {
                    hasRootFreeFlyer = true;
                    break;
                }
            }

            for (int j = 0; j < model->njnt; ++j) {
                const int jointType = model->jnt_type[j];
                if (jointType == mjJNT_HINGE || jointType == mjJNT_SLIDE) {
                    const int qadr        = model->jnt_qposadr[j];
                    const int vadr        = model->jnt_dofadr[j];
                    const char* jointName = mj_id2name(model.get(), mjOBJ_JOINT, j);
                    scalarJointCoordinates.push_back(ScalarJointCoordinate{qadr, vadr, jointName == nullptr ? "" : jointName});
                }
            }

            std::sort(scalarJointCoordinates.begin(), scalarJointCoordinates.end(),
                      [](const ScalarJointCoordinate& a, const ScalarJointCoordinate& b) { return a.qIndex < b.qIndex; });

            auto buildTasks = [this](const std::vector<IkTaskEntry>& src, std::vector<MujocoTaskRuntime>* dst) {
                dst->clear();
                dst->reserve(src.size());
                for (const auto& entry : src) {
                    const int bodyId = mj_name2id(model.get(), mjOBJ_BODY, entry.robotBodyName.c_str());
                    if (bodyId < 0) {
                        LOG(WARNING) << "Body not found in MuJoCo model, skip IK task: " << entry.robotBodyName;
                        continue;
                    }

                    MujocoTaskRuntime task;
                    task.bodyId        = bodyId;
                    task.humanBodyName = entry.humanBodyName;
                    task.posWeight     = entry.posWeight;
                    task.rotWeight     = entry.rotWeight;
                    task.posOffset     = entry.posOffset;
                    task.rotOffset     = entry.rotOffset;
                    dst->push_back(std::move(task));
                }
            };

            buildTasks(ikConfig.tasksTable1, &tasks1);
            buildTasks(ikConfig.tasksTable2, &tasks2);

            for (const auto& task : tasks1) {
                table1PosOffsets[task.humanBodyName] = task.posOffset - Eigen::Vector3d(0.0, 0.0, ikConfig.groundHeight);
                table1RotOffsets[task.humanBodyName] = task.rotOffset;
            }

            if (ikConfig.collisionAvoidance.enabled) {
                collisionLimit = std::make_unique<MujocoCollisionLimit>(model.get(), ikConfig.collisionAvoidance);
            }

            if (options.contactGround.enabled) {
                contactGround = std::make_unique<ContactGroundPipeline>(options.contactGround, model.get(), options.motionFps);
            }

            mju_copy(data->qpos, model->qpos0, model->nq);
            mju_zero(data->qvel, model->nv);
            mj_forward(model.get(), data.get());

            qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq);
            qvel = Eigen::VectorXd::Zero(model->nv);
        }

        void syncQposFromData() { qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq); }

        HumanFrame prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
            const bool useOffsetToGround = offsetToGround && !(contactGround && contactGround->enabled());
            return retarget_internal::scaleAndOffsetHumanFrameImpl(humanFrame, ikConfig, table1PosOffsets, table1RotOffsets,
                                                                   useOffsetToGround);
        }

        HumanFrame applyContactGround(const HumanFrame& prepared) const {
            if (contactGround && contactGround->enabled()) {
                return contactGround->processHumanFrame(prepared);
            }
            return prepared;
        }

        void finalizeRobotState() {
            if (contactGround && contactGround->enabled()) {
                contactGround->fixRobotPenetration(model.get(), data.get());
                syncQposFromData();
            }
        }

        void updateTaskTargets(const HumanFrame& frame) {
            auto fill = [&frame](std::vector<MujocoTaskRuntime>* tasks) {
                for (auto& task : *tasks) {
                    auto it = frame.find(task.humanBodyName);
                    if (it == frame.end()) {
                        continue;
                    }
                    task.targetPos = it->second.position;
                    task.targetRot = it->second.orientation;
                }
            };

            fill(&tasks1);
            fill(&tasks2);
        }

        double computeTaskError(const std::vector<MujocoTaskRuntime>& tasks) const {
            double sqErr = 0.0;
            for (const auto& task : tasks) {
                const double* xpos  = &data->xpos[3 * task.bodyId];
                const double* xquat = &data->xquat[4 * task.bodyId];
                const Eigen::Vector3d currPos(xpos[0], xpos[1], xpos[2]);
                Eigen::Quaterniond currRot(xquat[0], xquat[1], xquat[2], xquat[3]);
                currRot.normalize();

                const Eigen::Vector3d posErr = task.targetPos - currPos;
                const Eigen::Vector3d rotErr = retarget_internal::computeOrientationErrorWorld(currRot, task.targetRot);
                sqErr += posErr.squaredNorm() + rotErr.squaredNorm();
            }

            return std::sqrt(sqErr);
        }

        void solveTaskSet(const std::vector<MujocoTaskRuntime>& tasks) {
            if (tasks.empty()) {
                return;
            }

            const int nv    = model->nv;
            const double dt = options.integrationTimestep;
            if (dt <= 1e-12) {
                throw std::runtime_error("integrationTimestep must be positive.");
            }

            double currError = computeTaskError(tasks);
            solver::QPSolver solver;
            const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nv, nv);

            std::vector<mjtNum> jacp(3 * nv);
            std::vector<mjtNum> jacr(3 * nv);

            const int nCollisionRows = (collisionLimit != nullptr) ? collisionLimit->maxRows() : 0;

            for (int iter = 0; iter < options.maxIterations; ++iter) {
                mj_forward(model.get(), data.get());

                solver::QPData qp;
                qp.reset(nv, nv + nCollisionRows);

                qp.CI.topRows(nv).setIdentity();
                if (nCollisionRows > 0) {
                    qp.CI.bottomRows(nCollisionRows).setZero();
                }
                qp.ciLb.setConstant(-1e9);
                qp.ciUb.setConstant(1e9);

                if (options.useVelocityLimit) {
                    qp.ciLb.head(nv).setConstant(-options.velocityLimit * dt);
                    qp.ciUb.head(nv).setConstant(options.velocityLimit * dt);
                }

                for (int j = 0; j < model->njnt; ++j) {
                    const int jointType = model->jnt_type[j];
                    if (jointType != mjJNT_HINGE && jointType != mjJNT_SLIDE) {
                        continue;
                    }
                    if (model->jnt_limited[j] <= 0) {
                        continue;
                    }

                    const int qadr    = model->jnt_qposadr[j];
                    const int vadr    = model->jnt_dofadr[j];
                    const double qmin = model->jnt_range[2 * j + 0];
                    const double qmax = model->jnt_range[2 * j + 1];

                    qp.ciLb[vadr] = std::max(qp.ciLb[vadr], qmin - data->qpos[qadr]);
                    qp.ciUb[vadr] = std::min(qp.ciUb[vadr], qmax - data->qpos[qadr]);
                }

                for (const auto& task : tasks) {
                    std::fill(jacp.begin(), jacp.end(), 0.0);
                    std::fill(jacr.begin(), jacr.end(), 0.0);
                    mj_jacBody(model.get(), data.get(), jacp.data(), jacr.data(), task.bodyId);

                    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> JpWorld(jacp.data(), 3, nv);
                    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> JrWorld(jacr.data(), 3, nv);

                    const double* xpos  = &data->xpos[3 * task.bodyId];
                    const double* xquat = &data->xquat[4 * task.bodyId];
                    const Eigen::Vector3d currPos(xpos[0], xpos[1], xpos[2]);
                    Eigen::Quaterniond currRot(xquat[0], xquat[1], xquat[2], xquat[3]);
                    currRot.normalize();
                    const Eigen::Matrix3d Rwb = currRot.toRotationMatrix();
                    const Eigen::Matrix3d Rbw = Rwb.transpose();

                    Eigen::MatrixXd Jlocal(6, nv);
                    Jlocal.topRows(3)    = Rbw * JpWorld;
                    Jlocal.bottomRows(3) = Rbw * JrWorld;

                    Eigen::Quaterniond targetRot = task.targetRot;
                    targetRot.normalize();
                    const pinocchio::SE3 T_wb(Rwb, currPos);
                    const pinocchio::SE3 T_wt(targetRot.toRotationMatrix(), task.targetPos);
                    const pinocchio::SE3 T_bt               = T_wb.inverse() * T_wt;
                    const pinocchio::SE3 T_tb               = T_wt.inverse() * T_wb;
                    const Eigen::Matrix<double, 6, 1> error = pinocchio::log6(T_bt).toVector();
                    const Eigen::Matrix<double, 6, 6> jlog  = pinocchio::Jlog6(T_tb);
                    const Eigen::MatrixXd Jtask             = -jlog * Jlocal;

                    Eigen::MatrixXd weightedJ = Jtask;
                    weightedJ.topRows(3) *= task.posWeight;
                    weightedJ.bottomRows(3) *= task.rotWeight;

                    Eigen::Matrix<double, 6, 1> weightedError = -error;
                    weightedError.head<3>() *= task.posWeight;
                    weightedError.tail<3>() *= task.rotWeight;

                    const double lmMu = weightedError.squaredNorm();
                    qp.H.noalias() += weightedJ.transpose() * weightedJ + lmMu * I;
                    qp.g.noalias() += -(weightedError.transpose() * weightedJ).transpose();
                }

                qp.H.diagonal().array() += options.damping;

                if (nCollisionRows > 0 && collisionLimit != nullptr) {
                    collisionLimit->fillRows(data.get(), dt, 1.0, qp.CI, qp.ciLb, qp.ciUb, nv);
                }

                const solver::QPOutput& out = solver.solve(qp);
                if (out.status != solver::QPStatus::kOptimal) {
                    throw std::runtime_error("QP solver failed while retargeting.");
                }

                const Eigen::VectorXd deltaQ = out.x;
                qvel                         = deltaQ / dt;

                mj_integratePos(model.get(), data->qpos, qvel.data(), dt);
                for (int j = 0; j < model->njnt; ++j) {
                    const int jointType = model->jnt_type[j];
                    if ((jointType == mjJNT_HINGE || jointType == mjJNT_SLIDE) && model->jnt_limited[j] > 0) {
                        const int qadr    = model->jnt_qposadr[j];
                        const double qmin = model->jnt_range[2 * j + 0];
                        const double qmax = model->jnt_range[2 * j + 1];
                        data->qpos[qadr]  = std::min(std::max(data->qpos[qadr], qmin), qmax);
                    }
                }
                mju_copy(data->qvel, qvel.data(), model->nv);
                mj_forward(model.get(), data.get());
                syncQposFromData();

                const double nextError = computeTaskError(tasks);
                if (currError - nextError <= options.progressThreshold) {
                    break;
                }
                currError = nextError;
            }
        }

        void clipHingeQpos(Eigen::VectorXd& q) const {
            for (int j = 0; j < model->njnt; ++j) {
                const int jtype = model->jnt_type[j];
                if ((jtype == mjJNT_HINGE || jtype == mjJNT_SLIDE) && model->jnt_limited[j] > 0) {
                    const int qadr    = model->jnt_qposadr[j];
                    const double qmin = model->jnt_range[2 * j + 0];
                    const double qmax = model->jnt_range[2 * j + 1];
                    q[qadr]           = std::min(std::max(q[qadr], qmin), qmax);
                }
            }
        }

        void buildCausalOptIndices(bool smoothRootXyz, std::vector<int>* optVidx, std::vector<int>* smoothQidx) const {
            optVidx->clear();
            smoothQidx->clear();

            if (model->njnt > 0 && model->jnt_type[0] == mjJNT_FREE) {
                const int nvFree = std::min(6, static_cast<int>(model->nv));
                for (int v = 0; v < nvFree; ++v) {
                    optVidx->push_back(v);
                }
                const int nSmoothRoot = std::min(3, static_cast<int>(model->nq));
                for (int q = 0; q < nSmoothRoot; ++q) {
                    smoothQidx->push_back(q);
                }
            }

            for (int j = 0; j < model->njnt; ++j) {
                const int jtype = model->jnt_type[j];
                if (jtype != mjJNT_HINGE && jtype != mjJNT_SLIDE) {
                    continue;
                }
                optVidx->push_back(model->jnt_dofadr[j]);
                smoothQidx->push_back(model->jnt_qposadr[j]);
            }

            if (!smoothRootXyz) {
                smoothQidx->erase(std::remove_if(smoothQidx->begin(), smoothQidx->end(), [](int q) { return q < 3; }),
                                  smoothQidx->end());
            }
        }

        void projectDqDdq(Eigen::VectorXd& q, const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                           const std::vector<int>& smoothQidx, const CausalRefineParams& params) const {
            if (!params.enforceDqDdq || smoothQidx.empty()) {
                return;
            }

            const double dt  = std::max(params.dt, 1e-9);
            const double dt2 = dt * dt;

            for (int qadr : smoothQidx) {
                double dqLim  = params.dqMax;
                double ddqLim = params.ddqMax;
                if (qadr < 3) {
                    dqLim  = std::max(params.dqMax, 2.0);
                    ddqLim = std::max(params.ddqMax, 20.0);
                }

                double dqDelta = q[qadr] - qPrev[qadr];
                dqDelta        = std::min(std::max(dqDelta, -dqLim * dt), dqLim * dt);
                q[qadr]        = qPrev[qadr] + dqDelta;

                const double accTarget = 2.0 * qPrev[qadr] - qPrev2[qadr];
                double accDelta        = q[qadr] - accTarget;
                accDelta               = std::min(std::max(accDelta, -ddqLim * dt2), ddqLim * dt2);
                q[qadr]                = accTarget + accDelta;
            }
        }

        void accumulateGnTasks(const std::vector<MujocoTaskRuntime>& tasks, const std::vector<int>& optVidx,
                               Eigen::MatrixXd& H, Eigen::VectorXd& g) const {
            const int m = static_cast<int>(optVidx.size());
            std::vector<double> jacp(3 * model->nv, 0.0);
            std::vector<double> jacr(3 * model->nv, 0.0);
            Eigen::Matrix<double, 6, 1> err6;
            Eigen::Matrix<double, 6, Eigen::Dynamic> Jnv;

            for (const auto& task : tasks) {
                mujoco_task_internal::evalFrameTask(model.get(), data.get(), task.bodyId, task.targetPos, task.targetRot,
                                                    &err6, &Jnv, &jacp, &jacr);

                Eigen::Matrix<double, 6, Eigen::Dynamic> Jopt(6, m);
                Eigen::Matrix<double, 6, 1> weightedError = -err6;
                for (int col = 0; col < m; ++col) {
                    Jopt.col(col) = Jnv.col(optVidx[col]);
                }
                Jopt.topRows(3) *= task.posWeight;
                Jopt.bottomRows(3) *= task.rotWeight;
                weightedError.head<3>() *= task.posWeight;
                weightedError.tail<3>() *= task.rotWeight;

                H.noalias() += Jopt.transpose() * Jopt;
                g.noalias() += Jopt.transpose() * weightedError;
            }
        }

        void accumulateGnSmoothness(Eigen::MatrixXd& H, Eigen::VectorXd& g, const Eigen::VectorXd& q,
                                    const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                                    const std::vector<int>& smoothQidx, const std::vector<int>& optVidx, double wV,
                                    double wA) const {
            for (std::size_t k = 0; k < smoothQidx.size(); ++k) {
                const int qadr = smoothQidx[k];
                int vi         = -1;
                for (std::size_t i = 0; i < optVidx.size(); ++i) {
                    const int v = optVidx[i];
                    const int j = model->dof_jntid[v];
                    if (model->jnt_qposadr[j] == qadr) {
                        vi = static_cast<int>(i);
                        break;
                    }
                }
                if (vi < 0) {
                    continue;
                }

                if (wV > 0.0) {
                    const double e = q[qadr] - qPrev[qadr];
                    H(vi, vi) += wV;
                    g[vi] += wV * e;
                }
                if (wA > 0.0) {
                    const double accTarget = 2.0 * qPrev[qadr] - qPrev2[qadr];
                    const double e         = q[qadr] - accTarget;
                    H(vi, vi) += wA;
                    g[vi] += wA * e;
                }
            }
        }

        void accumulateMinkTrackingGrad(const std::vector<MujocoTaskRuntime>& tasks, Eigen::VectorXd& gradNv,
                                        double* trackingCost) const {
            std::vector<double> jacp(3 * model->nv, 0.0);
            std::vector<double> jacr(3 * model->nv, 0.0);
            Eigen::Matrix<double, 6, 1> err6;
            Eigen::Matrix<double, 6, Eigen::Dynamic> Jnv;

            for (const auto& task : tasks) {
                mujoco_task_internal::evalFrameTask(model.get(), data.get(), task.bodyId, task.targetPos, task.targetRot,
                                                    &err6, &Jnv, &jacp, &jacr);

                Eigen::Matrix<double, 6, 1> weightedErr = err6;
                Eigen::Matrix<double, 6, Eigen::Dynamic> weightedJ = Jnv;
                weightedErr.head<3>() *= task.posWeight;
                weightedErr.tail<3>() *= task.rotWeight;
                weightedJ.topRows(3) *= task.posWeight;
                weightedJ.bottomRows(3) *= task.rotWeight;

                *trackingCost += weightedErr.squaredNorm();
                gradNv.noalias() += 2.0 * weightedJ.transpose() * weightedErr;
            }
        }

        void computeLbfgsCostAndGrad(const Eigen::VectorXd& q, const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                                     double wV, double wA, double* cost, Eigen::VectorXd* gradQ) const {
            mju_copy(data->qpos, q.data(), model->nq);
            mj_forward(model.get(), data.get());

            double trackingCost = 0.0;
            Eigen::VectorXd gradNv = Eigen::VectorXd::Zero(model->nv);
            if (ikConfig.useTable1) {
                accumulateMinkTrackingGrad(tasks1, gradNv, &trackingCost);
            }
            if (ikConfig.useTable2) {
                accumulateMinkTrackingGrad(tasks2, gradNv, &trackingCost);
            }

            *cost = trackingCost;
            mujoco_task_internal::scatterNvGradToQpos(model.get(), data->qpos, gradNv, gradQ);

            if (wV > 0.0) {
                const Eigen::VectorXd delta = q - qPrev;
                *cost += wV * delta.squaredNorm();
                *gradQ += 2.0 * wV * delta;
            }
            if (wA > 0.0) {
                const Eigen::VectorXd acc = q - 2.0 * qPrev + qPrev2;
                *cost += wA * acc.squaredNorm();
                *gradQ += 2.0 * wA * acc;
            }
        }

        double computeTrackingCostSq(const std::vector<MujocoTaskRuntime>& tasks) const {
            double cost = 0.0;
            std::vector<double> jacp(3 * model->nv, 0.0);
            std::vector<double> jacr(3 * model->nv, 0.0);
            Eigen::Matrix<double, 6, 1> err6;
            Eigen::Matrix<double, 6, Eigen::Dynamic> Jnv;

            for (const auto& task : tasks) {
                mujoco_task_internal::evalFrameTask(model.get(), data.get(), task.bodyId, task.targetPos, task.targetRot,
                                                    &err6, &Jnv, &jacp, &jacr);
                Eigen::Matrix<double, 6, 1> weightedErr = err6;
                weightedErr.head<3>() *= task.posWeight;
                weightedErr.tail<3>() *= task.rotWeight;
                cost += weightedErr.squaredNorm();
            }
            return cost;
        }

        double computeTrackingCostSqAll() const {
            double cost = 0.0;
            if (ikConfig.useTable1) {
                cost += computeTrackingCostSq(tasks1);
            }
            if (ikConfig.useTable2) {
                cost += computeTrackingCostSq(tasks2);
            }
            return cost;
        }

        causal_lbfgs::BoxBounds buildQposBounds() const {
            causal_lbfgs::BoxBounds bounds;
            bounds.lower = Eigen::VectorXd::Constant(model->nq, -std::numeric_limits<double>::infinity());
            bounds.upper = Eigen::VectorXd::Constant(model->nq, std::numeric_limits<double>::infinity());
            for (int j = 0; j < model->njnt; ++j) {
                if (model->jnt_limited[j] <= 0) {
                    continue;
                }
                const int jtype = model->jnt_type[j];
                if (jtype != mjJNT_HINGE && jtype != mjJNT_SLIDE) {
                    continue;
                }
                const int qadr    = model->jnt_qposadr[j];
                bounds.lower[qadr] = model->jnt_range[2 * j + 0];
                bounds.upper[qadr] = model->jnt_range[2 * j + 1];
            }
            return bounds;
        }

        double computeLbfgsObjective(const Eigen::VectorXd& q, const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                                     double wV, double wA) const {
            mju_copy(data->qpos, q.data(), model->nq);
            mj_forward(model.get(), data.get());

            double cost = computeTrackingCostSqAll();
            if (wV > 0.0) {
                const Eigen::VectorXd delta = q - qPrev;
                cost += wV * delta.squaredNorm();
            }
            if (wA > 0.0) {
                const Eigen::VectorXd acc = q - 2.0 * qPrev + qPrev2;
                cost += wA * acc.squaredNorm();
            }
            return cost;
        }

        Eigen::VectorXd optimizeCausalGnImpl(const HumanFrame& preparedHuman, const Eigen::VectorXd& qInit,
                                             const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                                             const CausalRefineParams& params) {
            updateTaskTargets(preparedHuman);

            std::vector<int> optVidx;
            std::vector<int> smoothQidx;
            buildCausalOptIndices(params.smoothRootXyz, &optVidx, &smoothQidx);
            const int m = static_cast<int>(optVidx.size());
            if (m <= 0) {
                return qInit;
            }

            Eigen::VectorXd q = qInit;
            const double dt   = std::max(params.dt, 1e-9);
            const double dt2  = dt * dt;
            const double wV   = params.wVelocity / std::max(dt2, 1e-12);
            const double wA   = params.wAcceleration / std::max(dt2 * dt2, 1e-12);

            std::vector<double> dq(model->nv, 0.0);

            for (int step = 0; step < params.gnSteps; ++step) {
                mju_copy(data->qpos, q.data(), model->nq);
                mj_forward(model.get(), data.get());

                Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, m);
                Eigen::VectorXd g = Eigen::VectorXd::Zero(m);

                if (ikConfig.useTable1) {
                    accumulateGnTasks(tasks1, optVidx, H, g);
                }
                if (ikConfig.useTable2) {
                    accumulateGnTasks(tasks2, optVidx, H, g);
                }
                accumulateGnSmoothness(H, g, q, qPrev, qPrev2, smoothQidx, optVidx, wV, wA);

                Eigen::MatrixXd Hreg = H + params.gnDamping * Eigen::MatrixXd::Identity(m, m);
                Eigen::VectorXd dqSub = Hreg.ldlt().solve(g);
                dqSub                 = dqSub.cwiseMax(-params.gnMaxStep).cwiseMin(params.gnMaxStep);

                std::fill(dq.begin(), dq.end(), 0.0);
                for (int vi = 0; vi < m; ++vi) {
                    dq[optVidx[vi]] = -dqSub[vi];
                }
                mj_integratePos(model.get(), q.data(), dq.data(), 1.0);
                clipHingeQpos(q);
                projectDqDdq(q, qPrev, qPrev2, smoothQidx, params);
            }

            return q;
        }

        Eigen::VectorXd optimizeCausalLbfgsImpl(const HumanFrame& preparedHuman, const Eigen::VectorXd& qInit,
                                                const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                                                const CausalRefineParams& params) {
            (void)preparedHuman;
            const double wV = params.wVelocity;
            const double wA = params.wAcceleration;

            const causal_lbfgs::BoxBounds bounds = buildQposBounds();
            auto costGrad = [&](const Eigen::VectorXd& q, double* cost, Eigen::VectorXd* grad) {
                computeLbfgsCostAndGrad(q, qPrev, qPrev2, wV, wA, cost, grad);
            };

            causal_lbfgs::Options lbfgsOpts;
            lbfgsOpts.maxIter = params.fastOptIter;
            lbfgsOpts.ftol    = params.optTol;

            const causal_lbfgs::Result result =
                causal_lbfgs::minimizeWithCostGrad(costGrad, qInit, bounds, lbfgsOpts);
            Eigen::VectorXd qOut = result.success ? result.x : qInit;
            clipHingeQpos(qOut);
            return qOut;
        }

        Eigen::VectorXd optimizeCausalRefineImpl(const HumanFrame& preparedHuman, const Eigen::VectorXd& qInit,
                                                 const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                                                 const CausalRefineParams& params) {
            updateTaskTargets(preparedHuman);
            if (params.solver == CausalSolver::kNone) {
                return qInit;
            }
            if (params.solver == CausalSolver::kLbfgs) {
                return optimizeCausalLbfgsImpl(preparedHuman, qInit, qPrev, qPrev2, params);
            }
            return optimizeCausalGnImpl(preparedHuman, qInit, qPrev, qPrev2, params);
        }

        Eigen::VectorXd retargetLightIkImpl(const HumanFrame& humanFrame, bool offsetToGround, int maxIterations) {
            if (maxIterations <= 0) {
                return qpos;
            }

            HumanFrame prepared = applyContactGround(prepareHumanFrame(humanFrame, offsetToGround));
            updateTaskTargets(prepared);

            const int savedMaxIter = options.maxIterations;
            options.maxIterations  = maxIterations;
            if (ikConfig.useTable1) {
                solveTaskSet(tasks1);
            }
            if (ikConfig.useTable2) {
                solveTaskSet(tasks2);
            }
            options.maxIterations = savedMaxIter;
            return qpos;
        }
    };

    MujocoRetargetBackend::MujocoRetargetBackend(const std::filesystem::path& robotModelPath, IkConfig ikConfig, RetargetOptions options)
        : impl_(std::make_unique<Impl>(robotModelPath, std::move(ikConfig), std::move(options))) {}

    MujocoRetargetBackend::~MujocoRetargetBackend() = default;

    Eigen::VectorXd MujocoRetargetBackend::retargetFrame(const HumanFrame& humanFrame, bool offsetToGround) {
        auto t_now          = std::chrono::steady_clock::now();
        HumanFrame prepared = impl_->applyContactGround(impl_->prepareHumanFrame(humanFrame, offsetToGround));
        impl_->updateTaskTargets(prepared);
        if (impl_->ikConfig.useTable1) {
            impl_->solveTaskSet(impl_->tasks1);
        }
        if (impl_->ikConfig.useTable2) {
            impl_->solveTaskSet(impl_->tasks2);
        }
        impl_->finalizeRobotState();

        LOG(INFO) << "Retargeting took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_now).count() << " ms";
        return impl_->qpos;
    }

    HumanFrame MujocoRetargetBackend::prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
        return impl_->prepareHumanFrame(humanFrame, offsetToGround);
    }

    HumanFrame MujocoRetargetBackend::prepareRetargetInput(const HumanFrame& humanFrame, bool offsetToGround) {
        return impl_->applyContactGround(impl_->prepareHumanFrame(humanFrame, offsetToGround));
    }

    Eigen::VectorXd MujocoRetargetBackend::retargetLightIk(const HumanFrame& humanFrame, bool offsetToGround, int maxIterations) {
        return impl_->retargetLightIkImpl(humanFrame, offsetToGround, maxIterations);
    }

    Eigen::VectorXd MujocoRetargetBackend::optimizeCausalRefine(const HumanFrame& preparedHuman, const Eigen::VectorXd& qInit,
                                                                const Eigen::VectorXd& qPrev, const Eigen::VectorXd& qPrev2,
                                                                const CausalRefineParams& params) {
        if (qInit.size() != impl_->model->nq || qPrev.size() != impl_->model->nq || qPrev2.size() != impl_->model->nq) {
            throw std::runtime_error("optimizeCausalRefine qpos size mismatch.");
        }
        mju_copy(impl_->data->qpos, qInit.data(), impl_->model->nq);
        mj_forward(impl_->model.get(), impl_->data.get());
        impl_->syncQposFromData();
        return impl_->optimizeCausalRefineImpl(preparedHuman, qInit, qPrev, qPrev2, params);
    }

    void MujocoRetargetBackend::setQpos(const Eigen::VectorXd& qpos) {
        if (qpos.size() != impl_->model->nq) {
            throw std::runtime_error("setQpos size mismatch.");
        }

        mju_copy(impl_->data->qpos, qpos.data(), impl_->model->nq);
        mju_zero(impl_->data->qvel, impl_->model->nv);
        impl_->qvel.setZero();
        mj_forward(impl_->model.get(), impl_->data.get());
        impl_->syncQposFromData();
    }

    void MujocoRetargetBackend::finalizeContact() {
        impl_->finalizeRobotState();
    }

    const Eigen::VectorXd& MujocoRetargetBackend::currentQpos() const {
        return impl_->qpos;
    }

    bool MujocoRetargetBackend::hasRootFreeFlyer() const {
        return impl_->hasRootFreeFlyer;
    }

    const std::vector<ScalarJointCoordinate>& MujocoRetargetBackend::scalarJointCoordinates() const {
        return impl_->scalarJointCoordinates;
    }

    void MujocoRetargetBackend::setMotionFps(double fps) {
        if (impl_->contactGround) {
            impl_->contactGround->setFps(fps);
        }
    }

}  // namespace gmr
