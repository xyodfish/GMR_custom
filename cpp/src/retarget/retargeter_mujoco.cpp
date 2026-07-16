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
#include "mink_configuration_limit.h"
#include "retargeter_internal_utils.h"
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

            if (options.integrationTimestep <= 0.0) {
                options.integrationTimestep = model->opt.timestep;
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
            return retarget_internal::scaleHumanFrameOnly(humanFrame, ikConfig, useOffsetToGround);
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
            auto fill = [this, &frame](std::vector<MujocoTaskRuntime>* tasks) {
                for (auto& task : *tasks) {
                    auto it = frame.find(task.humanBodyName);
                    if (it == frame.end()) {
                        continue;
                    }
                    const Eigen::Vector3d posOff =
                        retarget_internal::ikTaskPosOffset(task.posOffset, ikConfig.groundHeight);
                    const auto [targetPos, targetRot] = retarget_internal::applyBodyOffset(
                        it->second.position, it->second.orientation, posOff, task.rotOffset);
                    task.targetPos = targetPos;
                    task.targetRot = targetRot;
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

                Eigen::Quaterniond targetRot = task.targetRot;
                targetRot.normalize();
                const pinocchio::SE3 T_wb(currRot.toRotationMatrix(), currPos);
                const pinocchio::SE3 T_wt(targetRot.toRotationMatrix(), task.targetPos);
                const Eigen::Matrix<double, 6, 1> error = pinocchio::log6(T_wb.inverse() * T_wt).toVector();
                sqErr += error.squaredNorm();
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
            solver::QPSolver solver(options.solverName);
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

                mink_limits::applyConfigurationLimit(model.get(), data->qpos, options.configurationLimitGain, &qp.ciLb,
                                                   &qp.ciUb);

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

                    const double lmMu = options.taskLmDamping * weightedError.squaredNorm();
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
