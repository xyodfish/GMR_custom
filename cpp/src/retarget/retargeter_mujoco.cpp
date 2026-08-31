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

    struct MobileArmRuntime {
        MobileArmChainConfig config;
        int shoulderBodyId = -1;
        double upperLength = 0.0;
        double forearmLength = 0.0;
        MujocoTaskRuntime elbowTask;
        MujocoTaskRuntime wristTask;
        std::optional<MujocoTaskRuntime> orientationTask;
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
        std::optional<MujocoTaskRuntime> mobileTorsoTask;
        std::optional<MujocoTaskRuntime> mobileHeadTask;
        std::vector<MobileArmRuntime> mobileArms;
        Eigen::Quaterniond mobileTorsoNeutralRot = Eigen::Quaterniond::Identity();
        Eigen::Quaterniond mobileHeadNeutralRelativeRot = Eigen::Quaterniond::Identity();
        Eigen::VectorXd mobilePostureCost;
        bool mobileInitialized = false;

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

            if (ikConfig.mobileUpperBody.enabled) {
                setupMobileUpperBody();
            }

            qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq);
            qvel = Eigen::VectorXd::Zero(model->nv);
        }

        int requireBody(const std::string& name) const {
            const int bodyId = mj_name2id(model.get(), mjOBJ_BODY, name.c_str());
            if (bodyId < 0) {
                throw std::runtime_error("mobile_upper_body references missing robot body: " + name);
            }

            return bodyId;
        }

        Eigen::Vector3d bodyPosition(int bodyId) const {
            return Eigen::Map<const Eigen::Vector3d>(&data->xpos[3 * bodyId]);
        }

        Eigen::Quaterniond bodyOrientation(int bodyId) const {
            const double* quat = &data->xquat[4 * bodyId];
            return Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]).normalized();
        }

        MujocoTaskRuntime makeMobileTask(
            const std::string& bodyName,
            const std::string& humanBodyName,
            double posWeight,
            double rotWeight) const {
            MujocoTaskRuntime task;
            task.bodyId = requireBody(bodyName);
            task.humanBodyName = humanBodyName;
            task.posWeight = posWeight;
            task.rotWeight = rotWeight;
            return task;
        }

        void setupMobileUpperBody() {
            if (!ikConfig.planarBase.enabled || model->nq < 3 || model->nv < 3) {
                throw std::runtime_error("mobile_upper_body requires a three-DoF planar base.");
            }

            const MobileUpperBodyConfig& cfg = ikConfig.mobileUpperBody;
            mobileTorsoTask = makeMobileTask(
                cfg.torsoFrame,
                cfg.torsoHumanBody,
                cfg.torsoPositionCost,
                cfg.torsoOrientationCost);
            mobileTorsoNeutralRot = bodyOrientation(mobileTorsoTask->bodyId);

            if (!cfg.headFrame.empty()) {
                if (cfg.headHumanBody.empty()) {
                    throw std::runtime_error("mobile_upper_body head_frame requires head_human_body.");
                }

                mobileHeadTask = makeMobileTask(cfg.headFrame, cfg.headHumanBody, 0.0, cfg.headOrientationCost);
                mobileHeadNeutralRelativeRot =
                    mobileTorsoNeutralRot.conjugate() * bodyOrientation(mobileHeadTask->bodyId);
            }

            mobileArms.reserve(cfg.armChains.size());
            for (const MobileArmChainConfig& chainConfig : cfg.armChains) {
                MobileArmRuntime arm;
                arm.config = chainConfig;
                arm.shoulderBodyId = requireBody(chainConfig.shoulderFrame);
                arm.elbowTask = makeMobileTask(
                    chainConfig.elbowFrame,
                    chainConfig.elbowHumanBody,
                    cfg.armPositionCost,
                    cfg.elbowOrientationCost);
                arm.wristTask = makeMobileTask(
                    chainConfig.wristFrame,
                    chainConfig.wristHumanBody,
                    cfg.armPositionCost,
                    0.0);
                arm.upperLength = (bodyPosition(arm.elbowTask.bodyId) - bodyPosition(arm.shoulderBodyId)).norm();
                arm.forearmLength = (bodyPosition(arm.wristTask.bodyId) - bodyPosition(arm.elbowTask.bodyId)).norm();
                if (arm.upperLength <= 1e-8 || arm.forearmLength <= 1e-8) {
                    throw std::runtime_error("mobile_upper_body arm chain has a zero-length segment.");
                }

                if (cfg.wristOrientationCost > 0.0) {
                    if (chainConfig.orientationFrame.empty()) {
                        throw std::runtime_error(
                            "mobile_upper_body wrist orientation requires orientation_frame.");
                    }

                    arm.orientationTask = makeMobileTask(
                        chainConfig.orientationFrame,
                        chainConfig.wristHumanBody,
                        0.0,
                        cfg.wristOrientationCost);
                }

                mobileArms.push_back(std::move(arm));
            }

            mobilePostureCost = Eigen::VectorXd::Constant(model->nv, cfg.postureCost);
            mobilePostureCost.head(std::min<mjtSize>(3, model->nv)).setZero();
            for (const auto& [jointName, cost] : cfg.jointPostureCost) {
                const int jointId = mj_name2id(model.get(), mjOBJ_JOINT, jointName.c_str());
                if (jointId < 0) {
                    throw std::runtime_error(
                        "mobile_upper_body posture references missing joint: " + jointName);
                }

                mobilePostureCost[model->jnt_dofadr[jointId]] = cost;
            }
        }

        void syncQposFromData() { qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq); }

        HumanFrame prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
            const bool useOffsetToGround = offsetToGround && !(contactGround && contactGround->enabled());
            return retarget_internal::scaleHumanFrameOnly(humanFrame, ikConfig, useOffsetToGround);
        }

        HumanFrame prepareRetargetInput(const HumanFrame& humanFrame, bool offsetToGround) const {
            return applyContactGround(prepareHumanFrame(humanFrame, offsetToGround));
        }

        HumanFrame prepareRetargetInput(
            const HumanFrame& humanFrame,
            const ContactGroundState& contactState,
            bool offsetToGround) const {
            HumanFrame prepared = prepareHumanFrame(humanFrame, offsetToGround);
            if (contactGround && contactGround->enabled()) {
                return contactGround->processHumanFrame(prepared, contactState);
            }

            return prepared;
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
                    const std::optional<retarget_internal::TaskTargetPose> target =
                        retarget_internal::taskTargetFromHumanFrame(frame, task.humanBodyName, task.posOffset,
                                                                    task.rotOffset, ikConfig.groundHeight);
                    if (!target.has_value()) {
                        continue;
                    }
                    task.targetPos = target->pos;
                    task.targetRot = target->rot;
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

        void solveTaskSet(
            const std::vector<MujocoTaskRuntime>& tasks,
            int maxIterations = -1,
            int minIterations = 0,
            int frozenDofs = 0,
            bool useMobilePosture = false) {
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

            const int iterationLimit = maxIterations > 0 ? maxIterations : options.maxIterations;
            for (int iter = 0; iter < iterationLimit; ++iter) {
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

                if (useMobilePosture) {
                    Eigen::VectorXd postureError(model->nv);
                    mj_differentiatePos(
                        model.get(),
                        postureError.data(),
                        1.0,
                        model->qpos0,
                        data->qpos);
                    const Eigen::VectorXd weightedError =
                        -mobilePostureCost.cwiseProduct(postureError);
                    const Eigen::VectorXd squaredCost = mobilePostureCost.array().square();
                    qp.H.diagonal() += squaredCost;
                    qp.H.diagonal().array() += weightedError.squaredNorm();
                    qp.g.noalias() -= mobilePostureCost.cwiseProduct(weightedError);
                }

                qp.H.diagonal().array() += options.damping;

                if (nCollisionRows > 0 && collisionLimit != nullptr) {
                    collisionLimit->fillRows(data.get(), dt, 1.0, qp.CI, qp.ciLb, qp.ciUb, nv);
                }

                const solver::QPOutput& out = solver.solve(qp);
                if (out.status != solver::QPStatus::kOptimal) {
                    throw std::runtime_error("QP solver failed while retargeting.");
                }

                Eigen::VectorXd deltaQ = out.x;
                deltaQ.head(std::min(frozenDofs, nv)).setZero();
                qvel                         = deltaQ / dt;

                mj_integratePos(model.get(), data->qpos, qvel.data(), dt);
                mju_copy(data->qvel, qvel.data(), model->nv);
                mj_forward(model.get(), data.get());
                syncQposFromData();

                const double nextError = computeTaskError(tasks);
                if (iter + 1 >= minIterations && currError - nextError <= options.progressThreshold) {
                    break;
                }
                currError = nextError;
            }
        }

        void solveEnabledTaskSets() {
            if (ikConfig.useTable1) {
                solveTaskSet(tasks1);
            }
            if (ikConfig.useTable2) {
                solveTaskSet(tasks2);
            }
        }

        const HumanBodyState& requireHumanBody(const HumanFrame& frame, const std::string& name) const {
            const auto it = frame.find(name);
            if (it == frame.end()) {
                throw std::runtime_error("mobile_upper_body input is missing human body: " + name);
            }

            return it->second;
        }

        static Eigen::Vector3d normalizedDirection(
            const Eigen::Vector3d& start,
            const Eigen::Vector3d& end,
            const std::string& label) {
            const Eigen::Vector3d direction = end - start;
            const double norm = direction.norm();
            if (norm <= 1e-8) {
                throw std::runtime_error("Cannot retarget zero-length human segment: " + label);
            }

            return direction / norm;
        }

        static Eigen::Quaterniond quatFromEulerXyz(const Eigen::Vector3d& angles) {
            return Eigen::Quaterniond(Eigen::AngleAxisd(angles.x(), Eigen::Vector3d::UnitX())) *
                Eigen::Quaterniond(Eigen::AngleAxisd(angles.y(), Eigen::Vector3d::UnitY())) *
                Eigen::Quaterniond(Eigen::AngleAxisd(angles.z(), Eigen::Vector3d::UnitZ()));
        }

        static Eigen::Vector3d eulerXyz(const Eigen::Quaterniond& orientation) {
            const Eigen::Matrix3d rotation = orientation.normalized().toRotationMatrix();
            const double y = std::asin(std::clamp(rotation(0, 2), -1.0, 1.0));
            const double cosY = std::cos(y);
            if (std::abs(cosY) < 1e-8) {
                return Eigen::Vector3d(std::atan2(rotation(2, 1), rotation(1, 1)), y, 0.0);
            }

            return Eigen::Vector3d(
                std::atan2(-rotation(1, 2), rotation(2, 2)),
                y,
                std::atan2(-rotation(0, 1), rotation(0, 0)));
        }

        static Eigen::Vector3d clampEuler(
            const Eigen::Quaterniond& orientation,
            const Eigen::Vector3d& limitDeg) {
            constexpr double kDegToRad = 0.017453292519943295;
            const Eigen::Vector3d limit = limitDeg * kDegToRad;
            return eulerXyz(orientation).cwiseMax(-limit).cwiseMin(limit);
        }

        void snapPlanarBase(const HumanFrame& prepared) {
            const PlanarBaseConfig& cfg = ikConfig.planarBase;
            const HumanBodyState& root = requireHumanBody(prepared, cfg.humanBody);
            Eigen::Quaterniond orientation = root.orientation.normalized();
            if (cfg.yawFrame == "g1_pelvis") {
                orientation = orientation * Eigen::Quaterniond(0.5, -0.5, -0.5, -0.5);
            }

            const Eigen::Matrix3d rotation = orientation.toRotationMatrix();
            data->qpos[0] = root.position.x();
            data->qpos[1] = root.position.y();
            data->qpos[2] = std::atan2(rotation(1, 0), rotation(0, 0));
            mj_forward(model.get(), data.get());
        }

        void setMobileTorsoTargets(const HumanFrame& raw) {
            const MobileUpperBodyConfig& cfg = ikConfig.mobileUpperBody;
            const HumanBodyState& torso = requireHumanBody(raw, cfg.torsoHumanBody);
            const Eigen::Quaterniond baseRotation(Eigen::AngleAxisd(data->qpos[2], Eigen::Vector3d::UnitZ()));
            const Eigen::Quaterniond humanTorsoRotation =
                (torso.orientation.normalized() * cfg.torsoRotationOffset).normalized();
            const Eigen::Quaterniond relative =
                baseRotation.conjugate() * humanTorsoRotation * mobileTorsoNeutralRot.conjugate();
            const Eigen::Quaterniond targetRotation =
                baseRotation * quatFromEulerXyz(clampEuler(relative, cfg.torsoOrientationLimitDeg)) *
                mobileTorsoNeutralRot;

            const Eigen::Vector3d localOffset(cfg.torsoLocalXy.x(), cfg.torsoLocalXy.y(), 0.0);
            Eigen::Vector3d targetPosition(data->qpos[0], data->qpos[1], 0.0);
            targetPosition.head<2>() += (baseRotation * localOffset).head<2>();
            targetPosition.z() = std::clamp(
                torso.position.z() * cfg.torsoHeightScale,
                cfg.torsoHeightRange.x(),
                cfg.torsoHeightRange.y());
            mobileTorsoTask->targetPos = targetPosition;
            mobileTorsoTask->targetRot = targetRotation.normalized();

            if (mobileHeadTask.has_value()) {
                const HumanBodyState& head = requireHumanBody(raw, cfg.headHumanBody);
                const Eigen::Quaterniond headRelative = torso.orientation.normalized().conjugate() *
                    head.orientation.normalized();
                mobileHeadTask->targetPos.setZero();
                mobileHeadTask->targetRot = targetRotation *
                    quatFromEulerXyz(clampEuler(headRelative, cfg.headOrientationLimitDeg)) *
                    mobileHeadNeutralRelativeRot;
            }
        }

        void setMobileArmTargets(const HumanFrame& raw) {
            for (MobileArmRuntime& arm : mobileArms) {
                const HumanBodyState& shoulder = requireHumanBody(raw, arm.config.shoulderHumanBody);
                const HumanBodyState& elbow = requireHumanBody(raw, arm.config.elbowHumanBody);
                const HumanBodyState& wrist = requireHumanBody(raw, arm.config.wristHumanBody);
                const Eigen::Vector3d upperDirection = normalizedDirection(
                    shoulder.position,
                    elbow.position,
                    arm.config.shoulderHumanBody + "->" + arm.config.elbowHumanBody);
                const Eigen::Vector3d forearmDirection = normalizedDirection(
                    elbow.position,
                    wrist.position,
                    arm.config.elbowHumanBody + "->" + arm.config.wristHumanBody);
                const Eigen::Vector3d elbowPosition =
                    bodyPosition(arm.shoulderBodyId) + arm.upperLength * upperDirection;

                arm.elbowTask.targetPos = elbowPosition;
                arm.elbowTask.targetRot =
                    (elbow.orientation.normalized() * arm.config.elbowRotationOffset).normalized();
                arm.wristTask.targetPos = elbowPosition + arm.forearmLength * forearmDirection;
                arm.wristTask.targetRot = Eigen::Quaterniond::Identity();
                if (arm.orientationTask.has_value()) {
                    arm.orientationTask->targetPos.setZero();
                    arm.orientationTask->targetRot =
                        (wrist.orientation.normalized() * arm.config.wristRotationOffset).normalized();
                }
            }
        }

        void applyMobileJointMargin() {
            constexpr double kDegToRad = 0.017453292519943295;
            const double margin = ikConfig.mobileUpperBody.jointLimitMarginDeg * kDegToRad;
            if (margin <= 0.0) {
                return;
            }

            for (int joint = 0; joint < model->njnt; ++joint) {
                if (!model->jnt_limited[joint] || model->jnt_type[joint] != mjJNT_HINGE) {
                    continue;
                }

                const double lower = model->jnt_range[2 * joint];
                const double upper = model->jnt_range[2 * joint + 1];
                if (upper - lower <= 2.0 * margin) {
                    continue;
                }

                const int qadr = model->jnt_qposadr[joint];
                data->qpos[qadr] = std::clamp(data->qpos[qadr], lower + margin, upper - margin);
            }

            mj_forward(model.get(), data.get());
            syncQposFromData();
        }

        void solveMobileUpperBody(const HumanFrame& raw, const HumanFrame& prepared) {
            const MobileUpperBodyConfig& cfg = ikConfig.mobileUpperBody;
            snapPlanarBase(prepared);
            const Eigen::Vector3d baseQpos(data->qpos[0], data->qpos[1], data->qpos[2]);
            setMobileTorsoTargets(raw);

            std::vector<MujocoTaskRuntime> torsoTasks = {*mobileTorsoTask};
            if (mobileHeadTask.has_value()) {
                torsoTasks.push_back(*mobileHeadTask);
            }

            solveTaskSet(
                torsoTasks,
                mobileInitialized ? cfg.torsoIterations : cfg.initialTorsoIterations,
                mobileInitialized ? cfg.torsoMinIterations : cfg.initialTorsoMinIterations,
                3,
                true);

            for (int pass = 0; pass < cfg.armTargetPasses; ++pass) {
                setMobileArmTargets(raw);
                std::vector<MujocoTaskRuntime> tasks = {*mobileTorsoTask};
                if (mobileHeadTask.has_value()) {
                    tasks.push_back(*mobileHeadTask);
                }

                for (const MobileArmRuntime& arm : mobileArms) {
                    tasks.push_back(arm.elbowTask);
                    tasks.push_back(arm.wristTask);
                    if (arm.orientationTask.has_value()) {
                        tasks.push_back(*arm.orientationTask);
                    }
                }

                solveTaskSet(
                    tasks,
                    mobileInitialized ? cfg.armIterations : cfg.initialArmIterations,
                    mobileInitialized ? cfg.armMinIterations : cfg.initialArmMinIterations,
                    3,
                    true);
            }

            data->qpos[0] = baseQpos.x();
            data->qpos[1] = baseQpos.y();
            data->qpos[2] = baseQpos.z();
            mj_forward(model.get(), data.get());
            applyMobileJointMargin();
            mobileInitialized = true;
            syncQposFromData();
        }

        Eigen::VectorXd retargetPrepared(const HumanFrame& raw, const HumanFrame& prepared, bool finalize) {
            if (ikConfig.mobileUpperBody.enabled) {
                solveMobileUpperBody(raw, prepared);
                if (finalize) {
                    finalizeRobotState();
                }

                return qpos;
            }

            updateTaskTargets(prepared);
            solveEnabledTaskSets();
            if (finalize) {
                finalizeRobotState();
            }
            return qpos;
        }

        Eigen::VectorXd retargetLightIkImpl(const HumanFrame& humanFrame, bool offsetToGround, int maxIterations) {
            if (maxIterations <= 0) {
                return qpos;
            }

            const int savedMaxIter = options.maxIterations;
            options.maxIterations  = maxIterations;
            retargetPrepared(humanFrame, prepareRetargetInput(humanFrame, offsetToGround), /*finalize=*/false);
            options.maxIterations = savedMaxIter;
            return qpos;
        }

        Eigen::VectorXd retargetPreparedLightIkImpl(const HumanFrame& rawFrame, const HumanFrame& preparedFrame,
                                                     int maxIterations) {
            if (maxIterations <= 0) {
                return qpos;
            }

            const int savedMaxIter = options.maxIterations;
            options.maxIterations  = maxIterations;
            retargetPrepared(rawFrame, preparedFrame, /*finalize=*/false);
            options.maxIterations = savedMaxIter;
            return qpos;
        }
    };

    MujocoRetargetBackend::MujocoRetargetBackend(const std::filesystem::path& robotModelPath, IkConfig ikConfig, RetargetOptions options)
        : impl_(std::make_unique<Impl>(robotModelPath, std::move(ikConfig), std::move(options))) {}

    MujocoRetargetBackend::~MujocoRetargetBackend() = default;

    Eigen::VectorXd MujocoRetargetBackend::retargetFrame(const HumanFrame& humanFrame, bool offsetToGround) {
        auto t_now = std::chrono::steady_clock::now();
        Eigen::VectorXd q = impl_->retargetPrepared(
            humanFrame,
            impl_->prepareRetargetInput(humanFrame, offsetToGround),
            /*finalize=*/true);

        LOG(INFO) << "Retargeting took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_now).count() << " ms";
        return q;
    }

    HumanFrame MujocoRetargetBackend::prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
        return impl_->prepareHumanFrame(humanFrame, offsetToGround);
    }

    HumanFrame MujocoRetargetBackend::prepareRetargetInput(const HumanFrame& humanFrame, bool offsetToGround) {
        return impl_->prepareRetargetInput(humanFrame, offsetToGround);
    }

    HumanFrame MujocoRetargetBackend::prepareRetargetInput(
        const HumanFrame& humanFrame,
        const ContactGroundState& contactState,
        bool offsetToGround) {
        return impl_->prepareRetargetInput(humanFrame, contactState, offsetToGround);
    }

    Eigen::VectorXd MujocoRetargetBackend::retargetPreparedFrame(const HumanFrame& rawFrame,
                                                                  const HumanFrame& preparedFrame) {
        return impl_->retargetPrepared(rawFrame, preparedFrame, /*finalize=*/false);
    }

    Eigen::VectorXd MujocoRetargetBackend::retargetPreparedLightIk(const HumanFrame& rawFrame,
                                                                    const HumanFrame& preparedFrame,
                                                                    int maxIterations) {
        return impl_->retargetPreparedLightIkImpl(rawFrame, preparedFrame, maxIterations);
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

    void MujocoRetargetBackend::finalizeContact(const ContactGroundState& state) {
        if (impl_->contactGround && impl_->contactGround->enabled()) {
            impl_->contactGround->fixRobotPenetration(impl_->model.get(), impl_->data.get(), state);
            impl_->syncQposFromData();
        }
    }

    ContactGroundState MujocoRetargetBackend::contactGroundState() const {
        return impl_->contactGround && impl_->contactGround->enabled()
            ? impl_->contactGround->state()
            : ContactGroundState{};
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
