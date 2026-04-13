#include "gmr/retarget/retargeter.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <mujoco/mujoco.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "gmr/solver/qp_solver.h"

namespace gmr {

    struct PinTaskRuntime {
        pinocchio::FrameIndex frameId = 0;
        bool useJointPose             = false;
        pinocchio::JointIndex jointId = 0;
        std::string humanBodyName;
        double posWeight             = 0.0;
        double rotWeight             = 0.0;
        Eigen::Vector3d posOffset    = Eigen::Vector3d::Zero();
        Eigen::Quaterniond rotOffset = Eigen::Quaterniond::Identity();

        Eigen::Vector3d targetPos    = Eigen::Vector3d::Zero();
        Eigen::Quaterniond targetRot = Eigen::Quaterniond::Identity();
    };

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

    std::string toLower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    Eigen::Vector3d computeOrientationErrorWorld(const Eigen::Quaterniond& current, const Eigen::Quaterniond& target) {
        Eigen::Quaterniond qErr = target * current.conjugate();
        if (qErr.w() < 0.0) {
            qErr.coeffs() *= -1.0;
        }

        const Eigen::Vector3d vec = qErr.vec();
        const double vecNorm      = vec.norm();
        if (vecNorm < 1e-12) {
            return Eigen::Vector3d::Zero();
        }

        const double angle = 2.0 * std::atan2(vecNorm, qErr.w());
        return vec / vecNorm * angle;
    }

    HumanFrame scaleAndOffsetHumanFrameImpl(const HumanFrame& frame, const IkConfig& ikConfig,
                                            const std::unordered_map<std::string, Eigen::Vector3d>& table1PosOffsets,
                                            const std::unordered_map<std::string, Eigen::Quaterniond>& table1RotOffsets,
                                            bool offsetToGround) {
        auto rootIt = frame.find(ikConfig.humanRootName);
        if (rootIt == frame.end()) {
            throw std::runtime_error("Human frame misses root body: " + ikConfig.humanRootName);
        }

        HumanFrame result;
        const Eigen::Vector3d rootPos    = rootIt->second.position;
        const Eigen::Quaterniond rootRot = rootIt->second.orientation;

        const auto rootScaleIt              = ikConfig.humanScaleTable.find(ikConfig.humanRootName);
        const double rootScale              = rootScaleIt == ikConfig.humanScaleTable.end() ? 1.0 : rootScaleIt->second;
        const Eigen::Vector3d scaledRootPos = rootScale * rootPos;
        result[ikConfig.humanRootName]      = HumanBodyState{scaledRootPos, rootRot};

        for (const auto& [bodyName, scale] : ikConfig.humanScaleTable) {
            if (bodyName == ikConfig.humanRootName) {
                continue;
            }

            auto bodyIt = frame.find(bodyName);
            if (bodyIt == frame.end()) {
                continue;
            }

            const Eigen::Vector3d localPos = (bodyIt->second.position - rootPos) * scale;
            result[bodyName]               = HumanBodyState{localPos + scaledRootPos, bodyIt->second.orientation};
        }

        for (auto& [bodyName, body] : result) {
            auto posIt = table1PosOffsets.find(bodyName);
            auto rotIt = table1RotOffsets.find(bodyName);
            if (posIt == table1PosOffsets.end() || rotIt == table1RotOffsets.end()) {
                continue;
            }

            body.orientation = body.orientation * rotIt->second;
            body.position += body.orientation * posIt->second;
        }

        if (offsetToGround) {
            double lowest = std::numeric_limits<double>::infinity();
            for (const auto& [bodyName, body] : result) {
                if (bodyName.find("Foot") == std::string::npos && bodyName.find("foot") == std::string::npos) {
                    continue;
                }
                lowest = std::min(lowest, body.position[2]);
            }

            if (std::isfinite(lowest)) {
                for (auto& [bodyName, body] : result) {
                    (void)bodyName;
                    body.position[2] = body.position[2] - lowest + 0.1;
                }
            }
        }

        return result;
    }

    std::string readTextFile(const std::filesystem::path& path) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open file: " + path.string());
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        return oss.str();
    }

    void writeTextFile(const std::filesystem::path& path, const std::string& text) {
        std::ofstream ofs(path);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + path.string());
        }
        ofs << text;
    }

    std::size_t findMatchingTagEnd(const std::string& xml, const std::string& tag, std::size_t openPos) {
        const std::string openTag  = "<" + tag;
        const std::string closeTag = "</" + tag + ">";

        int depth          = 1;
        std::size_t cursor = openPos + openTag.size();
        while (depth > 0) {
            const std::size_t nextOpen  = xml.find(openTag, cursor);
            const std::size_t nextClose = xml.find(closeTag, cursor);
            if (nextClose == std::string::npos) {
                throw std::runtime_error("Malformed XML: missing closing tag for <" + tag + ">.");
            }

            if (nextOpen != std::string::npos && nextOpen < nextClose) {
                depth += 1;
                cursor = nextOpen + openTag.size();
            } else {
                depth -= 1;
                cursor = nextClose + closeTag.size();
            }
        }
        return cursor;
    }

    std::string stripRepeatedTopLevelTag(const std::string& xml, const std::string& tag) {
        const std::string openTag = "<" + tag;

        std::string out             = xml;
        const std::size_t firstOpen = out.find(openTag);
        if (firstOpen == std::string::npos) {
            return out;
        }

        std::size_t firstEnd = findMatchingTagEnd(out, tag, firstOpen);
        while (true) {
            const std::size_t nextOpen = out.find(openTag, firstEnd);
            if (nextOpen == std::string::npos) {
                break;
            }
            const std::size_t nextEnd = findMatchingTagEnd(out, tag, nextOpen);
            out.erase(nextOpen, nextEnd - nextOpen);
        }

        return out;
    }

    std::string sanitizeMjcfForPinocchio(const std::filesystem::path& path) {
        std::string xml = readTextFile(path);
        xml             = stripRepeatedTopLevelTag(xml, "asset");
        xml             = stripRepeatedTopLevelTag(xml, "worldbody");
        return xml;
    }

    void buildSanitizedMjcfModel(const std::filesystem::path& path, pinocchio::Model* model) {
        const std::string sanitized         = sanitizeMjcfForPinocchio(path);
        const std::filesystem::path tmpPath = path.parent_path() / ".pinocchio_sanitized_tmp.xml";
        writeTextFile(tmpPath, sanitized);
        try {
            pinocchio::mjcf::buildModel(tmpPath.string(), *model);
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(tmpPath, ec);
            throw;
        }
        std::error_code ec;
        std::filesystem::remove(tmpPath, ec);
    }

    std::optional<pinocchio::FrameIndex> findFrameByNameAndType(const pinocchio::Model& model, const std::string& name,
                                                                pinocchio::FrameType type) {
        for (pinocchio::FrameIndex i = 0; i < model.nframes; ++i) {
            const auto& frame = model.frames[i];
            if (frame.name == name && frame.type == type) {
                return i;
            }
        }
        return std::nullopt;
    }

    std::pair<pinocchio::FrameIndex, pinocchio::FrameType> resolveTaskFrameId(const pinocchio::Model& model, const std::string& name) {
        static const std::array<pinocchio::FrameType, 4> kPriority = {pinocchio::BODY, pinocchio::OP_FRAME, pinocchio::FIXED_JOINT,
                                                                      pinocchio::JOINT};
        for (pinocchio::FrameType type : kPriority) {
            auto frameId = findFrameByNameAndType(model, name, type);
            if (frameId.has_value()) {
                return {*frameId, type};
            }
        }
        throw std::runtime_error("Frame not found in Pinocchio model: " + name);
    }

    RetargetBackend parseRetargetBackend(const std::string& backendName) {
        const std::string lowered = toLower(backendName);
        if (lowered == "pinocchio") {
            return RetargetBackend::kPinocchio;
        }
        if (lowered == "mujoco") {
            return RetargetBackend::kMujoco;
        }
        throw std::runtime_error("Unsupported backend: " + backendName + ". Expected pinocchio or mujoco.");
    }

    const char* toString(RetargetBackend backend) {
        switch (backend) {
            case RetargetBackend::kPinocchio:
                return "pinocchio";
            case RetargetBackend::kMujoco:
                return "mujoco";
        }
        return "unknown";
    }

    struct PinocchioRetargetBackend::Impl {
        pinocchio::Model model;
        std::unique_ptr<pinocchio::Data> data;

        IkConfig ikConfig;
        RetargetOptions options;

        std::vector<PinTaskRuntime> tasks1;
        std::vector<PinTaskRuntime> tasks2;
        std::unordered_map<std::string, Eigen::Vector3d> table1PosOffsets;
        std::unordered_map<std::string, Eigen::Quaterniond> table1RotOffsets;

        bool hasRootFreeFlyer = false;
        std::vector<ScalarJointCoordinate> scalarJointCoordinates;
        Eigen::VectorXd qpos;
        Eigen::VectorXd qposPin;
        Eigen::VectorXd qvel;

        Impl(const std::filesystem::path& robotModelPath, IkConfig ikConfigIn, RetargetOptions optionsIn)
            : ikConfig(std::move(ikConfigIn)), options(std::move(optionsIn)) {
            const std::string extension = robotModelPath.extension().string();
            try {
                if (extension == ".urdf") {
                    pinocchio::urdf::buildModel(robotModelPath.string(), pinocchio::JointModelFreeFlyer(), model);
                } else if (extension == ".xml" || extension == ".mjcf") {
                    buildSanitizedMjcfModel(robotModelPath, &model);
                } else {
                    throw std::runtime_error("Unsupported robot model extension for Pinocchio: " + extension);
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to load robot model for Pinocchio: " + robotModelPath.string() + " error=" + e.what());
            }

            if (model.nq <= 0 || model.nv <= 0) {
                throw std::runtime_error("Pinocchio model has invalid nq/nv.");
            }

            data             = std::make_unique<pinocchio::Data>(model);
            hasRootFreeFlyer = (model.njoints > 1 && model.joints[1].nq() == 7 && model.joints[1].nv() == 6);

            for (pinocchio::JointIndex jointId = 1; jointId < model.njoints; ++jointId) {
                const auto& jointModel = model.joints[jointId];
                if (jointModel.nq() == 1 && jointModel.nv() == 1) {
                    scalarJointCoordinates.push_back(ScalarJointCoordinate{jointModel.idx_q(), jointModel.idx_v(), model.names[jointId]});
                }
            }

            qposPin = pinocchio::neutral(model);
            qvel    = Eigen::VectorXd::Zero(model.nv);
            qpos    = pinocchioToMujocoQpos(qposPin);

            auto buildTasks = [this](const std::vector<IkTaskEntry>& src, std::vector<PinTaskRuntime>* dst) {
                dst->clear();
                dst->reserve(src.size());
                for (const auto& entry : src) {
                    PinTaskRuntime task;
                    auto [frameId, frameType] = resolveTaskFrameId(model, entry.robotBodyName);
                    (void)frameType;
                    task.frameId = frameId;
                    if (hasRootFreeFlyer && entry.robotBodyName == ikConfig.robotRootName) {
                        task.useJointPose = true;
                        task.jointId      = 1;
                    }
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

            syncDataFromQpos();
        }

        Eigen::VectorXd pinocchioToMujocoQpos(const Eigen::VectorXd& qPinIn) const {
            Eigen::VectorXd qMj = qPinIn;
            if (hasRootFreeFlyer && qMj.size() >= 7) {
                qMj[3] = qPinIn[6];
                qMj[4] = qPinIn[3];
                qMj[5] = qPinIn[4];
                qMj[6] = qPinIn[5];
            }
            return qMj;
        }

        Eigen::VectorXd mujocoToPinocchioQpos(const Eigen::VectorXd& qMjIn) const {
            Eigen::VectorXd qPin = qMjIn;
            if (hasRootFreeFlyer && qPin.size() >= 7) {
                qPin[3] = qMjIn[4];
                qPin[4] = qMjIn[5];
                qPin[5] = qMjIn[6];
                qPin[6] = qMjIn[3];
            }
            return qPin;
        }

        void syncDataFromQpos() {
            qpos = pinocchioToMujocoQpos(qposPin);
            pinocchio::forwardKinematics(model, *data, qposPin, qvel);
            pinocchio::updateFramePlacements(model, *data);
        }

        HumanFrame prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
            return scaleAndOffsetHumanFrameImpl(humanFrame, ikConfig, table1PosOffsets, table1RotOffsets, offsetToGround);
        }

        void updateTaskTargets(const HumanFrame& frame) {
            auto fill = [&frame](std::vector<PinTaskRuntime>* tasks) {
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

        double computeTaskError(const std::vector<PinTaskRuntime>& tasks) const {
            double sqErr = 0.0;
            for (const auto& task : tasks) {
                Eigen::Vector3d currPos    = Eigen::Vector3d::Zero();
                Eigen::Quaterniond currRot = Eigen::Quaterniond::Identity();
                if (task.useJointPose) {
                    if (task.jointId >= model.njoints) {
                        continue;
                    }
                    const pinocchio::SE3& pose = data->oMi[task.jointId];
                    currPos                    = pose.translation();
                    currRot                    = Eigen::Quaterniond(pose.rotation());
                } else {
                    if (task.frameId >= model.nframes) {
                        continue;
                    }
                    const pinocchio::SE3& pose = data->oMf[task.frameId];
                    currPos                    = pose.translation();
                    currRot                    = Eigen::Quaterniond(pose.rotation());
                }
                currRot.normalize();

                const Eigen::Vector3d posErr = task.targetPos - currPos;
                const Eigen::Vector3d rotErr = computeOrientationErrorWorld(currRot, task.targetRot);
                sqErr += posErr.squaredNorm() + rotErr.squaredNorm();
            }
            return std::sqrt(sqErr);
        }

        void solveTaskSet(const std::vector<PinTaskRuntime>& tasks) {
            if (tasks.empty()) {
                return;
            }

            const int nv    = model.nv;
            const double dt = options.integrationTimestep;
            if (dt <= 1e-12) {
                throw std::runtime_error("integrationTimestep must be positive.");
            }
            const double invDt = 1.0 / dt;

            double currError = computeTaskError(tasks);
            solver::QPSolver solver;

            for (int iter = 0; iter < options.maxIterations; ++iter) {
                solver::QPData qp;
                qp.reset(nv, nv);

                qp.CI.setIdentity();
                qp.ciLb.setConstant(-1e9);
                qp.ciUb.setConstant(1e9);

                if (options.useVelocityLimit) {
                    qp.ciLb.setConstant(-options.velocityLimit);
                    qp.ciUb.setConstant(options.velocityLimit);
                }

                for (pinocchio::JointIndex jointId = 1; jointId < model.njoints; ++jointId) {
                    const auto& jointModel = model.joints[jointId];
                    if (jointModel.nq() != 1 || jointModel.nv() != 1) {
                        continue;
                    }

                    const int qadr    = jointModel.idx_q();
                    const int vadr    = jointModel.idx_v();
                    const double qmin = model.lowerPositionLimit[qadr];
                    const double qmax = model.upperPositionLimit[qadr];
                    if (std::isfinite(qmin) && std::isfinite(qmax)) {
                        qp.ciLb[vadr] = std::max(qp.ciLb[vadr], (qmin - qposPin[qadr]) / dt);
                        qp.ciUb[vadr] = std::min(qp.ciUb[vadr], (qmax - qposPin[qadr]) / dt);
                    }
                }

                Eigen::Matrix<double, 6, Eigen::Dynamic> frameJacobian(6, nv);
                pinocchio::computeJointJacobians(model, *data, qposPin);

                for (const auto& task : tasks) {
                    Eigen::Vector3d currPos    = Eigen::Vector3d::Zero();
                    Eigen::Quaterniond currRot = Eigen::Quaterniond::Identity();

                    if (task.useJointPose) {
                        if (task.jointId >= model.njoints) {
                            continue;
                        }
                        frameJacobian = pinocchio::getJointJacobian(model, *data, task.jointId, pinocchio::LOCAL_WORLD_ALIGNED);
                        const pinocchio::SE3& pose = data->oMi[task.jointId];
                        currPos                    = pose.translation();
                        currRot                    = Eigen::Quaterniond(pose.rotation());
                    } else {
                        if (task.frameId >= model.nframes) {
                            continue;
                        }
                        frameJacobian.setZero();
                        pinocchio::computeFrameJacobian(model, *data, qposPin, task.frameId, pinocchio::LOCAL_WORLD_ALIGNED, frameJacobian);
                        const pinocchio::SE3& pose = data->oMf[task.frameId];
                        currPos                    = pose.translation();
                        currRot                    = Eigen::Quaterniond(pose.rotation());
                    }

                    currRot.normalize();

                    const auto Jp = frameJacobian.topRows<3>();
                    const auto Jr = frameJacobian.bottomRows<3>();

                    const Eigen::Vector3d posErr       = task.targetPos - currPos;
                    const Eigen::Vector3d rotErr       = computeOrientationErrorWorld(currRot, task.targetRot);
                    const Eigen::Vector3d posTargetVel = posErr * invDt;
                    const Eigen::Vector3d rotTargetVel = rotErr * invDt;

                    if (task.posWeight > 0.0) {
                        qp.H.noalias() += task.posWeight * (Jp.transpose() * Jp);
                        qp.g.noalias() += -task.posWeight * (Jp.transpose() * posTargetVel);
                    }
                    if (task.rotWeight > 0.0) {
                        qp.H.noalias() += task.rotWeight * (Jr.transpose() * Jr);
                        qp.g.noalias() += -task.rotWeight * (Jr.transpose() * rotTargetVel);
                    }
                }

                qp.H.diagonal().array() += options.damping;

                const solver::QPOutput& out = solver.solve(qp);
                if (out.status != solver::QPStatus::kOptimal) {
                    throw std::runtime_error("QP solver failed while retargeting.");
                }

                qvel    = out.x;
                qposPin = pinocchio::integrate(model, qposPin, qvel * dt);
                syncDataFromQpos();

                const double nextError = computeTaskError(tasks);
                if (currError - nextError <= options.progressThreshold) {
                    break;
                }
                currError = nextError;
            }
        }
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

        Impl(const std::filesystem::path& robotModelPath, IkConfig ikConfigIn, RetargetOptions optionsIn)
            : ikConfig(std::move(ikConfigIn)), options(std::move(optionsIn)) {
            const std::string extension = toLower(robotModelPath.extension().string());
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
                        throw std::runtime_error("Body not found in MuJoCo model: " + entry.robotBodyName);
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

            mju_copy(data->qpos, model->qpos0, model->nq);
            mju_zero(data->qvel, model->nv);
            mj_forward(model.get(), data.get());

            qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq);
            qvel = Eigen::VectorXd::Zero(model->nv);
        }

        void syncQposFromData() { qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq); }

        HumanFrame prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
            return scaleAndOffsetHumanFrameImpl(humanFrame, ikConfig, table1PosOffsets, table1RotOffsets, offsetToGround);
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
                const Eigen::Vector3d rotErr = computeOrientationErrorWorld(currRot, task.targetRot);
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
            const double invDt = 1.0 / dt;

            double currError = computeTaskError(tasks);
            solver::QPSolver solver;

            std::vector<mjtNum> jacp(3 * nv);
            std::vector<mjtNum> jacr(3 * nv);

            for (int iter = 0; iter < options.maxIterations; ++iter) {
                solver::QPData qp;
                qp.reset(nv, nv);

                qp.CI.setIdentity();
                qp.ciLb.setConstant(-1e9);
                qp.ciUb.setConstant(1e9);

                if (options.useVelocityLimit) {
                    qp.ciLb.setConstant(-options.velocityLimit);
                    qp.ciUb.setConstant(options.velocityLimit);
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

                    qp.ciLb[vadr] = std::max(qp.ciLb[vadr], (qmin - data->qpos[qadr]) / dt);
                    qp.ciUb[vadr] = std::min(qp.ciUb[vadr], (qmax - data->qpos[qadr]) / dt);
                }

                for (const auto& task : tasks) {
                    std::fill(jacp.begin(), jacp.end(), 0.0);
                    std::fill(jacr.begin(), jacr.end(), 0.0);
                    mj_jacBody(model.get(), data.get(), jacp.data(), jacr.data(), task.bodyId);

                    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> Jp(jacp.data(), 3, nv);
                    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> Jr(jacr.data(), 3, nv);

                    const double* xpos  = &data->xpos[3 * task.bodyId];
                    const double* xquat = &data->xquat[4 * task.bodyId];
                    const Eigen::Vector3d currPos(xpos[0], xpos[1], xpos[2]);
                    Eigen::Quaterniond currRot(xquat[0], xquat[1], xquat[2], xquat[3]);
                    currRot.normalize();

                    const Eigen::Vector3d posErr       = task.targetPos - currPos;
                    const Eigen::Vector3d rotErr       = computeOrientationErrorWorld(currRot, task.targetRot);
                    const Eigen::Vector3d posTargetVel = posErr * invDt;
                    const Eigen::Vector3d rotTargetVel = rotErr * invDt;

                    if (task.posWeight > 0.0) {
                        qp.H.noalias() += task.posWeight * (Jp.transpose() * Jp);
                        qp.g.noalias() += -task.posWeight * (Jp.transpose() * posTargetVel);
                    }
                    if (task.rotWeight > 0.0) {
                        qp.H.noalias() += task.rotWeight * (Jr.transpose() * Jr);
                        qp.g.noalias() += -task.rotWeight * (Jr.transpose() * rotTargetVel);
                    }
                }

                qp.H.diagonal().array() += options.damping;

                const solver::QPOutput& out = solver.solve(qp);
                if (out.status != solver::QPStatus::kOptimal) {
                    throw std::runtime_error("QP solver failed while retargeting.");
                }

                qvel = out.x;

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
    };

    PinocchioRetargetBackend::PinocchioRetargetBackend(const std::filesystem::path& robotModelPath, IkConfig ikConfig,
                                                       RetargetOptions options)
        : impl_(std::make_unique<Impl>(robotModelPath, std::move(ikConfig), std::move(options))) {}

    PinocchioRetargetBackend::~PinocchioRetargetBackend() = default;

    Eigen::VectorXd PinocchioRetargetBackend::retargetFrame(const HumanFrame& humanFrame, bool offsetToGround) {
        HumanFrame prepared = impl_->prepareHumanFrame(humanFrame, offsetToGround);
        impl_->updateTaskTargets(prepared);
        if (impl_->ikConfig.useTable1) {
            impl_->solveTaskSet(impl_->tasks1);
        }
        if (impl_->ikConfig.useTable2) {
            impl_->solveTaskSet(impl_->tasks2);
        }
        return impl_->qpos;
    }

    HumanFrame PinocchioRetargetBackend::prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
        return impl_->prepareHumanFrame(humanFrame, offsetToGround);
    }

    void PinocchioRetargetBackend::setQpos(const Eigen::VectorXd& qpos) {
        if (qpos.size() != impl_->qpos.size()) {
            throw std::runtime_error("setQpos size mismatch.");
        }
        impl_->qpos    = qpos;
        impl_->qposPin = impl_->mujocoToPinocchioQpos(impl_->qpos);
        impl_->qvel.setZero();
        impl_->syncDataFromQpos();
    }

    const Eigen::VectorXd& PinocchioRetargetBackend::currentQpos() const { return impl_->qpos; }

    bool PinocchioRetargetBackend::hasRootFreeFlyer() const { return impl_->hasRootFreeFlyer; }

    const std::vector<ScalarJointCoordinate>& PinocchioRetargetBackend::scalarJointCoordinates() const {
        return impl_->scalarJointCoordinates;
    }

    MujocoRetargetBackend::MujocoRetargetBackend(const std::filesystem::path& robotModelPath, IkConfig ikConfig, RetargetOptions options)
        : impl_(std::make_unique<Impl>(robotModelPath, std::move(ikConfig), std::move(options))) {}

    MujocoRetargetBackend::~MujocoRetargetBackend() = default;

    Eigen::VectorXd MujocoRetargetBackend::retargetFrame(const HumanFrame& humanFrame, bool offsetToGround) {
        HumanFrame prepared = impl_->prepareHumanFrame(humanFrame, offsetToGround);
        impl_->updateTaskTargets(prepared);
        if (impl_->ikConfig.useTable1) {
            impl_->solveTaskSet(impl_->tasks1);
        }
        if (impl_->ikConfig.useTable2) {
            impl_->solveTaskSet(impl_->tasks2);
        }
        return impl_->qpos;
    }

    HumanFrame MujocoRetargetBackend::prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
        return impl_->prepareHumanFrame(humanFrame, offsetToGround);
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

    const Eigen::VectorXd& MujocoRetargetBackend::currentQpos() const { return impl_->qpos; }

    bool MujocoRetargetBackend::hasRootFreeFlyer() const { return impl_->hasRootFreeFlyer; }

    const std::vector<ScalarJointCoordinate>& MujocoRetargetBackend::scalarJointCoordinates() const {
        return impl_->scalarJointCoordinates;
    }

    std::unique_ptr<Retargeter> createRetargeter(RetargetBackend backend, const std::filesystem::path& robotModelPath, IkConfig ikConfig,
                                                 RetargetOptions options) {
        if (backend == RetargetBackend::kPinocchio) {
            return std::make_unique<PinocchioRetargetBackend>(robotModelPath, std::move(ikConfig), options);
        }
        if (backend == RetargetBackend::kMujoco) {
            return std::make_unique<MujocoRetargetBackend>(robotModelPath, std::move(ikConfig), options);
        }
        throw std::runtime_error("Unsupported retarget backend.");
    }

}  // namespace gmr
