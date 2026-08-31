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

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/collision/fcl-pinocchio-conversions.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

#include "gmr/solver/qp_solver.h"
#include "gmr/retarget/contact_ground.h"
#include "retargeter_internal_utils.h"

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
            std::optional<pinocchio::FrameIndex> frameId = findFrameByNameAndType(model, name, type);
            if (frameId.has_value()) {
                return {*frameId, type};
            }
        }
        throw std::runtime_error("Frame not found in Pinocchio model: " + name);
    }

    struct PinocchioRetargetBackend::Impl {
        pinocchio::Model model;
        std::unique_ptr<pinocchio::Data> data;
        pinocchio::GeometryModel geomModel;
        std::unique_ptr<pinocchio::GeometryData> geomData;
        bool hasCollisionGeometry = false;

        IkConfig ikConfig;
        RetargetOptions options;

        std::vector<PinTaskRuntime> tasks1;
        std::vector<PinTaskRuntime> tasks2;

        bool hasRootFreeFlyer = false;
        std::vector<ScalarJointCoordinate> scalarJointCoordinates;
        Eigen::VectorXd qpos;
        Eigen::VectorXd qposPin;
        Eigen::VectorXd qvel;

        std::unique_ptr<ContactGroundPipeline> contactGround;
        std::vector<pinocchio::FrameIndex> penetrationFootFrames;
        std::vector<pinocchio::FrameIndex> penetrationTrunkFrames;
        std::vector<pinocchio::FrameIndex> penetrationGroundFrames;
        std::vector<pinocchio::FrameIndex> penetrationLyingFrames;
        std::vector<pinocchio::GeomIndex> penetrationFootGeoms;
        std::vector<pinocchio::GeomIndex> penetrationTrunkGeoms;
        std::vector<pinocchio::GeomIndex> penetrationGroundGeoms;
        std::vector<pinocchio::GeomIndex> penetrationLyingGeoms;

        Impl(const std::filesystem::path& robotModelPath, IkConfig ikConfigIn, RetargetOptions optionsIn)
            : ikConfig(std::move(ikConfigIn)), options(std::move(optionsIn)) {
            const std::string extension = robotModelPath.extension().string();
            try {
                if (extension == ".urdf") {
                    pinocchio::urdf::buildModel(robotModelPath.string(), pinocchio::JointModelFreeFlyer(), model);
                    pinocchio::urdf::buildGeom(model, robotModelPath.string(), pinocchio::GeometryType::COLLISION, geomModel);
                    hasCollisionGeometry = (geomModel.ngeoms > 0);
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
            if (hasCollisionGeometry) {
                geomData = std::make_unique<pinocchio::GeometryData>(geomModel);
            }
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
                    // For G1 URDF, pelvis BODY frame coincides with free-flyer joint (delta=0);
                    // A/B shows this branch does not change trajectories. Kept for root-joint
                    // Jacobian path; other robots may differ if BODY≠JOINT.
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

            if (options.integrationTimestep <= 0.0) {
                options.integrationTimestep = 2e-3;
            }

            if (options.contactGround.enabled) {
                contactGround = std::make_unique<ContactGroundPipeline>(options.contactGround, nullptr, options.motionFps);
                resolvePenetrationFrames();
                resolvePenetrationGeoms();
            }

            syncDataFromQpos();
        }

        void resolvePenetrationFrames() {
            auto resolveNames = [this](const std::vector<std::string>& names, std::vector<pinocchio::FrameIndex>* out) {
                out->clear();
                out->reserve(names.size());
                for (const auto& name : names) {
                    try {
                        out->push_back(resolveTaskFrameId(model, name).first);
                    } catch (const std::exception&) {
                        // Skip bodies missing from the Pinocchio/URDF model.
                    }
                }
                std::sort(out->begin(), out->end());
                out->erase(std::unique(out->begin(), out->end()), out->end());
            };

            const auto& cfg = options.contactGround;
            resolveNames(cfg.robotFootBodies, &penetrationFootFrames);
            resolveNames(cfg.robotTrunkBodies, &penetrationTrunkFrames);

            penetrationGroundFrames = penetrationFootFrames;
            penetrationGroundFrames.insert(penetrationGroundFrames.end(), penetrationTrunkFrames.begin(),
                                           penetrationTrunkFrames.end());
            std::sort(penetrationGroundFrames.begin(), penetrationGroundFrames.end());
            penetrationGroundFrames.erase(std::unique(penetrationGroundFrames.begin(), penetrationGroundFrames.end()),
                                          penetrationGroundFrames.end());

            std::vector<pinocchio::FrameIndex> legFrames;
            std::vector<pinocchio::FrameIndex> armFrames;
            resolveNames(cfg.robotLegBodies, &legFrames);
            resolveNames(cfg.robotArmBodies, &armFrames);
            penetrationLyingFrames = penetrationGroundFrames;
            penetrationLyingFrames.insert(penetrationLyingFrames.end(), legFrames.begin(), legFrames.end());
            penetrationLyingFrames.insert(penetrationLyingFrames.end(), armFrames.begin(), armFrames.end());
            std::sort(penetrationLyingFrames.begin(), penetrationLyingFrames.end());
            penetrationLyingFrames.erase(std::unique(penetrationLyingFrames.begin(), penetrationLyingFrames.end()),
                                         penetrationLyingFrames.end());
        }

        const std::vector<pinocchio::FrameIndex>& activePenetrationFrames(bool lowPose) const {
            if (!contactGround) {
                static const std::vector<pinocchio::FrameIndex> kEmpty;
                return kEmpty;
            }
            if (lowPose) {
                return penetrationLyingFrames;
            }
            if (contactGround->config().footGroundLimitEnabled &&
                contactGround->config().penetrationExcludeFeetWhenFootLimit) {
                return penetrationTrunkFrames;
            }
            return penetrationGroundFrames;
        }

        void resolvePenetrationGeoms() {
            auto resolveGeomByBodyNames = [this](const std::vector<std::string>& bodyNames,
                                                 std::vector<pinocchio::GeomIndex>* out) {
                out->clear();
                if (!hasCollisionGeometry) {
                    return;
                }
                std::vector<pinocchio::JointIndex> rootJoints;
                rootJoints.reserve(bodyNames.size());
                for (const auto& name : bodyNames) {
                    try {
                        const pinocchio::FrameIndex frameId = resolveTaskFrameId(model, name).first;
                        rootJoints.push_back(model.frames[frameId].parentJoint);
                    } catch (const std::exception&) {
                    }
                }
                std::sort(rootJoints.begin(), rootJoints.end());
                rootJoints.erase(std::unique(rootJoints.begin(), rootJoints.end()), rootJoints.end());

                for (pinocchio::GeomIndex gid = 0; gid < geomModel.ngeoms; ++gid) {
                    const auto& go = geomModel.geometryObjects[gid];
                    for (pinocchio::JointIndex rootJoint : rootJoints) {
                        if (isJointInSubtree(go.parentJoint, rootJoint)) {
                            out->push_back(gid);
                            break;
                        }
                    }
                }
                std::sort(out->begin(), out->end());
                out->erase(std::unique(out->begin(), out->end()), out->end());
            };

            const auto& cfg = options.contactGround;
            resolveGeomByBodyNames(cfg.robotFootBodies, &penetrationFootGeoms);
            resolveGeomByBodyNames(cfg.robotTrunkBodies, &penetrationTrunkGeoms);
            for (const auto& geomName : cfg.footCollisionGeoms) {
                for (pinocchio::GeomIndex gid = 0; gid < geomModel.ngeoms; ++gid) {
                    if (geomModel.geometryObjects[gid].name == geomName) {
                        penetrationFootGeoms.push_back(gid);
                    }
                }
            }
            std::sort(penetrationFootGeoms.begin(), penetrationFootGeoms.end());
            penetrationFootGeoms.erase(std::unique(penetrationFootGeoms.begin(), penetrationFootGeoms.end()),
                                       penetrationFootGeoms.end());

            penetrationGroundGeoms = penetrationFootGeoms;
            penetrationGroundGeoms.insert(penetrationGroundGeoms.end(), penetrationTrunkGeoms.begin(), penetrationTrunkGeoms.end());
            std::sort(penetrationGroundGeoms.begin(), penetrationGroundGeoms.end());
            penetrationGroundGeoms.erase(std::unique(penetrationGroundGeoms.begin(), penetrationGroundGeoms.end()),
                                         penetrationGroundGeoms.end());

            std::vector<pinocchio::GeomIndex> legGeoms;
            std::vector<pinocchio::GeomIndex> armGeoms;
            resolveGeomByBodyNames(cfg.robotLegBodies, &legGeoms);
            resolveGeomByBodyNames(cfg.robotArmBodies, &armGeoms);
            penetrationLyingGeoms = penetrationGroundGeoms;
            penetrationLyingGeoms.insert(penetrationLyingGeoms.end(), legGeoms.begin(), legGeoms.end());
            penetrationLyingGeoms.insert(penetrationLyingGeoms.end(), armGeoms.begin(), armGeoms.end());
            std::sort(penetrationLyingGeoms.begin(), penetrationLyingGeoms.end());
            penetrationLyingGeoms.erase(std::unique(penetrationLyingGeoms.begin(), penetrationLyingGeoms.end()),
                                        penetrationLyingGeoms.end());
        }

        const std::vector<pinocchio::GeomIndex>& activePenetrationGeoms(bool lowPose) const {
            if (!contactGround) {
                static const std::vector<pinocchio::GeomIndex> kEmpty;
                return kEmpty;
            }
            if (lowPose) {
                return penetrationLyingGeoms;
            }
            if (contactGround->config().footGroundLimitEnabled &&
                contactGround->config().penetrationExcludeFeetWhenFootLimit) {
                return penetrationTrunkGeoms;
            }
            return penetrationGroundGeoms;
        }

        bool useHppfclPenetration() const {
            if (!contactGround) {
                return false;
            }
            std::string mode = contactGround->config().pinocchioPenetrationMode;
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return mode != "frame_z";
        }

        bool isJointInSubtree(pinocchio::JointIndex jointId, pinocchio::JointIndex rootJoint) const {
            pinocchio::JointIndex cursor = jointId;
            while (cursor > 0 && cursor < model.njoints) {
                if (cursor == rootJoint) {
                    return true;
                }
                cursor = model.parents[cursor];
            }
            return false;
        }

        double estimatePenetrationDepthHppfcl(const std::vector<pinocchio::GeomIndex>& geoms, double margin, double groundZ) {
            if (!hasCollisionGeometry || !geomData || geoms.empty()) {
                return 0.0;
            }
            hpp::fcl::Halfspace floorPlane(Eigen::Vector3d::UnitZ(), groundZ);
            const hpp::fcl::Transform3f floorTf = hpp::fcl::Transform3f::Identity();
            hpp::fcl::DistanceRequest req(true);
            req.gjk_max_iterations     = 128;
            double minDist             = std::numeric_limits<double>::infinity();
            for (pinocchio::GeomIndex gid : geoms) {
                if (gid >= geomModel.ngeoms) {
                    continue;
                }
                const auto& go = geomModel.geometryObjects[gid];
                if (!go.geometry) {
                    continue;
                }
                hpp::fcl::DistanceResult out;
                const hpp::fcl::Transform3f worldTf = pinocchio::toFclTransform3f(geomData->oMg[gid]);
                const double dist = hpp::fcl::distance(go.geometry.get(), worldTf, &floorPlane, floorTf, req, out);
                minDist           = std::min(minDist, dist);
            }
            if (!std::isfinite(minDist)) {
                return 0.0;
            }
            return std::max(0.0, margin - minDist);
        }

        /// Pinocchio-only clearance: lift free-flyer z so monitored body frames clear groundZ+margin.
        void finalizeRobotState(const ContactGroundState* state = nullptr) {
            if (!contactGround || !contactGround->enabled() || !contactGround->config().fixRobotPenetration) {
                if (contactGround) {
                    contactGround->setLastRootLift(0.0);
                }
                return;
            }
            if (!hasRootFreeFlyer || qposPin.size() < 3) {
                contactGround->setLastRootLift(0.0);
                return;
            }

            const bool lowPose = state != nullptr ? state->lowPose : contactGround->isLowPose();
            const auto& frames = activePenetrationFrames(lowPose);
            const auto& geoms  = activePenetrationGeoms(lowPose);
            if (frames.empty() && geoms.empty()) {
                contactGround->setLastRootLift(0.0);
                return;
            }

            const double margin      = lowPose
                ? contactGround->config().lyingPenetrationMargin
                : contactGround->config().penetrationMargin;
            const double groundZ     = contactGround->config().groundZ;
            const double footClear   = std::max(0.0, contactGround->config().pinocchioFootClearance);
            const int iterations     = std::max(1, contactGround->config().penetrationMaxIterations);
            double totalLift         = 0.0;

            for (int iter = 0; iter < iterations; ++iter) {
                syncDataFromQpos();
                double depth = 0.0;
                if (useHppfclPenetration() && !geoms.empty() && hasCollisionGeometry && geomData) {
                    pinocchio::updateGeometryPlacements(model, *data, geomModel, *geomData);
                    depth = estimatePenetrationDepthHppfcl(geoms, margin, groundZ);
                } else {
                    double minZ = std::numeric_limits<double>::infinity();
                    for (pinocchio::FrameIndex frameId : frames) {
                        if (frameId >= model.nframes) {
                            continue;
                        }
                        double z = data->oMf[frameId].translation().z();
                        // Foot body origins sit above the sole; clear by configured thickness.
                        if (footClear > 0.0 &&
                            std::binary_search(penetrationFootFrames.begin(), penetrationFootFrames.end(), frameId)) {
                            z -= footClear;
                        }
                        minZ = std::min(minZ, z);
                    }
                    if (!std::isfinite(minZ)) {
                        break;
                    }
                    depth = std::max(0.0, groundZ + margin - minZ);
                }
                if (depth <= 1e-9) {
                    break;
                }
                qposPin[2] += depth;
                totalLift += depth;
            }
            if (totalLift > 0.0) {
                syncDataFromQpos();
            } else {
                qpos = pinocchioToMujocoQpos(qposPin);
            }
            contactGround->setLastRootLift(totalLift);
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

        void updateTaskTargets(const HumanFrame& frame) {
            auto fill = [this, &frame](std::vector<PinTaskRuntime>* tasks) {
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

        double computeTaskError(const std::vector<PinTaskRuntime>& tasks) const {
            double sqErr = 0.0;
            for (const auto& task : tasks) {
                pinocchio::SE3 currPose;
                if (task.useJointPose) {
                    if (task.jointId >= model.njoints) {
                        continue;
                    }
                    currPose = data->oMi[task.jointId];
                } else {
                    if (task.frameId >= model.nframes) {
                        continue;
                    }
                    currPose = data->oMf[task.frameId];
                }

                Eigen::Quaterniond targetRot = task.targetRot;
                targetRot.normalize();
                const pinocchio::SE3 targetPose(targetRot.toRotationMatrix(), task.targetPos);
                const Eigen::Matrix<double, 6, 1> error = pinocchio::log6(currPose.inverse() * targetPose).toVector();
                sqErr += error.squaredNorm();
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

            double currError = computeTaskError(tasks);
            solver::QPSolver solver(options.solverName);
            const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nv, nv);

            for (int iter = 0; iter < options.maxIterations; ++iter) {
                solver::QPData qp;
                qp.reset(nv, nv);

                qp.CI.setIdentity();
                qp.ciLb.setConstant(-1e9);
                qp.ciUb.setConstant(1e9);

                if (options.useVelocityLimit) {
                    qp.ciLb.setConstant(-options.velocityLimit * dt);
                    qp.ciUb.setConstant(options.velocityLimit * dt);
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
                        qp.ciLb[vadr] = std::max(qp.ciLb[vadr], qmin - qposPin[qadr]);
                        qp.ciUb[vadr] = std::min(qp.ciUb[vadr], qmax - qposPin[qadr]);
                    }
                }

                pinocchio::computeJointJacobians(model, *data, qposPin);

                for (const auto& task : tasks) {
                    pinocchio::SE3 currPose;
                    Eigen::Matrix<double, 6, Eigen::Dynamic> jacobianLocal(6, nv);

                    if (task.useJointPose) {
                        if (task.jointId >= model.njoints) {
                            continue;
                        }
                        jacobianLocal = pinocchio::getJointJacobian(model, *data, task.jointId, pinocchio::LOCAL);
                        currPose      = data->oMi[task.jointId];
                    } else {
                        if (task.frameId >= model.nframes) {
                            continue;
                        }
                        jacobianLocal.setZero();
                        pinocchio::computeFrameJacobian(model, *data, qposPin, task.frameId, pinocchio::LOCAL, jacobianLocal);
                        currPose = data->oMf[task.frameId];
                    }

                    Eigen::Quaterniond targetRot = task.targetRot;
                    targetRot.normalize();
                    const pinocchio::SE3 targetPose(targetRot.toRotationMatrix(), task.targetPos);
                    const pinocchio::SE3 T_bt = currPose.inverse() * targetPose;
                    const pinocchio::SE3 T_tb = targetPose.inverse() * currPose;

                    const Eigen::Matrix<double, 6, 1> error = pinocchio::log6(T_bt).toVector();
                    const Eigen::Matrix<double, 6, 6> jlog  = pinocchio::Jlog6(T_tb);
                    const Eigen::MatrixXd taskJacobian      = -jlog * jacobianLocal;

                    Eigen::MatrixXd weightedJacobian = taskJacobian;
                    weightedJacobian.topRows(3) *= task.posWeight;
                    weightedJacobian.bottomRows(3) *= task.rotWeight;

                    Eigen::Matrix<double, 6, 1> weightedError = -error;
                    weightedError.head<3>() *= task.posWeight;
                    weightedError.tail<3>() *= task.rotWeight;

                    const double lmMu = options.taskLmDamping * weightedError.squaredNorm();
                    qp.H.noalias() += weightedJacobian.transpose() * weightedJacobian + lmMu * I;
                    qp.g.noalias() += -(weightedError.transpose() * weightedJacobian).transpose();
                }

                qp.H.diagonal().array() += options.damping;

                const solver::QPOutput& out = solver.solve(qp);
                if (out.status != solver::QPStatus::kOptimal) {
                    throw std::runtime_error("QP solver failed while retargeting.");
                }

                const Eigen::VectorXd deltaQ = out.x;
                qvel                         = deltaQ / dt;
                qposPin                      = pinocchio::integrate(model, qposPin, deltaQ);
                syncDataFromQpos();

                const double nextError = computeTaskError(tasks);
                if (currError - nextError <= options.progressThreshold) {
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

        Eigen::VectorXd retargetPrepared(const HumanFrame& prepared, bool finalize) {
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
            retargetPrepared(prepareRetargetInput(humanFrame, offsetToGround), /*finalize=*/false);
            options.maxIterations = savedMaxIter;
            return qpos;
        }

        Eigen::VectorXd retargetPreparedLightIkImpl(const HumanFrame& preparedFrame, int maxIterations) {
            if (maxIterations <= 0) {
                return qpos;
            }

            const int savedMaxIter = options.maxIterations;
            options.maxIterations  = maxIterations;
            retargetPrepared(preparedFrame, /*finalize=*/false);
            options.maxIterations = savedMaxIter;
            return qpos;
        }
    };

    PinocchioRetargetBackend::PinocchioRetargetBackend(const std::filesystem::path& robotModelPath, IkConfig ikConfig,
                                                       RetargetOptions options)
        : impl_(std::make_unique<Impl>(robotModelPath, std::move(ikConfig), std::move(options))) {}

    PinocchioRetargetBackend::~PinocchioRetargetBackend() = default;

    Eigen::VectorXd PinocchioRetargetBackend::retargetFrame(const HumanFrame& humanFrame, bool offsetToGround) {
        return impl_->retargetPrepared(impl_->prepareRetargetInput(humanFrame, offsetToGround), /*finalize=*/true);
    }

    HumanFrame PinocchioRetargetBackend::prepareHumanFrame(const HumanFrame& humanFrame, bool offsetToGround) const {
        return impl_->prepareHumanFrame(humanFrame, offsetToGround);
    }

    HumanFrame PinocchioRetargetBackend::prepareRetargetInput(const HumanFrame& humanFrame, bool offsetToGround) {
        return impl_->prepareRetargetInput(humanFrame, offsetToGround);
    }

    HumanFrame PinocchioRetargetBackend::prepareRetargetInput(
        const HumanFrame& humanFrame,
        const ContactGroundState& contactState,
        bool offsetToGround) {
        return impl_->prepareRetargetInput(humanFrame, contactState, offsetToGround);
    }

    Eigen::VectorXd PinocchioRetargetBackend::retargetPreparedFrame(const HumanFrame&, const HumanFrame& preparedFrame) {
        return impl_->retargetPrepared(preparedFrame, /*finalize=*/false);
    }

    Eigen::VectorXd PinocchioRetargetBackend::retargetPreparedLightIk(const HumanFrame&, const HumanFrame& preparedFrame,
                                                                      int maxIterations) {
        return impl_->retargetPreparedLightIkImpl(preparedFrame, maxIterations);
    }

    Eigen::VectorXd PinocchioRetargetBackend::retargetLightIk(const HumanFrame& humanFrame, bool offsetToGround, int maxIterations) {
        return impl_->retargetLightIkImpl(humanFrame, offsetToGround, maxIterations);
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

    void PinocchioRetargetBackend::finalizeContact() { impl_->finalizeRobotState(); }

    void PinocchioRetargetBackend::finalizeContact(const ContactGroundState& state) {
        impl_->finalizeRobotState(&state);
    }

    ContactGroundState PinocchioRetargetBackend::contactGroundState() const {
        return impl_->contactGround && impl_->contactGround->enabled()
            ? impl_->contactGround->state()
            : ContactGroundState{};
    }

    const Eigen::VectorXd& PinocchioRetargetBackend::currentQpos() const { return impl_->qpos; }

    bool PinocchioRetargetBackend::hasRootFreeFlyer() const { return impl_->hasRootFreeFlyer; }

    const std::vector<ScalarJointCoordinate>& PinocchioRetargetBackend::scalarJointCoordinates() const {
        return impl_->scalarJointCoordinates;
    }

    void PinocchioRetargetBackend::setMotionFps(double fps) {
        if (impl_->contactGround) {
            impl_->contactGround->setFps(fps);
        }
    }

}  // namespace gmr
