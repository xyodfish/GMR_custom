#include "gmr/retarget/contact_ground.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/QR>

namespace gmr {

    bool ContactGroundState::hasSupportContact() const {
        return std::any_of(
            footContacts.begin(),
            footContacts.end(),
            [](const auto& item) { return item.second; });
    }

    namespace {

        std::vector<int> resolveBodyIds(const mjModel* model, const std::vector<std::string>& names) {
            std::vector<int> ids;
            ids.reserve(names.size());
            for (const auto& name : names) {
                const int bid = mj_name2id(model, mjOBJ_BODY, name.c_str());
                if (bid >= 0) {
                    ids.push_back(bid);
                }
            }
            return ids;
        }

        std::vector<int> collectFootBodySubtree(const mjModel* model, const std::vector<std::string>& rootNames) {
            std::set<int> bodyIds;
            for (const auto& name : rootNames) {
                const int rootId = mj_name2id(model, mjOBJ_BODY, name.c_str());
                if (rootId < 0) {
                    continue;
                }
                std::vector<int> stack{rootId};
                while (!stack.empty()) {
                    const int bodyId = stack.back();
                    stack.pop_back();
                    if (!bodyIds.insert(bodyId).second) {
                        continue;
                    }
                    for (int childId = 1; childId < model->nbody; ++childId) {
                        if (model->body_parentid[childId] == bodyId) {
                            stack.push_back(childId);
                        }
                    }
                }
            }
            return {bodyIds.begin(), bodyIds.end()};
        }

        std::vector<int> collectGeomIdsForBodies(const mjModel* model, const std::vector<int>& bodyIds) {
            std::set<int> bodySet(bodyIds.begin(), bodyIds.end());
            std::vector<int> geomIds;
            for (int geomId = 0; geomId < model->ngeom; ++geomId) {
                if (bodySet.count(model->geom_bodyid[geomId]) > 0) {
                    geomIds.push_back(geomId);
                }
            }
            return geomIds;
        }

        std::vector<int> collisionGeomIds(const mjModel* model, std::vector<int> geomIds) {
            geomIds.erase(
                std::remove_if(
                    geomIds.begin(),
                    geomIds.end(),
                    [&](int geomId) {
                        return model->geom_contype[geomId] == 0 && model->geom_conaffinity[geomId] == 0;
                    }),
                geomIds.end());
            return geomIds;
        }

        std::vector<int> resolveExplicitFootGeomIds(const mjModel* model, const std::vector<std::string>& geomNames) {
            std::vector<int> ids;
            for (const auto& name : geomNames) {
                const int gid = mj_name2id(model, mjOBJ_GEOM, name.c_str());
                if (gid >= 0) {
                    ids.push_back(gid);
                }
            }
            return ids;
        }

        std::vector<int> mergeUnique(std::vector<int> a, const std::vector<int>& b) {
            a.insert(a.end(), b.begin(), b.end());
            std::sort(a.begin(), a.end());
            a.erase(std::unique(a.begin(), a.end()), a.end());
            return a;
        }

        double measureGroundPenetrationDepth(const mjModel* model, mjData* data, const std::vector<int>& geomIds, int floorGeomId,
                                             double penetrationMargin) {
            if (geomIds.empty()) {
                return 0.0;
            }

            // 把当前 qpos 转成最新的 MuJoCo 世界几何状态；
            mj_forward(model, data);
            if (floorGeomId < 0) {
                double minZ = std::numeric_limits<double>::infinity();
                for (int geomId : geomIds) {
                    minZ = std::min(minZ, static_cast<double>(data->geom_xpos[3 * geomId + 2]));
                }
                return std::max(0.0, penetrationMargin - minZ);
            }

            mjtNum fromto[6] = {0};
            double maxDepth  = 0.0;
            for (int geomId : geomIds) {
                const mjtNum dist = mj_geomDistance(model, data, geomId, floorGeomId, 10.0, fromto);
                if (static_cast<double>(dist) < penetrationMargin) {
                    maxDepth = std::max(maxDepth, penetrationMargin - static_cast<double>(dist));
                }
            }
            return maxDepth;
        }

        struct GroundSample {
            double distance = std::numeric_limits<double>::infinity();
            int geomId = -1;
            Eigen::Vector3d point = Eigen::Vector3d::Zero();
        };

        GroundSample minimumGroundSample(
            const mjModel* model,
            mjData* data,
            const std::vector<int>& geomIds,
            int floorGeomId) {
            GroundSample sample;
            if (geomIds.empty() || floorGeomId < 0) {
                return sample;
            }

            mjtNum fromto[6] = {0};
            for (int geomId : geomIds) {
                const double distance = static_cast<double>(
                    mj_geomDistance(model, data, geomId, floorGeomId, 10.0, fromto));
                if (distance < sample.distance) {
                    sample.distance = distance;
                    sample.geomId = geomId;
                    sample.point = Eigen::Vector3d(fromto[0], fromto[1], fromto[2]);
                }

            }

            return sample;
        }

        double fixRobotGroundPenetration(mjModel* model, mjData* data, const std::vector<int>& geomIds, int floorGeomId,
                                         double penetrationMargin, int maxIterations) {
            if (model->nq < 3 || geomIds.empty()) {
                return 0.0;
            }
            double totalLift     = 0.0;
            const int iterations = std::max(1, maxIterations);
            for (int iter = 0; iter < iterations; ++iter) {
                //为了让所有被监控 geom 距离地面至少 penetrationMargin，需要把 root z 抬高多少
                const double depth = measureGroundPenetrationDepth(model, data, geomIds, floorGeomId, penetrationMargin);
                if (depth <= 1e-9) {
                    break;
                }
                data->qpos[2] += depth;
                totalLift += depth;
            }
            if (totalLift > 0.0) {
                mj_forward(model, data);
            }
            return totalLift;
        }

        double clip01(double v) {
            return std::min(1.0, std::max(1e-6, v));
        }

        std::vector<std::string> mergeUniqueNames(std::vector<std::string> a, const std::vector<std::string>& b) {
            a.insert(a.end(), b.begin(), b.end());
            std::sort(a.begin(), a.end());
            a.erase(std::unique(a.begin(), a.end()), a.end());
            return a;
        }

    }  // namespace

    void ContactGroundPipeline::cachePenetrationBodyNames() {
        trunkBodyNames_  = config_.robotTrunkBodies;
        groundBodyNames_ = mergeUniqueNames(config_.robotFootBodies, config_.robotTrunkBodies);
        lyingBodyNames_  = mergeUniqueNames(mergeUniqueNames(groundBodyNames_, config_.robotLegBodies), config_.robotArmBodies);
    }

    ContactGroundPipeline::ContactGroundPipeline(ContactGroundConfig config, const mjModel* model, double fps)
        : config_(std::move(config)),
          model_(model),
          lpfAlphaAt30Fps_(config_.lpfAlpha),
          footLockAlphaAt30Fps_(config_.footLockEmaAlpha) {
        setFps(fps);
        cachePenetrationBodyNames();
        footPosBuf_.clear();
        if (!model_) {
            return;
        }

        const std::vector<int> footBodyIds    = resolveBodyIds(model_, config_.robotFootBodies);
        const std::vector<int> subtreeBodyIds = collectFootBodySubtree(model_, config_.robotFootBodies);
        const std::vector<int> trunkBodyIds   = resolveBodyIds(model_, config_.robotTrunkBodies);
        const std::vector<int> legBodyIds     = resolveBodyIds(model_, config_.robotLegBodies);
        const std::vector<int> armBodyIds     = resolveBodyIds(model_, config_.robotArmBodies);

        const std::vector<int> explicitGeomIds = resolveExplicitFootGeomIds(model_, config_.footCollisionGeoms);
        const std::vector<int> subtreeGeomIds = collisionGeomIds(
            model_,
            collectGeomIdsForBodies(model_, subtreeBodyIds));
        footGeomIds_ = explicitGeomIds.empty() ? subtreeGeomIds : explicitGeomIds;

        const std::set<int> explicitGeomSet(explicitGeomIds.begin(), explicitGeomIds.end());
        const std::size_t nMappedFeet = std::min(config_.footBodies.size(), config_.robotFootBodies.size());
        for (std::size_t i = 0; i < nMappedFeet; ++i) {
            const std::vector<int> bodyIds = collectFootBodySubtree(model_, {config_.robotFootBodies[i]});
            std::vector<int> geomIds = collisionGeomIds(
                model_,
                collectGeomIdsForBodies(model_, bodyIds));
            if (!explicitGeomSet.empty()) {
                geomIds.erase(
                    std::remove_if(
                        geomIds.begin(),
                        geomIds.end(),
                        [&](int geomId) { return explicitGeomSet.count(geomId) == 0; }),
                    geomIds.end());
            }

            footGeomIdsByContact_[config_.footBodies[i]] = std::move(geomIds);
        }

        trunkGeomIds_                          = collectGeomIdsForBodies(model_, trunkBodyIds);
        legGeomIds_                            = collectGeomIdsForBodies(model_, legBodyIds);
        armGeomIds_                            = collectGeomIdsForBodies(model_, armBodyIds);
        groundGeomIds_                         = mergeUnique(footGeomIds_, trunkGeomIds_);
        lyingGroundGeomIds_                    = mergeUnique(mergeUnique(groundGeomIds_, legGeomIds_), armGeomIds_);
        floorGeomId_                           = mj_name2id(model_, mjOBJ_GEOM, config_.floorGeomName.c_str());

        const std::vector<int> legSubtreeBodies = collectFootBodySubtree(
            model_,
            mergeUniqueNames(config_.robotLegBodies, config_.robotFootBodies));
        const std::set<int> legBodySet(legSubtreeBodies.begin(), legSubtreeBodies.end());
        for (int jointId = 0; jointId < model_->njnt; ++jointId) {
            const int type = model_->jnt_type[jointId];
            if ((type == mjJNT_HINGE || type == mjJNT_SLIDE) &&
                legBodySet.count(model_->jnt_bodyid[jointId]) > 0) {
                legJointIds_.push_back(jointId);
            }

        }
    }

    void ContactGroundPipeline::setFps(double fps) {
        if (fps > 0.0) {
            fps_ = fps;
        }

        const double exponent = 30.0 / fps_;
        config_.lpfAlpha = 1.0 - std::pow(1.0 - clip01(lpfAlphaAt30Fps_), exponent);
        config_.footLockEmaAlpha = 1.0 - std::pow(1.0 - clip01(footLockAlphaAt30Fps_), exponent);
    }

    HumanFrame ContactGroundPipeline::processHumanFrame(const HumanFrame& humanData) {
        return processHumanFrameImpl(humanData, nullptr);
    }

    HumanFrame ContactGroundPipeline::processHumanFrame(
        const HumanFrame& humanData,
        const ContactGroundState& contactState) {
        return processHumanFrameImpl(humanData, &contactState);
    }

    HumanFrame ContactGroundPipeline::processHumanFrameImpl(
        const HumanFrame& humanData,
        const ContactGroundState* contactState) {
        if (!config_.enabled) {
            return humanData;
        }

        std::unordered_map<std::string, Eigen::Vector3d> footPositions;
        for (const auto& name : config_.footBodies) {
            auto it = humanData.find(name);
            if (it != humanData.end()) {
                footPositions[name] = it->second.position;
            }

        }

        if (footPositions.empty()) {
            return humanData;
        }

        double observedMinFootZ = std::numeric_limits<double>::infinity();
        for (const auto& [_, pos] : footPositions) {
            observedMinFootZ = std::min(observedMinFootZ, pos.z());
        }

        if (!groundAlignInitialized_) {
            if (observedMinFootZ > config_.heightThreshold) {
                const double targetZ = config_.groundZ + config_.groundMargin;
                groundAlignOffset_ = observedMinFootZ - targetZ;
            }

            groundAlignInitialized_ = true;
        }

        std::unordered_map<std::string, bool> contacts;
        if (contactState != nullptr) {
            for (const auto& [name, pos] : footPositions) {
                const auto it = contactState->footContacts.find(name);
                if (it == contactState->footContacts.end()) {
                    throw std::runtime_error("Explicit contact state is missing foot: " + name);
                }

                contacts[name] = it->second;
            }
        } else {
            const int velWindow = std::max(2, config_.velWindow);
            footPosBuf_.push_back(footPositions);
            while (static_cast<int>(footPosBuf_.size()) > velWindow) {
                footPosBuf_.pop_front();
            }

            const double dt = std::max(
                (static_cast<double>(footPosBuf_.size()) - 1.0) / fps_,
                1.0 / fps_);
            for (const auto& [name, pos] : footPositions) {
                const bool wasContact = lastContacts_.count(name) > 0 && lastContacts_.at(name);
                const double zLimit = wasContact
                    ? config_.heightOffThreshold
                    : config_.heightThreshold;
                const bool zOk = pos.z() - groundAlignOffset_ <= zLimit;

                bool velOk = true;
                double verticalSpeed = 0.0;
                if (footPosBuf_.size() >= 2) {
                    const Eigen::Vector3d displacement = pos - footPosBuf_.front().at(name);
                    const double speed = displacement.norm() / dt;
                    verticalSpeed = displacement.z() / dt;
                    velOk = speed <= config_.velThreshold;
                }

                const double liftoffHeight = config_.groundZ + config_.groundMargin +
                    0.25 * config_.heightThreshold;
                const bool lifting = wasContact &&
                    pos.z() - groundAlignOffset_ > liftoffHeight &&
                    verticalSpeed > 0.0;
                contacts[name] = zOk && velOk && !lifting;
            }
        }

        lastContacts_ = contacts;

        if (observedMinFootZ - groundAlignOffset_ > config_.airborneHeightThreshold) {
            groundAlignOffset_ *= config_.airborneOffsetDecay;
        }

        std::vector<double> activeZ;
        activeZ.reserve(contacts.size());
        for (const auto& [name, inContact] : contacts) {
            if (!inContact) {
                continue;
            }
            auto it = humanData.find(name);
            if (it != humanData.end()) {
                activeZ.push_back(it->second.position.z());
            }

        }

        if (!activeZ.empty()) {
            const double targetZ   = config_.groundZ + config_.groundMargin;
            const double rawOffset = *std::min_element(activeZ.begin(), activeZ.end()) - targetZ;
            const double alpha     = clip01(config_.lpfAlpha);
            groundAlignOffset_     = alpha * rawOffset + (1.0 - alpha) * groundAlignOffset_;
        }

        lastMinFootZ_ = observedMinFootZ - groundAlignOffset_;
        auto hipIt = humanData.find(config_.humanRootName);
        if (hipIt != humanData.end()) {
            lastHumanHipZ_ = hipIt->second.position.z() - groundAlignOffset_;
        }

        const Eigen::Vector3d offsetVec(0.0, 0.0, groundAlignOffset_);
        HumanFrame aligned = humanData;
        for (auto& [bodyName, state] : aligned) {
            state.position -= offsetVec;
            if (config_.enableFootLock && contacts.count(bodyName) > 0 && contacts.at(bodyName)) {
                if (lockedFeet_.count(bodyName) == 0) {
                    lockedFeet_[bodyName] = state.position;
                } else {
                    const double alpha    = clip01(config_.footLockEmaAlpha);
                    lockedFeet_[bodyName] = (1.0 - alpha) * lockedFeet_[bodyName] + alpha * state.position;
                }
                state.position = lockedFeet_[bodyName];
            } else {
                lockedFeet_.erase(bodyName);
            }

        }

        return aligned;
    }

    bool ContactGroundPipeline::isLowPose() const {
        // hip 很低
        if (lastHumanHipZ_ <= config_.lyingHipHeightThreshold) {
            return true;
        }

        // 脚比较低，同时 hip 也不高 能覆盖 蹲下 跪地 低位等动作
        if (lastMinFootZ_ <= config_.lowPoseFootHeightThreshold) {
            return lastHumanHipZ_ <= config_.lowPoseMaxHipHeight;
        }
        return false;
    }

    double ContactGroundPipeline::activePenetrationMargin() const {
        return isLowPose() ? config_.lyingPenetrationMargin : config_.penetrationMargin;
    }

    const std::vector<std::string>& ContactGroundPipeline::activePenetrationBodyNames() const {
        if (isLowPose()) {
            return lyingBodyNames_;
        }
        if (config_.footGroundLimitEnabled && config_.penetrationExcludeFeetWhenFootLimit) {
            return trunkBodyNames_;
        }
        return groundBodyNames_;
    }

    ContactGroundState ContactGroundPipeline::state() const {
        return ContactGroundState{lastContacts_, isLowPose()};
    }

    std::pair<const std::vector<int>&, double> ContactGroundPipeline::penetrationTargets(bool lowPose) const {
        if (lowPose) {
            return {lyingGroundGeomIds_, config_.lyingPenetrationMargin};
        }
        if (config_.footGroundLimitEnabled && config_.penetrationExcludeFeetWhenFootLimit) {
            return {trunkGeomIds_, config_.penetrationMargin};
        }
        return {groundGeomIds_, config_.penetrationMargin};
    }

    // IK / retarget 解完机器人姿态之后，检查机器人指定 geom 是否穿地；
    // 如果穿地，就直接把机器人 root 的 z 方向 qpos[2] 抬高一点。
    // 正常姿态：检查脚部 + 躯干 geom（foot_ground_limit 时仅躯干）；
    // 低姿态：额外检查腿部 + 手臂 geom。
    double ContactGroundPipeline::fixRobotPenetration(mjModel* model, mjData* data) {
        return fixRobotPenetration(model, data, state());
    }

    double ContactGroundPipeline::fixRobotPenetration(
        mjModel* model,
        mjData* data,
        const ContactGroundState& state) {
        lastRootLift_ = 0.0;
        if (!config_.enabled || !config_.fixRobotPenetration || model == nullptr || data == nullptr) {
            return 0.0;
        }

        std::vector<int> supportGeomIds;
        for (const auto& [contactName, active] : state.footContacts) {
            const auto it = footGeomIdsByContact_.find(contactName);
            if (active && it != footGeomIdsByContact_.end()) {
                supportGeomIds = mergeUnique(std::move(supportGeomIds), it->second);
            }
        }

        if (config_.snapSupportToGround && !supportGeomIds.empty()) {
            mj_forward(model, data);
            const GroundSample lowestSupport = minimumGroundSample(
                model,
                data,
                supportGeomIds,
                floorGeomId_);
            if (lowestSupport.distance > config_.penetrationMargin && model->nq >= 3) {
                // Land the lowest support foot. Using the highest foot here would push
                // the lower one through the floor before the leg IK even starts.
                const double correction = config_.penetrationMargin - lowestSupport.distance;
                data->qpos[2] += correction;
                lastRootLift_ += correction;
                mj_forward(model, data);
            }

            correctSwingFootPenetration(model, data, state);
        }

        // The final root correction is one-sided: it can only remove penetration.
        const auto [geomIds, margin] = penetrationTargets(state.lowPose);
        lastRootLift_ +=
            fixRobotGroundPenetration(model, data, geomIds, floorGeomId_, margin, config_.penetrationMaxIterations);
        return lastRootLift_;
    }

    void ContactGroundPipeline::correctSwingFootPenetration(
        mjModel* model,
        mjData* data,
        const ContactGroundState& state) {
        if (legJointIds_.empty()) {
            return;
        }

        std::vector<mjtNum> jacobian(3 * model->nv, 0.0);
        for (int iteration = 0; iteration < config_.penetrationMaxIterations; ++iteration) {
            mj_forward(model, data);
            std::vector<Eigen::RowVectorXd> rows;
            std::vector<double> targets;
            for (const auto& [contactName, geomIds] : footGeomIdsByContact_) {
                const GroundSample sample = minimumGroundSample(model, data, geomIds, floorGeomId_);
                if (!std::isfinite(sample.distance)) {
                    continue;
                }

                const auto contactIt = state.footContacts.find(contactName);
                const bool support = contactIt != state.footContacts.end() && contactIt->second;
                if (!support && sample.distance >= -1e-5) {
                    continue;
                }

                mj_jac(
                    model,
                    data,
                    jacobian.data(),
                    nullptr,
                    sample.point.data(),
                    model->geom_bodyid[sample.geomId]);
                Eigen::RowVectorXd row(legJointIds_.size());
                for (std::size_t j = 0; j < legJointIds_.size(); ++j) {
                    row[j] = jacobian[2 * model->nv + model->jnt_dofadr[legJointIds_[j]]];
                }

                rows.push_back(std::move(row));
                targets.push_back((support ? config_.penetrationMargin : 0.0) - sample.distance);
            }

            if (rows.empty()) {
                return;
            }

            Eigen::MatrixXd J(rows.size(), legJointIds_.size());
            Eigen::VectorXd target(rows.size());
            for (std::size_t r = 0; r < rows.size(); ++r) {
                J.row(r) = rows[r];
                target[r] = targets[r];
            }

            Eigen::VectorXd dq = J.completeOrthogonalDecomposition().solve(target);
            const double maxStep = 0.03 * 30.0 / fps_;
            dq = dq.cwiseMax(-maxStep).cwiseMin(maxStep);
            if (dq.norm() < 1e-7) {
                return;
            }

            for (std::size_t j = 0; j < legJointIds_.size(); ++j) {
                const int jointId = legJointIds_[j];
                const int qadr = model->jnt_qposadr[jointId];
                data->qpos[qadr] += dq[j];
                if (model->jnt_limited[jointId]) {
                    data->qpos[qadr] = std::clamp(
                        data->qpos[qadr],
                        model->jnt_range[2 * jointId],
                        model->jnt_range[2 * jointId + 1]);
                }

            }

        }
    }

}  // namespace gmr
