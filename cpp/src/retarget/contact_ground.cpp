#include "gmr/retarget/contact_ground.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace gmr {

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

    }  // namespace

    ContactGroundPipeline::ContactGroundPipeline(ContactGroundConfig config, const mjModel* model, double fps)
        : config_(std::move(config)), model_(model), fps_(fps > 0.0 ? fps : 30.0) {
        if (!model_) {
            return;
        }

        const std::vector<int> footBodyIds    = resolveBodyIds(model_, config_.robotFootBodies);
        const std::vector<int> subtreeBodyIds = collectFootBodySubtree(model_, config_.robotFootBodies);
        const std::vector<int> trunkBodyIds   = resolveBodyIds(model_, config_.robotTrunkBodies);
        const std::vector<int> legBodyIds     = resolveBodyIds(model_, config_.robotLegBodies);
        const std::vector<int> armBodyIds     = resolveBodyIds(model_, config_.robotArmBodies);

        const std::vector<int> explicitGeomIds = resolveExplicitFootGeomIds(model_, config_.footCollisionGeoms);
        const std::vector<int> subtreeGeomIds  = collectGeomIdsForBodies(model_, subtreeBodyIds);
        footGeomIds_                           = mergeUnique(explicitGeomIds, subtreeGeomIds);
        trunkGeomIds_                          = collectGeomIdsForBodies(model_, trunkBodyIds);
        legGeomIds_                            = collectGeomIdsForBodies(model_, legBodyIds);
        armGeomIds_                            = collectGeomIdsForBodies(model_, armBodyIds);
        groundGeomIds_                         = mergeUnique(footGeomIds_, trunkGeomIds_);
        lyingGroundGeomIds_                    = mergeUnique(mergeUnique(groundGeomIds_, legGeomIds_), armGeomIds_);
        floorGeomId_                           = mj_name2id(model_, mjOBJ_GEOM, config_.floorGeomName.c_str());
        footPosBuf_.clear();
    }

    void ContactGroundPipeline::setFps(double fps) {
        if (fps > 0.0) {
            fps_ = fps;
        }
    }

    HumanFrame ContactGroundPipeline::processHumanFrame(const HumanFrame& humanData) {
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

        const int velWindow = std::max(2, config_.velWindow);
        footPosBuf_.push_back(footPositions);
        while (static_cast<int>(footPosBuf_.size()) > velWindow) {
            footPosBuf_.pop_front();
        }

        std::unordered_map<std::string, bool> contacts;
        const double dt = std::max((static_cast<double>(footPosBuf_.size()) - 1.0) / fps_, 1.0 / fps_);
        for (const auto& [name, pos] : footPositions) {
            const bool wasContact = lastContacts_.count(name) > 0 && lastContacts_.at(name);
            const double zLimit   = wasContact ? config_.heightOffThreshold : config_.heightThreshold;
            const bool zOk        = pos.z() <= zLimit;

            bool velOk = true;
            if (footPosBuf_.size() >= 2) {
                const Eigen::Vector3d displacement = pos - footPosBuf_.front().at(name);
                const double speed                 = displacement.norm() / dt;
                velOk                              = speed <= config_.velThreshold;
            }
            contacts[name] = zOk && velOk;
        }
        lastContacts_ = contacts;
        lastMinFootZ_ = std::numeric_limits<double>::infinity();
        for (const auto& [_, pos] : footPositions) {
            lastMinFootZ_ = std::min(lastMinFootZ_, pos.z());
        }

        auto hipIt = humanData.find(config_.humanRootName);
        if (hipIt != humanData.end()) {
            lastHumanHipZ_ = hipIt->second.position.z();
        }

        double maxFootZ = -std::numeric_limits<double>::infinity();
        for (const auto& [_, pos] : footPositions) {
            maxFootZ = std::max(maxFootZ, pos.z());
        }
        if (maxFootZ > config_.airborneHeightThreshold) {
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

    // IK / retarget 解完机器人姿态之后，检查机器人指定 geom 是否穿地；
    // 如果穿地，就直接把机器人 root 的 z 方向 qpos[2] 抬高一点。
    // 正常姿态：检查脚部 + 躯干 geom 是否穿地； 低姿态：额外检查腿部 + 手臂 geom 是否穿地。
    double ContactGroundPipeline::fixRobotPenetration(mjModel* model, mjData* data) {
        lastRootLift_ = 0.0;
        if (!config_.enabled || !config_.fixRobotPenetration || model == nullptr || data == nullptr) {
            return 0.0;
        }

        const bool lowPose              = isLowPose();
        const std::vector<int>& geomIds = lowPose ? lyingGroundGeomIds_ : groundGeomIds_;
        const double margin             = lowPose ? config_.lyingPenetrationMargin : config_.penetrationMargin;
        lastRootLift_ = fixRobotGroundPenetration(model, data, geomIds, floorGeomId_, margin, config_.penetrationMaxIterations);
        return lastRootLift_;
    }

}  // namespace gmr
