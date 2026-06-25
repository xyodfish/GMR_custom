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

        const std::vector<int> explicitGeomIds = resolveExplicitFootGeomIds(model_, config_.footCollisionGeoms);
        const std::vector<int> subtreeGeomIds  = collectGeomIdsForBodies(model_, subtreeBodyIds);
        footGeomIds_                           = mergeUnique(explicitGeomIds, subtreeGeomIds);
        trunkGeomIds_                          = collectGeomIdsForBodies(model_, trunkBodyIds);
        legGeomIds_                            = collectGeomIdsForBodies(model_, legBodyIds);
        groundGeomIds_                         = mergeUnique(footGeomIds_, trunkGeomIds_);
        lyingGroundGeomIds_                    = mergeUnique(groundGeomIds_, legGeomIds_);
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

        // 把来的数据加入footPos队列 保留长度为 velWindow
        const int velWindow = std::max(2, config_.velWindow);
        footPosBuf_.push_back(footPositions);
        while (static_cast<int>(footPosBuf_.size()) > velWindow) {
            footPosBuf_.pop_front();
        }

        std::unordered_map<std::string, bool> contacts;
        const double dt = std::max((static_cast<double>(footPosBuf_.size()) - 1.0) / fps_, 1.0 / fps_);
        for (const auto& [name, pos] : footPositions) {
            const bool wasContact = lastContacts_.count(name) > 0 && lastContacts_.at(name);

            //已经接触的脚，用更高一点的离地阈值，避免 contact 状态频繁抖动。
            const double zLimit = wasContact ? config_.heightOffThreshold : config_.heightThreshold;
            const bool zOk      = pos.z() <= zLimit;

            bool velOk = true;
            if (footPosBuf_.size() >= 2) {
                const Eigen::Vector3d displacement = pos - footPosBuf_.front().at(name);
                const double speed                 = displacement.norm() / dt;
                velOk                              = speed <= config_.velThreshold;
            }
            contacts[name] = zOk && velOk;
        }
        lastContacts_ = contacts;

        // 记录人 hip 高度
        auto hipIt = humanData.find(config_.humanRootName);
        if (hipIt != humanData.end()) {
            lastHumanHipZ_ = hipIt->second.position.z();
        }

        double maxFootZ = -std::numeric_limits<double>::infinity();
        for (const auto& [_, pos] : footPositions) {
            maxFootZ = std::max(maxFootZ, pos.z());
        }

        // 如果双脚都明显离地，逐渐衰减 groundAlignOffset_
        // 人在跳起 / 腾空时，不强行继续把脚压到地面，避免错误地拉低整个人体目标
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

        // 找到 activeZ中的z最低的脚，把它对齐到 groundZ + groundMargin 附近
        // 使用低通滤波避免跳变抖动
        if (!activeZ.empty()) {
            const double targetZ   = config_.groundZ + config_.groundMargin;
            const double rawOffset = *std::min_element(activeZ.begin(), activeZ.end()) - targetZ;
            const double alpha     = clip01(config_.lpfAlpha);
            groundAlignOffset_     = alpha * rawOffset + (1.0 - alpha) * groundAlignOffset_;
        }

        // 把整个人体 frame 沿 z 方向整体平移 offsetVec
        // 接触中的脚会被锁在一个 EMA 平滑的位置上，减少 foot sliding。
        // 脚一旦不再 contact，就从 lockedFeet_ 里删掉。
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

    double ContactGroundPipeline::fixRobotPenetration(mjModel* model, mjData* data) {
        lastRootLift_ = 0.0;
        if (!config_.enabled || !config_.fixRobotPenetration || model == nullptr || data == nullptr) {
            return 0.0;
        }

        const bool lying                = lastHumanHipZ_ <= config_.lyingHipHeightThreshold;
        const std::vector<int>& geomIds = lying ? lyingGroundGeomIds_ : groundGeomIds_;
        const double margin             = lying ? config_.lyingPenetrationMargin : config_.penetrationMargin;
        lastRootLift_ = fixRobotGroundPenetration(model, data, geomIds, floorGeomId_, margin, config_.penetrationMaxIterations);
        return lastRootLift_;
    }

}  // namespace gmr
