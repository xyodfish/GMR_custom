#include "gmr/retarget/ik_config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <iostream>

namespace gmr {

    namespace {

        Eigen::Quaterniond parseQuatWxyz(const nlohmann::json& quatJson) {
            if (!quatJson.is_array() || quatJson.size() != 4) {
                throw std::runtime_error("Quaternion must be an array with 4 values (wxyz).");
            }
            return Eigen::Quaterniond(quatJson[0].get<double>(), quatJson[1].get<double>(), quatJson[2].get<double>(),
                                      quatJson[3].get<double>());
        }

        Eigen::Vector3d parseVec3(const nlohmann::json& vecJson) {
            if (!vecJson.is_array() || vecJson.size() != 3) {
                throw std::runtime_error("Vector3 must be an array with 3 values.");
            }
            return Eigen::Vector3d(vecJson[0].get<double>(), vecJson[1].get<double>(), vecJson[2].get<double>());
        }

        Eigen::Vector2d parseVec2(const nlohmann::json& vecJson) {
            if (!vecJson.is_array() || vecJson.size() != 2) {
                throw std::runtime_error("Vector2 must be an array with 2 values.");
            }

            return Eigen::Vector2d(vecJson[0].get<double>(), vecJson[1].get<double>());
        }

        std::vector<IkTaskEntry> parseTaskTable(const nlohmann::json& table) {
            std::vector<IkTaskEntry> tasks;
            for (auto it = table.begin(); it != table.end(); ++it) {
                const auto& entry = it.value();
                if (!entry.is_array() || entry.size() != 5) {
                    throw std::runtime_error("Task entry format should be [human_body, pos_weight, rot_weight, pos_offset, rot_offset].");
                }

                IkTaskEntry task;
                task.robotBodyName = it.key();
                task.humanBodyName = entry[0].get<std::string>();
                task.posWeight     = entry[1].get<double>();
                task.rotWeight     = entry[2].get<double>();
                task.posOffset     = parseVec3(entry[3]);
                task.rotOffset     = parseQuatWxyz(entry[4]);

                if (task.posWeight != 0.0 || task.rotWeight != 0.0) {
                    tasks.push_back(std::move(task));
                }
            }
            return tasks;
        }

        std::vector<std::string> parseGeomGroup(const nlohmann::json& j) {
            if (j.is_string()) {
                return {j.get<std::string>()};
            }
            if (!j.is_array()) {
                throw std::runtime_error("collision_avoidance.self_collision_pairs: geom group must be string or array of strings.");
            }
            std::vector<std::string> names;
            names.reserve(j.size());
            for (const auto& el : j) {
                names.push_back(el.get<std::string>());
            }
            return names;
        }

        CollisionAvoidanceConfig parseCollisionAvoidance(const nlohmann::json& obj) {
            CollisionAvoidanceConfig cfg;
            cfg.enabled           = obj.value("enabled", false);
            cfg.minDistance       = obj.value("min_distance", 0.005);
            cfg.detectionDistance = obj.value("detection_distance", 0.15);
            cfg.gain              = obj.value("gain", 0.85);
            cfg.boundRelaxation   = obj.value("bound_relaxation", 0.0);
            const auto pairsIt    = obj.find("self_collision_pairs");
            if (pairsIt != obj.end() && pairsIt->is_array()) {
                for (const auto& pairJson : *pairsIt) {
                    if (!pairJson.is_array() || pairJson.size() != 2) {
                        throw std::runtime_error("collision_avoidance.self_collision_pairs: each entry must be [groupA, groupB].");
                    }
                    auto left  = parseGeomGroup(pairJson[0]);
                    auto right = parseGeomGroup(pairJson[1]);
                    if (!left.empty() && !right.empty()) {
                        cfg.selfCollisionPairs.emplace_back(std::move(left), std::move(right));
                    }
                }
            }

            if (cfg.enabled) {
                std::cout << "Collision avoidance enabled" << std::endl;
            }
            return cfg;
        }

        PlanarBaseConfig parsePlanarBase(const nlohmann::json& obj) {
            PlanarBaseConfig cfg;
            cfg.enabled = true;
            cfg.frameName = obj.at("frame_name").get<std::string>();
            cfg.humanBody = obj.at("human_body").get<std::string>();
            cfg.groundZ = obj.value("ground_z", 0.0);
            cfg.yawFrame = obj.value("yaw_frame", "");
            cfg.positionSource = obj.value("position_source", "scaled_human_root");
            return cfg;
        }

        MobileUpperBodyConfig parseMobileUpperBody(const nlohmann::json& obj) {
            MobileUpperBodyConfig cfg;
            cfg.enabled = true;
            cfg.torsoFrame = obj.at("torso_frame").get<std::string>();
            cfg.torsoHumanBody = obj.at("torso_human_body").get<std::string>();
            cfg.torsoRotationOffset = parseQuatWxyz(
                obj.value("torso_rotation_offset", nlohmann::json::array({1.0, 0.0, 0.0, 0.0})));
            cfg.torsoLocalXy = parseVec2(obj.value("torso_local_xy", nlohmann::json::array({0.107, 0.0})));
            cfg.torsoHeightScale = obj.value("torso_height_scale", 0.75);
            cfg.torsoHeightRange = parseVec2(obj.value("torso_height_range", nlohmann::json::array({0.72, 1.12})));
            cfg.torsoOrientationLimitDeg = parseVec3(
                obj.value("torso_orientation_limit_deg", nlohmann::json::array({20.0, 20.0, 30.0})));
            cfg.torsoPositionCost = obj.value("torso_position_cost", 120.0);
            cfg.torsoOrientationCost = obj.value("torso_orientation_cost", 30.0);
            cfg.torsoIterations = obj.value("torso_iterations", 20);
            cfg.torsoMinIterations = obj.value("torso_min_iterations", 4);
            cfg.initialTorsoIterations = obj.value("initial_torso_iterations", 60);
            cfg.initialTorsoMinIterations = obj.value("initial_torso_min_iterations", 30);
            cfg.headFrame = obj.value("head_frame", "");
            cfg.headHumanBody = obj.value("head_human_body", "");
            cfg.headOrientationLimitDeg = parseVec3(
                obj.value("head_orientation_limit_deg", nlohmann::json::array({30.0, 30.0, 60.0})));
            cfg.headOrientationCost = obj.value("head_orientation_cost", 2.0);
            cfg.armPositionCost = obj.value("arm_position_cost", 120.0);
            cfg.elbowOrientationCost = obj.value("elbow_orientation_cost", 1.0);
            cfg.wristOrientationCost = obj.value("wrist_orientation_cost", 2.0);
            cfg.armIterations = obj.value("arm_iterations", 15);
            cfg.armMinIterations = obj.value("arm_min_iterations", 3);
            cfg.initialArmIterations = obj.value("initial_arm_iterations", 40);
            cfg.initialArmMinIterations = obj.value("initial_arm_min_iterations", 20);
            cfg.armTargetPasses = obj.value("arm_target_passes", 2);
            cfg.jointLimitMarginDeg = obj.value("joint_limit_margin_deg", 0.0);
            cfg.postureCost = obj.value("posture_cost", 0.05);
            if (cfg.postureCost < 0.0) {
                throw std::runtime_error("mobile_upper_body.posture_cost must be non-negative.");
            }

            if (const auto postureIt = obj.find("joint_posture_cost");
                postureIt != obj.end() && postureIt->is_object()) {
                for (auto it = postureIt->begin(); it != postureIt->end(); ++it) {
                    const double cost = it.value().get<double>();
                    if (cost < 0.0) {
                        throw std::runtime_error(
                            "mobile_upper_body.joint_posture_cost must be non-negative.");
                    }

                    cfg.jointPostureCost[it.key()] = cost;
                }
            }

            const auto& arms = obj.at("arm_chains");
            if (!arms.is_array() || arms.empty()) {
                throw std::runtime_error("mobile_upper_body.arm_chains must be a non-empty array.");
            }

            cfg.armChains.reserve(arms.size());
            for (const auto& arm : arms) {
                MobileArmChainConfig chain;
                chain.shoulderFrame = arm.at("shoulder_frame").get<std::string>();
                chain.elbowFrame = arm.at("elbow_frame").get<std::string>();
                chain.wristFrame = arm.at("wrist_frame").get<std::string>();
                chain.orientationFrame = arm.value("orientation_frame", "");
                chain.shoulderHumanBody = arm.at("shoulder_human_body").get<std::string>();
                chain.elbowHumanBody = arm.at("elbow_human_body").get<std::string>();
                chain.wristHumanBody = arm.at("wrist_human_body").get<std::string>();
                chain.elbowRotationOffset = parseQuatWxyz(
                    arm.value("elbow_rotation_offset", nlohmann::json::array({1.0, 0.0, 0.0, 0.0})));
                chain.wristRotationOffset = parseQuatWxyz(
                    arm.value("wrist_rotation_offset", nlohmann::json::array({1.0, 0.0, 0.0, 0.0})));
                cfg.armChains.push_back(std::move(chain));
            }

            return cfg;
        }

    }  // namespace

    IkConfig loadIkConfig(const std::filesystem::path& configPath, double actualHumanHeight) {
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open IK config: " + configPath.string());
        }

        nlohmann::json root;
        ifs >> root;

        IkConfig config;
        config.robotRootName         = root.at("robot_root_name").get<std::string>();
        config.humanRootName         = root.at("human_root_name").get<std::string>();
        config.groundHeight          = root.at("ground_height").get<double>();
        config.humanHeightAssumption = root.at("human_height_assumption").get<double>();
        config.useTable1             = root.at("use_ik_match_table1").get<bool>();
        config.useTable2             = root.at("use_ik_match_table2").get<bool>();

        const double ratio = actualHumanHeight > 0.0 ? actualHumanHeight / config.humanHeightAssumption : 1.0;

        for (auto it = root.at("human_scale_table").begin(); it != root.at("human_scale_table").end(); ++it) {
            config.humanScaleTable[it.key()] = it.value().get<double>() * ratio;
        }

        config.tasksTable1 = parseTaskTable(root.at("ik_match_table1"));
        config.tasksTable2 = parseTaskTable(root.at("ik_match_table2"));

        const auto colIt = root.find("collision_avoidance");
        if (colIt != root.end() && colIt->is_object()) {
            config.collisionAvoidance = parseCollisionAvoidance(*colIt);
        }

        const auto planarIt = root.find("planar_base");
        if (planarIt != root.end() && planarIt->is_object()) {
            config.planarBase = parsePlanarBase(*planarIt);
        }

        const auto mobileIt = root.find("mobile_upper_body");
        if (mobileIt != root.end() && mobileIt->is_object()) {
            if (!config.planarBase.enabled) {
                throw std::runtime_error("mobile_upper_body requires planar_base configuration.");
            }

            config.mobileUpperBody = parseMobileUpperBody(*mobileIt);
        }

        return config;
    }

}  // namespace gmr
