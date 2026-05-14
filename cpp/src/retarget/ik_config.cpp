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
            std::cout << "Collision avoidance enabled" << std::endl;
        }

        return config;
    }

}  // namespace gmr
