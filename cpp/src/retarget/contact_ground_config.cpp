#include "gmr/retarget/contact_ground_config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace gmr {

    namespace {

        nlohmann::json loadJsonFile(const std::filesystem::path& path) {
            std::ifstream ifs(path);
            if (!ifs.is_open()) {
                throw std::runtime_error("Failed to open JSON: " + path.string());
            }
            nlohmann::json root;
            ifs >> root;
            return root;
        }

        nlohmann::json stripMeta(const nlohmann::json& obj) {
            nlohmann::json out = nlohmann::json::object();
            if (!obj.is_object()) {
                return out;
            }
            for (auto it = obj.begin(); it != obj.end(); ++it) {
                if (it.key().rfind("_", 0) == 0) {
                    continue;
                }
                out[it.key()] = it.value();
            }
            return out;
        }

        nlohmann::json deepMerge(const nlohmann::json& base, const nlohmann::json& overrideJson) {
            if (!overrideJson.is_object()) {
                return base;
            }
            if (!base.is_object()) {
                return overrideJson;
            }
            nlohmann::json merged = base;
            for (auto it = overrideJson.begin(); it != overrideJson.end(); ++it) {
                if (merged.contains(it.key()) && merged[it.key()].is_object() && it.value().is_object()) {
                    merged[it.key()] = deepMerge(merged[it.key()], it.value());
                } else {
                    merged[it.key()] = it.value();
                }
            }
            return merged;
        }

        nlohmann::json resolvePresetEntry(const nlohmann::json& presets, const nlohmann::json& entry) {
            if (!entry.is_object()) {
                return stripMeta(entry);
            }
            if (!entry.contains("preset")) {
                return stripMeta(entry);
            }
            const std::string baseName = entry.at("preset").get<std::string>();
            if (!presets.contains(baseName)) {
                throw std::runtime_error("contact_ground preset not found: " + baseName);
            }
            nlohmann::json base = resolvePresetEntry(presets, presets.at(baseName));
            nlohmann::json overrideJson = stripMeta(entry);
            overrideJson.erase("preset");
            return deepMerge(base, overrideJson);
        }

        std::vector<std::string> readStringArray(const nlohmann::json& j, const char* key) {
            std::vector<std::string> out;
            if (!j.contains(key) || !j.at(key).is_array()) {
                return out;
            }
            for (const auto& el : j.at(key)) {
                out.push_back(el.get<std::string>());
            }
            return out;
        }

        ContactGroundConfig parseContactGroundJson(const nlohmann::json& j) {
            ContactGroundConfig cfg;
            cfg.enabled = j.value("enabled", cfg.enabled);
            cfg.footBodies = readStringArray(j, "foot_bodies");
            cfg.humanRootName = j.value("human_root_name", cfg.humanRootName);
            cfg.velThreshold = j.value("vel_threshold", cfg.velThreshold);
            cfg.heightThreshold = j.value("height_threshold", cfg.heightThreshold);
            cfg.heightOffThreshold = j.value("height_off_threshold", cfg.heightOffThreshold);
            cfg.velWindow = j.value("vel_window", cfg.velWindow);
            cfg.groundZ = j.value("ground_z", cfg.groundZ);
            cfg.groundMargin = j.value("ground_margin", cfg.groundMargin);
            cfg.lpfAlpha = j.value("lpf_alpha", cfg.lpfAlpha);
            cfg.enableFootLock = j.value("enable_foot_lock", cfg.enableFootLock);
            cfg.footLockEmaAlpha = j.value("foot_lock_ema_alpha", cfg.footLockEmaAlpha);
            cfg.fixRobotPenetration = j.value("fix_robot_penetration", cfg.fixRobotPenetration);
            cfg.snapSupportToGround = j.value("snap_support_to_ground", cfg.snapSupportToGround);
            cfg.correctSwingFootPenetration =
                j.value("correct_swing_foot_penetration", cfg.correctSwingFootPenetration);
            cfg.footGroundLimitEnabled = j.value("foot_ground_limit_enabled", cfg.footGroundLimitEnabled);
            cfg.penetrationExcludeFeetWhenFootLimit =
                j.value("penetration_exclude_feet_when_foot_limit", cfg.penetrationExcludeFeetWhenFootLimit);
            cfg.penetrationMargin = j.value("penetration_margin", cfg.penetrationMargin);
            cfg.lyingHipHeightThreshold = j.value("lying_hip_height_threshold", cfg.lyingHipHeightThreshold);
            cfg.lowPoseFootHeightThreshold = j.value("low_pose_foot_height_threshold", cfg.lowPoseFootHeightThreshold);
            cfg.lowPoseMaxHipHeight = j.value("low_pose_max_hip_height", cfg.lowPoseMaxHipHeight);
            cfg.lyingPenetrationMargin = j.value("lying_penetration_margin", cfg.lyingPenetrationMargin);
            cfg.penetrationMaxIterations = j.value("penetration_max_iterations", cfg.penetrationMaxIterations);
            cfg.pinocchioPenetrationMode = j.value("pinocchio_penetration_mode", cfg.pinocchioPenetrationMode);
            cfg.pinocchioFootClearance = j.value("pinocchio_foot_clearance", cfg.pinocchioFootClearance);
            cfg.airborneHeightThreshold = j.value("airborne_height_threshold", cfg.airborneHeightThreshold);
            cfg.airborneOffsetDecay = j.value("airborne_offset_decay", cfg.airborneOffsetDecay);
            cfg.floorGeomName = j.value("floor_geom_name", cfg.floorGeomName);
            cfg.footCollisionGeoms = readStringArray(j, "foot_collision_geoms");
            cfg.robotFootBodies = readStringArray(j, "robot_foot_bodies");
            cfg.robotTrunkBodies = readStringArray(j, "robot_trunk_bodies");
            cfg.robotLegBodies = readStringArray(j, "robot_leg_bodies");
            cfg.robotArmBodies = readStringArray(j, "robot_arm_bodies");
            if (cfg.footBodies.empty() && cfg.robotFootBodies.size() == 2) {
                cfg.footBodies = {"left_foot", "right_foot"};
            }

            return cfg;
        }

    }  // namespace

    ContactGroundConfig robotContactGroundPreset(const std::filesystem::path& gmrRoot, const std::string& robot) {
        const auto presetsPath = gmrRoot / "general_motion_retargeting/ik_configs/contact_ground_presets.json";
        const nlohmann::json presets = loadJsonFile(presetsPath);
        nlohmann::json merged = stripMeta(presets.at("_default"));
        if (presets.contains(robot)) {
            merged = deepMerge(merged, resolvePresetEntry(presets, presets.at(robot)));
        }
        return parseContactGroundJson(merged);
    }

    ContactGroundConfig buildContactGroundConfig(const std::filesystem::path& gmrRoot, const std::string& robot,
                                                 const std::filesystem::path& ikConfigPath, const std::string& humanRootName,
                                                 const ContactGroundCliOverrides& cliOverrides) {
        const auto presetsPath = gmrRoot / "general_motion_retargeting/ik_configs/contact_ground_presets.json";
        const nlohmann::json presets = loadJsonFile(presetsPath);
        nlohmann::json merged = stripMeta(presets.at("_default"));
        if (presets.contains(robot)) {
            merged = deepMerge(merged, resolvePresetEntry(presets, presets.at(robot)));
        }

        const nlohmann::json ikRoot = loadJsonFile(ikConfigPath);
        if (ikRoot.contains("contact_ground") && ikRoot.at("contact_ground").is_object()) {
            merged = deepMerge(merged, ikRoot.at("contact_ground"));
        }

        ContactGroundConfig cfg = parseContactGroundJson(merged);
        if (!merged.contains("human_root_name")) {
            cfg.humanRootName = humanRootName;
        }
        if (ikRoot.contains("foot_ground_limit") && ikRoot.at("foot_ground_limit").is_object()) {
            cfg.footGroundLimitEnabled = ikRoot.at("foot_ground_limit").value("enabled", cfg.footGroundLimitEnabled);
        }
        if (cliOverrides.enabled.has_value()) {
            cfg.enabled = *cliOverrides.enabled;
        }
        if (cliOverrides.footGroundLimit.has_value()) {
            cfg.footGroundLimitEnabled = *cliOverrides.footGroundLimit;
        }
        if (cliOverrides.fixRobotPenetration.has_value()) {
            cfg.fixRobotPenetration = *cliOverrides.fixRobotPenetration;
        }
        return cfg;
    }

}  // namespace gmr
