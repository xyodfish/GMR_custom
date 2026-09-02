#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "gmr/retarget/batch_trajectory_retarget.h"
#include "gmr/retarget/contact_ground_config.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/repo_paths.h"
#include "gmr/retarget/retargeter.h"
#include "gmr/retarget/robot_trajectory_pipeline.h"

namespace {

std::string getArg(
    int argc,
    char** argv,
    const std::string& name,
    const std::string& defaultValue = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            return argv[i + 1];
        }
    }

    return defaultValue;
}

bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) {
            return true;
        }
    }

    return false;
}

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  gmr_robot_to_robot_cli"
        << " --gmr_root <path>"
        << " --input <g1.csv|qpos.json|qpos.npy>"
        << " --robot_b <target_robot>"
        << " --out_json <target.qpos.json>"
        << " [--mapping <yaml>]"
        << " [--fps <value>]"
        << " [--max_frames <N>]"
        << " [--fast]"
        << " [--parallel]"
        << " [--verbose]"
        << " [--postprocess none|minimal]"
        << " [--no_ground_align]"
        << " [--no_contact_ground]"
        << " [--align_wrists]"
        << " [--joint_limit_margin_deg <value>]"
        << " [--dump_source_json <path>]"
        << " [--dump_human_json <path>]\n";
}

nlohmann::json qposJson(const std::vector<Eigen::VectorXd>& frames) {
    nlohmann::json out = nlohmann::json::array();
    for (const Eigen::VectorXd& q : frames) {
        out.push_back(std::vector<double>(q.data(), q.data() + q.size()));
    }

    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (hasFlag(argc, argv, "--help")) {
            printUsage();
            return 0;
        }

        const std::filesystem::path gmrRoot(getArg(argc, argv, "--gmr_root"));
        const std::filesystem::path inputPath(getArg(argc, argv, "--input"));
        const std::string robot = getArg(argc, argv, "--robot_b");
        const std::filesystem::path outputPath(getArg(argc, argv, "--out_json"));
        if (gmrRoot.empty() || inputPath.empty() || robot.empty() || outputPath.empty()) {
            printUsage();
            throw std::runtime_error("--gmr_root, --input, --robot_b, and --out_json are required.");
        }

        const std::string mappingArg = getArg(argc, argv, "--mapping");
        const std::filesystem::path mappingPath = mappingArg.empty()
            ? gmrRoot / "config/retarget/source/unitree_g1_to_smplx_proxy.yaml"
            : std::filesystem::path(mappingArg);
        const double fpsOverride = std::stod(getArg(argc, argv, "--fps", "0"));
        const int maxFrames = std::stoi(getArg(argc, argv, "--max_frames", "0"));
        const bool fast = hasFlag(argc, argv, "--fast");
        const bool parallel = hasFlag(argc, argv, "--parallel");
        const bool verbose = hasFlag(argc, argv, "--verbose");
        const bool contactGround = !hasFlag(argc, argv, "--no_contact_ground");
        const bool groundAlign = !hasFlag(argc, argv, "--no_ground_align");
        const bool alignWrists = hasFlag(argc, argv, "--align_wrists") && !hasFlag(argc, argv, "--no_align_wrists");
        const std::string postprocess = getArg(argc, argv, "--postprocess", "none");
        if (postprocess != "none" && postprocess != "minimal") {
            throw std::runtime_error("--postprocess must be 'none' or 'minimal'.");
        }

        const auto source = gmr::loadSourceRobotTrajectory(
            inputPath,
            mappingPath,
            gmrRoot,
            fpsOverride,
            maxFrames);

        const std::string dumpSource = getArg(argc, argv, "--dump_source_json");
        if (!dumpSource.empty()) {
            const std::filesystem::path sourceOutput(dumpSource);
            if (!sourceOutput.parent_path().empty()) {
                std::filesystem::create_directories(sourceOutput.parent_path());
            }

            nlohmann::json sourceJson;
            sourceJson["robot"] = "unitree_g1";
            sourceJson["fps"] = source.fps;
            sourceJson["nq"] = source.qposFrames.front().size();
            sourceJson["num_frames"] = source.qposFrames.size();
            sourceJson["qpos_frames"] = qposJson(source.qposFrames);
            std::ofstream sourceFile(sourceOutput);
            if (!sourceFile) {
                throw std::runtime_error("Failed to open source qpos JSON: " + sourceOutput.string());
            }

            sourceFile << sourceJson.dump(2) << '\n';
        }

        const auto canonical = gmr::buildCanonicalRobotTrajectory(
            source,
            mappingPath,
            gmrRoot,
            groundAlign);

        const std::string dumpHuman = getArg(argc, argv, "--dump_human_json");
        if (!dumpHuman.empty()) {
            gmr::writeCanonicalRobotTrajectory(dumpHuman, canonical);
        }

        const std::filesystem::path robotXml = gmr::resolveRobotXml(gmrRoot, robot);
        const std::filesystem::path ikPath = gmr::resolveIkConfig(gmrRoot, "smplx", robot);
        gmr::IkConfig ikConfig = gmr::loadIkConfig(ikPath, canonical.canonicalHeight);

        gmr::RetargetOptions ikOptions;
        ikOptions.damping = std::stod(getArg(argc, argv, "--damping", "0.5"));
        ikOptions.maxIterations = std::stoi(getArg(argc, argv, "--max_iter", "15"));
        ikOptions.integrationTimestep = std::stod(getArg(argc, argv, "--integration_timestep", "0"));
        ikOptions.solverName = getArg(argc, argv, "--solver", "daqp");
        ikOptions.motionFps = source.fps;

        gmr::ContactGroundCliOverrides contactOverrides;
        contactOverrides.enabled = contactGround;
        ikOptions.contactGround = gmr::buildContactGroundConfig(
            gmrRoot,
            robot,
            ikPath,
            ikConfig.humanRootName,
            contactOverrides);
        ikOptions.contactGround.velWindow = std::max(
            2,
            static_cast<int>(std::lround(ikOptions.contactGround.velWindow * source.fps / 30.0)));

        gmr::BatchTrajectoryConfig batchConfig;
        batchConfig.windowSize = std::max(3, static_cast<int>(std::lround(16.0 * source.fps / 30.0)));
        batchConfig.windowStride = std::max(1, static_cast<int>(std::lround(8.0 * source.fps / 30.0)));
        batchConfig.gnSteps = fast ? 2 : 4;
        batchConfig.gnMaxStep = 0.12;
        batchConfig.motionDt = 1.0 / source.fps;
        batchConfig.finalizeContact = false;
        batchConfig.wFootHeight = 1000.0;
        batchConfig.parallelBootstrap = parallel;
        batchConfig.parallelFinalize = parallel;
        batchConfig.verbose = verbose;
        batchConfig.jointLimitMarginDeg = std::stod(
            getArg(argc, argv, "--joint_limit_margin_deg", "0"));
        if (fast) {
            batchConfig.gnLineSearchAlphas = {1.0};
            batchConfig.useBandedSolver = true;
        }

        std::unique_ptr<gmr::Retargeter> retargeter;
        if (robot != "unitree_h2") {
            retargeter = gmr::createRetargeter(
                gmr::RetargetBackend::kMujoco,
                robotXml,
                ikConfig,
                ikOptions);
            retargeter->setMotionFps(source.fps);
        }

        const auto begin = std::chrono::steady_clock::now();
        std::vector<Eigen::VectorXd> qpos;
        gmr::BatchTrajectoryProfile profile;
        if (robot == "unitree_h2") {
            qpos = gmr::mapCompatibleRobotTrajectory(
                source.qposFrames,
                gmr::resolveRobotXml(gmrRoot, "unitree_g1"),
                robotXml);
            profile.nFrames = static_cast<int>(qpos.size());
        } else if (ikConfig.mobileUpperBody.enabled) {
            qpos.reserve(canonical.sequence.frames.size());
            for (const gmr::HumanFrame& frame : canonical.sequence.frames) {
                qpos.push_back(retargeter->retargetFrame(frame, false));
            }

            profile.nFrames = static_cast<int>(qpos.size());
        } else {
            gmr::BatchTrajectoryRetargeter batchTo(robotXml, ikConfig, batchConfig);
            gmr::BatchIkBootstrapContext bootstrap{
                gmr::RetargetBackend::kMujoco,
                ikOptions,
                ikOptions.contactGround};
            qpos = batchTo.retargetBatch(
                canonical.sequence.frames,
                *retargeter,
                false,
                &bootstrap,
                &canonical.sequence.footContacts);
            profile = batchTo.lastProfile();
        }

        const double wallMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        if (robot == "unitree_h2" || ikConfig.mobileUpperBody.enabled) {
            profile.totalMs = wallMs;
        }

        gmr::RobotPostprocessResult postprocessResult;
        const bool ranPostprocess = postprocess == "minimal" &&
            (robot == "unitree_h2" || retargeter->hasRootFreeFlyer());
        if (ranPostprocess) {
            postprocessResult = gmr::postprocessRobotTrajectory(
                qpos,
                robotXml,
                canonical.sequence.footContacts,
                source.fps,
                alignWrists);
        }

        nlohmann::json output;
        output["robot"] = robot;
        output["source_robot"] = "unitree_g1";
        output["src_human"] = "smplx";
        output["method"] = robot == "unitree_h2"
            ? "robot_to_robot_compatible_joint_map_cpp"
            : (ikConfig.mobileUpperBody.enabled
                ? "robot_to_robot_mobile_upper_body_cpp"
                : "robot_to_robot_batch_trajectory_optimization_cpp");
        output["num_frames"] = qpos.size();
        output["fps"] = source.fps;
        output["nq"] = qpos.front().size();
        output["qpos_frames"] = qposJson(qpos);
        output["canonical_quality"] = {
            {"semantic_position_rmse_m", canonical.quality.semanticPositionRmseM},
            {"semantic_position_p95_m", canonical.quality.semanticPositionP95M},
            {"semantic_rotation_mean_deg", canonical.quality.semanticRotationMeanDeg},
            {"contact_slip_mean_mps", canonical.quality.contactSlipMeanMps},
        };
        output["postprocess"] = {
            {"mode", postprocess},
            {"applied", ranPostprocess},
            {"stance_slip_mps", {
                {"before", postprocessResult.stanceSlipBeforeMps},
                {"after", postprocessResult.stanceSlipAfterMps},
            }},
        };
        output["profile"] = {
            {"prepare_ms", profile.prepareMs},
            {"bootstrap_ms", profile.bootstrapMs},
            {"optimize_ms", profile.optimizeMs},
            {"finalize_ms", profile.finalizeMs},
            {"total_ms", profile.totalMs},
            {"wall_ms", wallMs},
        };

        if (!outputPath.parent_path().empty()) {
            std::filesystem::create_directories(outputPath.parent_path());
        }

        std::ofstream file(outputPath);
        if (!file) {
            throw std::runtime_error("Failed to open output JSON: " + outputPath.string());
        }

        file << output.dump(2) << '\n';
        std::cout << "[robot-to-robot-cpp] saved: " << outputPath << '\n';
        std::cout << "[robot-to-robot-cpp] frames=" << qpos.size()
                  << " solve=" << profile.totalMs << " ms wall=" << wallMs << " ms\n";
        if (ranPostprocess) {
            std::cout << "[robot-to-robot-cpp] stance slip "
                      << postprocessResult.stanceSlipBeforeMps << " -> "
                      << postprocessResult.stanceSlipAfterMps << " m/s\n";
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[gmr_robot_to_robot_cli] Error: " << error.what() << '\n';
        return 1;
    }
}
