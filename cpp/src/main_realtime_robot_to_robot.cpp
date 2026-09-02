#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gmr/retarget/contact_ground_config.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/online_qp_config.h"
#include "gmr/retarget/online_qp_retarget.h"
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
        << "  gmr_realtime_robot_to_robot_cli"
        << " --input <g1.csv|qpos.json|qpos.npy>"
        << " --out_json <h2.qpos.json>"
        << " [--gmr_root .]"
        << " [--robot_b unitree_h2]"
        << " [--mode causal|lookahead]"
        << " [--online_canonical]"
        << " [--stream_jsonl]"
        << " [--max_frames N]"
        << " [--dump_source_json <path>]\n";
}

nlohmann::json qposJson(const std::vector<Eigen::VectorXd>& frames) {
    nlohmann::json out = nlohmann::json::array();
    for (const Eigen::VectorXd& q : frames) {
        out.push_back(std::vector<double>(q.data(), q.data() + q.size()));
    }

    return out;
}

void writeSourceTrajectory(
    const std::filesystem::path& path,
    const gmr::SourceRobotTrajectory& source) {
    if (path.empty()) {
        return;
    }

    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    nlohmann::json output;
    output["robot"] = "unitree_g1";
    output["fps"] = source.fps;
    output["nq"] = source.qposFrames.front().size();
    output["num_frames"] = source.qposFrames.size();
    output["qpos_frames"] = qposJson(source.qposFrames);

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open source qpos JSON: " + path.string());
    }

    file << output.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (hasFlag(argc, argv, "--help")) {
            printUsage();
            return 0;
        }

        const std::filesystem::path gmrRoot(getArg(argc, argv, "--gmr_root", "."));
        const std::filesystem::path inputPath(getArg(argc, argv, "--input"));
        const std::filesystem::path outputPath(getArg(argc, argv, "--out_json"));
        const std::string robot = getArg(argc, argv, "--robot_b", "unitree_h2");
        const std::string mode = getArg(argc, argv, "--mode", "lookahead");
        const bool useOnlineCanonical = hasFlag(argc, argv, "--online_canonical");
        const bool useCompatibleJointMap = robot == "unitree_h2";
        const bool streamJsonl = hasFlag(argc, argv, "--stream_jsonl");
        if (inputPath.empty() || outputPath.empty()) {
            printUsage();
            throw std::runtime_error("--input and --out_json are required.");
        }

        if (mode != "causal" && mode != "lookahead") {
            throw std::runtime_error("--mode must be 'causal' or 'lookahead'.");
        }

        const std::string mappingArg = getArg(argc, argv, "--mapping");
        const std::filesystem::path mappingPath = mappingArg.empty()
            ? gmrRoot / "config/retarget/source/unitree_g1_to_smplx_proxy.yaml"
            : std::filesystem::path(mappingArg);
        const int maxFrames = std::stoi(getArg(argc, argv, "--max_frames", "0"));
        const gmr::SourceRobotTrajectory source = gmr::loadSourceRobotTrajectory(
            inputPath,
            mappingPath,
            gmrRoot,
            0.0,
            maxFrames);
        writeSourceTrajectory(getArg(argc, argv, "--dump_source_json"), source);

        gmr::SourceRobotFrameMapper sourceMapper(mappingPath, gmrRoot);
        std::unique_ptr<gmr::OnlineCanonicalFitter> canonicalFitter;
        if (useOnlineCanonical) {
            canonicalFitter = std::make_unique<gmr::OnlineCanonicalFitter>(mappingPath, source.fps);
        }

        const std::filesystem::path robotXml = gmr::resolveRobotXml(gmrRoot, robot);
        const std::filesystem::path ikPath = gmr::resolveIkConfig(gmrRoot, "smplx", robot);
        gmr::IkConfig ikConfig = gmr::loadIkConfig(ikPath, sourceMapper.outputHeight());

        gmr::RetargetOptions ikOptions;
        ikOptions.damping = 0.5;
        ikOptions.maxIterations = 15;
        ikOptions.solverName = "daqp";
        ikOptions.motionFps = source.fps;

        gmr::ContactGroundCliOverrides contactOverrides;
        contactOverrides.enabled = true;
        ikOptions.contactGround = gmr::buildContactGroundConfig(
            gmrRoot,
            robot,
            ikPath,
            ikConfig.humanRootName,
            contactOverrides);
        // Explicit source contacts make the support-foot height the ground reference.
        // Tracking it directly avoids a lagging whole-body Z target fighting the foot QP.
        ikOptions.contactGround.lpfAlpha = 1.0;
        ikOptions.contactGround.velWindow = std::max(
            2,
            static_cast<int>(std::lround(ikOptions.contactGround.velWindow * source.fps / 30.0)));
        ikOptions.contactGround.snapSupportToGround = false;
        ikOptions.contactGround.footGroundLimitEnabled = false;
        ikOptions.contactGround.penetrationMargin = 0.002;
        ikOptions.contactGround.lyingPenetrationMargin = 0.002;
        if (robot == "unitree_h2") {
            ikOptions.contactGround.footBodies = {"left_foot", "right_foot"};
            for (const std::string& side : {"left", "right"}) {
                for (int index = 1; index <= 7; ++index) {
                    ikOptions.contactGround.footCollisionGeoms.push_back(
                        side + "_foot" + std::to_string(index) + "_collision");
                }

            }

        }

        gmr::OnlineQpConfig qpConfig =
            gmr::OnlineQpConfig::fromPreset(gmr::OnlineQpPreset::kAntiSlip);
        qpConfig.useLookahead = mode == "lookahead";
        qpConfig.horizon = std::max(
            2,
            1 + static_cast<int>(std::lround(2.0 * source.fps / 30.0)));
        qpConfig.bootstrapGmrFrames = 1;
        qpConfig.finalizeContact = false;

        std::unique_ptr<gmr::Retargeter> retargeter = gmr::createRetargeter(
            gmr::RetargetBackend::kMujoco,
            robotXml,
            ikConfig,
            ikOptions);
        retargeter->setMotionFps(source.fps);
        const Eigen::VectorXd initialTargetQpos = retargeter->currentQpos();

        gmr::OnlineQpRetargeter onlineQp(robotXml, ikConfig, qpConfig);
        onlineQp.setMotionFps(source.fps);
        onlineQp.applyContactGroundConfig(ikOptions.contactGround);

        sourceMapper.reset();
        if (canonicalFitter) {
            canonicalFitter->reset();
        }

        onlineQp.reset();
        std::vector<Eigen::VectorXd> outputFrames;
        outputFrames.reserve(source.qposFrames.size());
        const int pipelineLatencyFrames =
            useCompatibleJointMap
                ? 0
                : (canonicalFitter ? canonicalFitter->latencyFrames() : 0) +
                    (qpConfig.useLookahead ? qpConfig.horizon - 1 : 0);
        auto appendOutput = [&](Eigen::VectorXd q) {
            outputFrames.push_back(std::move(q));
            if (streamJsonl) {
                const Eigen::VectorXd& emitted = outputFrames.back();
                nlohmann::json line;
                line["frame_index"] = outputFrames.size() - 1;
                line["pipeline_latency_frames"] = pipelineLatencyFrames;
                if (outputFrames.size() == 1) {
                    line["initial_qpos"] = std::vector<double>(
                        initialTargetQpos.data(),
                        initialTargetQpos.data() + initialTargetQpos.size());
                }

                line["qpos"] = std::vector<double>(emitted.data(), emitted.data() + emitted.size());
                std::cout << line.dump() << '\n' << std::flush;
            }
        };

        double mappingMs = 0.0;
        double canonicalMs = 0.0;
        double solveMs = 0.0;
        auto submitFrame = [&](
            const gmr::HumanFrame& frame,
            const gmr::ContactGroundState* contactState) {
            const auto solveBegin = std::chrono::steady_clock::now();
            if (qpConfig.useLookahead) {
                if (contactState == nullptr) {
                    onlineQp.pushArrivedFrame(frame);
                } else {
                    onlineQp.pushArrivedFrame(frame, *contactState);
                }

                while (onlineQp.canStepArrived(false)) {
                    appendOutput(onlineQp.stepArrived(*retargeter, false, false));
                }

            } else {
                appendOutput(contactState == nullptr
                    ? onlineQp.retargetFrame(frame, *retargeter, false)
                    : onlineQp.retargetFrame(frame, *contactState, *retargeter, false));
            }

            solveMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - solveBegin).count();
        };

        if (useCompatibleJointMap) {
            gmr::CompatibleRobotMapper mapper(
                gmr::resolveRobotXml(gmrRoot, "unitree_g1"),
                robotXml);
            for (const Eigen::VectorXd& sourceQpos : source.qposFrames) {
                const auto mapBegin = std::chrono::steady_clock::now();
                appendOutput(mapper.mapFrame(sourceQpos));
                mappingMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - mapBegin).count();
            }
        } else {
            for (const Eigen::VectorXd& sourceQpos : source.qposFrames) {
                const auto mapBegin = std::chrono::steady_clock::now();
                gmr::HumanFrame frame = sourceMapper.mapFrame(sourceQpos);
                mappingMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - mapBegin).count();

                if (canonicalFitter) {
                    const auto canonicalBegin = std::chrono::steady_clock::now();
                    canonicalFitter->pushFrame(std::move(frame), sourceMapper.footContacts());
                    canonicalMs += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - canonicalBegin).count();
                    while (canonicalFitter->canPop(false)) {
                        const auto popBegin = std::chrono::steady_clock::now();
                        gmr::OnlineCanonicalOutput output = canonicalFitter->popFrame(false);
                        canonicalMs += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - popBegin).count();
                        const gmr::ContactGroundState contactState{output.footContacts, false};
                        submitFrame(output.frame, &contactState);
                    }
                } else {
                    submitFrame(frame, nullptr);
                }
            }

            if (canonicalFitter) {
                while (canonicalFitter->canPop(true)) {
                    const auto popBegin = std::chrono::steady_clock::now();
                    gmr::OnlineCanonicalOutput output = canonicalFitter->popFrame(true);
                    canonicalMs += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - popBegin).count();
                    const gmr::ContactGroundState contactState{output.footContacts, false};
                    submitFrame(output.frame, &contactState);
                }
            }

            if (qpConfig.useLookahead) {
                const auto flushBegin = std::chrono::steady_clock::now();
                while (onlineQp.canStepArrived(true)) {
                    appendOutput(onlineQp.stepArrived(*retargeter, false, true));
                }

                solveMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - flushBegin).count();
            }
        }

        if (outputFrames.size() != source.qposFrames.size()) {
            throw std::runtime_error("Realtime retargeter must emit exactly one target frame per source frame.");
        }

        const double totalMs = mappingMs + canonicalMs + solveMs;
        const double msPerFrame = totalMs / static_cast<double>(outputFrames.size());
        const double frameBudgetMs = 1000.0 / source.fps;
        nlohmann::json output;
        output["robot"] = robot;
        output["source_robot"] = "unitree_g1";
        output["method"] = useCompatibleJointMap
            ? "realtime_compatible_joint_map_cpp"
            : (useOnlineCanonical
                ? "online_canonical_robot_to_robot_qp_cpp"
                : "direct_robot_to_robot_online_qp_cpp");
        output["mode"] = mode;
        output["fps"] = source.fps;
        output["num_frames"] = outputFrames.size();
        output["pipeline_latency_frames"] = pipelineLatencyFrames;
        output["nq"] = outputFrames.front().size();
        output["qpos_frames"] = qposJson(outputFrames);
        output["profile"] = {
            {"mapping_ms", mappingMs},
            {"canonical_ms", canonicalMs},
            {"solve_ms", solveMs},
            {"total_ms", totalMs},
            {"ms_per_frame", msPerFrame},
            {"frame_budget_ms", frameBudgetMs},
            {"realtime", msPerFrame <= frameBudgetMs},
            {"qp_fallback_count", onlineQp.qpFallbackCount()},
        };
        output["config"] = {
            {"canonical_fit", useOnlineCanonical && !useCompatibleJointMap},
            {"compatible_joint_map", useCompatibleJointMap},
            {"canonical_latency_frames", canonicalFitter ? canonicalFitter->latencyFrames() : 0},
            {"pipeline_latency_frames", pipelineLatencyFrames},
            {"trajectory_smoothing", useOnlineCanonical && !useCompatibleJointMap},
            {"horizon", qpConfig.horizon},
            {"velocity_weight", qpConfig.wVelocity},
            {"acceleration_weight", qpConfig.wAcceleration},
            {"bootstrap_frames", qpConfig.bootstrapGmrFrames},
            {"preset", "anti_slip"},
        };

        if (!outputPath.parent_path().empty()) {
            std::filesystem::create_directories(outputPath.parent_path());
        }

        std::ofstream file(outputPath);
        if (!file) {
            throw std::runtime_error("Failed to open output JSON: " + outputPath.string());
        }

        file << output.dump(2) << '\n';
        std::ostream& log = streamJsonl ? std::cerr : std::cout;
        log << "[realtime-robot-to-robot] saved: " << outputPath << '\n';
        log << "[realtime-robot-to-robot] frames=" << outputFrames.size()
            << " mapping=" << mappingMs << " ms canonical=" << canonicalMs
            << " ms solve=" << solveMs
            << " ms total=" << totalMs << " ms (" << msPerFrame << " ms/frame)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[gmr_realtime_robot_to_robot_cli] Error: " << error.what() << '\n';
        return 1;
    }
}
