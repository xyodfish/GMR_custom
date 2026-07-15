#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gmr/retarget/causal_trajectory_config.h"
#include "gmr/retarget/causal_trajectory_retarget.h"
#include "gmr/retarget/contact_ground_config.h"
#include "gmr/retarget/human_frame_io.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/repo_paths.h"
#include "gmr/retarget/retargeter.h"

namespace {

    std::string getArg(int argc, char** argv, const std::string& name, const std::string& defaultValue = "") {
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
        std::cout << "Usage:\n"
                  << "  gmr_causal_to_cli"
                  << " --gmr_root <path>"
                  << " --robot <robot_name>"
                  << " --human_frame_json <json>"
                  << " --out_json <path>"
                  << " [--backend mujoco_se3]"
                  << " [--solver <lbfgs|gn|none>]"
                  << " [--gn_steps 3]"
                  << " [--light_ik_iters 5]"
                  << " [--fast_opt_iter 5]"
                  << " [--max_frames N]"
                  << " [--benchmark]"
                  << "\n";
    }

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 7 || hasFlag(argc, argv, "--help")) {
            printUsage();
            return 0;
        }

        const std::filesystem::path gmrRoot(getArg(argc, argv, "--gmr_root"));
        const std::string robot       = getArg(argc, argv, "--robot");
        const std::filesystem::path humanJson(getArg(argc, argv, "--human_frame_json"));
        const std::string outJson     = getArg(argc, argv, "--out_json");
        const std::string backendName = getArg(argc, argv, "--backend", "mujoco_se3");

        if (gmrRoot.empty() || robot.empty() || humanJson.empty() || outJson.empty()) {
            throw std::runtime_error("Missing required args.");
        }

        const double actualHumanHeight = std::stod(getArg(argc, argv, "--actual_human_height", "0"));
        const int maxFramesInput       = std::stoi(getArg(argc, argv, "--max_frames", "0"));
        const bool benchmark           = hasFlag(argc, argv, "--benchmark");
        const bool offsetToGround      = hasFlag(argc, argv, "--offset_to_ground");

        gmr::RetargetOptions ikOpts;
        ikOpts.damping       = std::stod(getArg(argc, argv, "--damping", "0.5"));
        ikOpts.maxIterations = std::stoi(getArg(argc, argv, "--max_iter", "15"));

        const gmr::RetargetBackend backend = gmr::parseRetargetBackend(backendName);
        const std::filesystem::path robotXml = gmr::resolveRobotXml(gmrRoot, robot);
        const std::filesystem::path ikPath   = gmr::resolveIkConfig(gmrRoot, getArg(argc, argv, "--src_human", "smplx"), robot);
        gmr::IkConfig ikConfig               = gmr::loadIkConfig(ikPath, actualHumanHeight);

        gmr::ContactGroundCliOverrides cgCli;
        if (hasFlag(argc, argv, "--contact_ground")) {
            cgCli.enabled = true;
        }
        if (hasFlag(argc, argv, "--no_contact_ground")) {
            cgCli.enabled = false;
        }
        ikOpts.contactGround = gmr::buildContactGroundConfig(gmrRoot, robot, ikPath, ikConfig.humanRootName, cgCli);

        gmr::CausalTrajectoryConfig causalCfg;
        const std::string solverName = getArg(argc, argv, "--solver", "lbfgs");
        if (solverName == "none") {
            causalCfg.solver = gmr::CausalSolver::kNone;
        } else if (solverName == "gn") {
            causalCfg.solver = gmr::CausalSolver::kGn;
        } else if (solverName == "lbfgs") {
            causalCfg.solver = gmr::CausalSolver::kLbfgs;
        } else {
            throw std::runtime_error("Unknown --solver (use lbfgs, gn, or none).");
        }
        causalCfg.gnSteps               = std::stoi(getArg(argc, argv, "--gn_steps", "3"));
        causalCfg.lightIkWarmstartIters = std::stoi(getArg(argc, argv, "--light_ik_iters", "5"));
        causalCfg.fastOptIter             = std::stoi(getArg(argc, argv, "--fast_opt_iter", "5"));

        std::unique_ptr<gmr::Retargeter> retargeter = gmr::createRetargeter(backend, robotXml, ikConfig, ikOpts);
        gmr::CausalTrajectoryRetargeter causalTo(causalCfg);

        const gmr::HumanFrameSequence sequence = gmr::loadHumanFrameSequence(humanJson);
        if (sequence.frames.empty()) {
            throw std::runtime_error("Empty human frame sequence.");
        }
        if (sequence.fps > 0.0) {
            retargeter->setMotionFps(sequence.fps);
            causalTo.setMotionFps(sequence.fps);
        }

        std::size_t frameCount = sequence.frames.size();
        if (maxFramesInput > 0) {
            frameCount = std::min(frameCount, static_cast<std::size_t>(maxFramesInput));
        }

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Eigen::VectorXd> qFrames;
        qFrames.reserve(frameCount);
        double sumFrameMs = 0.0;
        double maxFrameMs = 0.0;

        causalTo.reset();
        for (std::size_t i = 0; i < frameCount; ++i) {
            qFrames.push_back(causalTo.retargetFrame(sequence.frames[i], *retargeter, offsetToGround));
            const double frameMs = causalTo.lastFrameMs();
            sumFrameMs += frameMs;
            maxFrameMs = std::max(maxFrameMs, frameMs);
        }
        const double wallMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double msPerFrame = frameCount > 0 ? sumFrameMs / static_cast<double>(frameCount) : 0.0;

        nlohmann::json out;
        out["robot"]      = robot;
        out["method"]     = "causal_trajectory_optimization_cpp";
        out["num_frames"] = frameCount;
        out["fps"]        = sequence.fps;
        out["nq"]         = qFrames.empty() ? 0 : qFrames.front().size();
        out["profile"]    = {{"total_ms", sumFrameMs},
                             {"wall_ms", wallMs},
                             {"ms_per_frame", msPerFrame},
                             {"max_frame_ms", maxFrameMs},
                             {"effective_fps", msPerFrame > 0.0 ? 1000.0 / msPerFrame : 0.0}};
        out["config"]     = {{"solver", solverName},
                             {"gn_steps", causalCfg.gnSteps},
                             {"light_ik_warmstart_iters", causalCfg.lightIkWarmstartIters},
                             {"fast_opt_iter", causalCfg.fastOptIter},
                             {"use_gmr_init", causalCfg.useGmrInit},
                             {"backend", backendName}};

        nlohmann::json qJson = nlohmann::json::array();
        for (const auto& q : qFrames) {
            std::vector<double> row(q.data(), q.data() + q.size());
            qJson.push_back(row);
        }
        out["qpos_frames"] = qJson;

        std::ofstream ofs(outJson);
        ofs << out.dump(2) << std::endl;

        std::cout << "Saved causal TO to: " << outJson << std::endl;
        if (benchmark || true) {
            std::cout << "[causal-to-cpp] frames=" << frameCount << " total=" << sumFrameMs << "ms"
                      << " (" << msPerFrame << " ms/f, " << (msPerFrame > 0.0 ? 1000.0 / msPerFrame : 0.0) << " FPS)"
                      << " wall=" << wallMs << "ms max_frame=" << maxFrameMs << "ms" << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gmr_causal_to_cli] Error: " << e.what() << std::endl;
        return 1;
    }
}
