#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "gmr/retarget/batch_trajectory_retarget.h"
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
                  << "  gmr_batch_to_cli"
                  << " --gmr_root <path>"
                  << " --robot <robot_name>"
                  << " --human_frame_json <json>"
                  << " --out_json <path>"
                  << " [--src_human smplx|bvh_lafan1|bvh_nokov]"
                  << " [--actual_human_height <m>]"
                  << " [--backend mujoco_se3]"
                  << " [--window_size 16]"
                  << " [--window_stride 8]"
                  << " [--gn_steps 3]"
                  << " [--joint_limit_margin_deg 0]"
                  << " [--fast]"
                  << " [--ceiling]"
                  << " [--no_foot_penalties]"
                  << " [--q_init_json <path>]"
                  << " [--no_parallel]"
                  << " [--parallel]"
                  << " [--gn_line_search best|armijo]"
                  << " [--banded_solver]"
                  << " [--no_banded_solver]"
                  << " [--max_iter <int>]"
                  << " [--integration_timestep <float>]"
                  << " [--solver daqp|qpoases]"
                  << " [--max_frames N]"
                  << " [--benchmark]"
                  << "\n"
                  << "  Or use scripts/tools/run_cpp_batch_to.py --input_file <.pt|.npz|.bvh> ...\n";
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

        const double actualHumanHeightCli = std::stod(getArg(argc, argv, "--actual_human_height", "0"));
        const int maxFramesInput       = std::stoi(getArg(argc, argv, "--max_frames", "0"));
        const bool benchmark           = hasFlag(argc, argv, "--benchmark");
        const bool fast                = hasFlag(argc, argv, "--fast");
        const bool ceiling             = hasFlag(argc, argv, "--ceiling");
        const bool noFootPenalties     = hasFlag(argc, argv, "--no_foot_penalties");
        const bool noParallel          = hasFlag(argc, argv, "--no_parallel");
        const bool enableParallel      = hasFlag(argc, argv, "--parallel");
        const bool offsetToGround      = hasFlag(argc, argv, "--offset_to_ground");

        const gmr::HumanFrameSequence sequence = gmr::loadHumanFrameSequence(humanJson);
        if (sequence.frames.empty()) {
            throw std::runtime_error("Empty human frame sequence.");
        }

        std::string srcHuman = getArg(argc, argv, "--src_human", "");
        if (srcHuman.empty()) {
            srcHuman = sequence.srcHuman.empty() ? "smplx" : sequence.srcHuman;
        }
        double actualHumanHeight = actualHumanHeightCli;
        if (actualHumanHeight <= 0.0 && sequence.actualHumanHeight > 0.0) {
            actualHumanHeight = sequence.actualHumanHeight;
        }

        gmr::RetargetOptions ikOpts;
        ikOpts.damping       = std::stod(getArg(argc, argv, "--damping", "0.5"));
        ikOpts.maxIterations = std::stoi(getArg(argc, argv, "--max_iter", "15"));
        ikOpts.integrationTimestep = std::stod(getArg(argc, argv, "--integration_timestep", "0"));
        ikOpts.solverName    = getArg(argc, argv, "--solver", "daqp");

        const gmr::RetargetBackend backend = gmr::parseRetargetBackend(backendName);
        const std::filesystem::path robotXml = gmr::resolveRobotXml(gmrRoot, robot);
        const std::filesystem::path ikPath   = gmr::resolveIkConfig(gmrRoot, srcHuman, robot);
        gmr::IkConfig ikConfig               = gmr::loadIkConfig(ikPath, actualHumanHeight);

        gmr::ContactGroundCliOverrides cgCli;
        if (hasFlag(argc, argv, "--contact_ground")) {
            cgCli.enabled = true;
        }
        if (hasFlag(argc, argv, "--no_contact_ground")) {
            cgCli.enabled = false;
        }
        ikOpts.contactGround = gmr::buildContactGroundConfig(gmrRoot, robot, ikPath, ikConfig.humanRootName, cgCli);

        gmr::BatchTrajectoryConfig batchCfg;
        batchCfg.windowSize   = std::stoi(getArg(argc, argv, "--window_size", fast ? "16" : "16"));
        batchCfg.windowStride = std::stoi(getArg(argc, argv, "--window_stride", fast ? "8" : "8"));
        batchCfg.gnSteps      = std::stoi(getArg(argc, argv, "--gn_steps", fast ? "2" : "3"));
        if (fast) {
            batchCfg.gnLineSearchAlphas = {1.0};
            batchCfg.useBandedSolver    = true;
        }
        if (ceiling) {
            batchCfg.windowSize           = 16;
            batchCfg.windowStride         = 16;
            batchCfg.gnSteps              = 1;
            batchCfg.gnLineSearchAlphas   = {1.0};
            batchCfg.enableFootPenalties  = false;
            batchCfg.finalizeContact      = false;
        }
        if (noFootPenalties) {
            batchCfg.enableFootPenalties = false;
        }
        if (noParallel) {
            batchCfg.parallelBootstrap = false;
            batchCfg.parallelFinalize  = false;
        }
        if (enableParallel) {
            batchCfg.parallelBootstrap = true;
            batchCfg.parallelFinalize  = true;
        }
        const std::string lineSearchMode = getArg(argc, argv, "--gn_line_search", "best");
        if (lineSearchMode == "best") {
            batchCfg.gnLineSearchMode = gmr::GnLineSearchMode::kBest;
        } else if (lineSearchMode == "armijo") {
            batchCfg.gnLineSearchMode = gmr::GnLineSearchMode::kArmijo;
        }
        if (hasFlag(argc, argv, "--banded_solver")) {
            batchCfg.useBandedSolver = true;
        }
        if (hasFlag(argc, argv, "--no_banded_solver")) {
            batchCfg.useBandedSolver = false;
        }
        const std::string qInitJson = getArg(argc, argv, "--q_init_json", "");
        if (!qInitJson.empty()) {
            batchCfg.qInitJsonPath = qInitJson;
        }
        batchCfg.verbose = hasFlag(argc, argv, "--verbose");
        if (!getArg(argc, argv, "--joint_limit_margin_deg").empty()) {
            batchCfg.jointLimitMarginDeg = std::stod(getArg(argc, argv, "--joint_limit_margin_deg"));
        }

        std::unique_ptr<gmr::Retargeter> ikRetargeter =
            gmr::createRetargeter(backend, robotXml, ikConfig, ikOpts);

        gmr::BatchTrajectoryRetargeter batchTo(robotXml, ikConfig, batchCfg);

        if (sequence.fps > 0) {
            ikRetargeter->setMotionFps(sequence.fps);
            ikOpts.motionFps = sequence.fps;
        }

        gmr::BatchIkBootstrapContext ikBootstrap{backend, ikOpts, ikOpts.contactGround};

        std::size_t frameCount = sequence.frames.size();
        if (maxFramesInput > 0) {
            frameCount = std::min(frameCount, static_cast<std::size_t>(maxFramesInput));
        }
        std::vector<gmr::HumanFrame> frames(sequence.frames.begin(), sequence.frames.begin() + frameCount);

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Eigen::VectorXd> qBatch = batchTo.retargetBatch(frames, *ikRetargeter, offsetToGround, &ikBootstrap);
        const double wallMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const gmr::BatchTrajectoryProfile prof = batchTo.lastProfile();

        nlohmann::json out;
        out["robot"]      = robot;
        out["src_human"]  = srcHuman;
        out["method"]     = "batch_trajectory_optimization_gn_cpp";
        out["num_frames"] = frameCount;
        out["fps"]        = sequence.fps;
        out["nq"]         = batchTo.modelNq();
        out["profile"]    = {{"prepare_ms", prof.prepareMs},
                             {"bootstrap_ms", prof.bootstrapMs},
                             {"optimize_ms", prof.optimizeMs},
                             {"finalize_ms", prof.finalizeMs},
                             {"total_ms", prof.totalMs},
                             {"ms_per_frame", prof.msPerFrame()},
                             {"effective_fps", prof.effectiveFps()},
                             {"wall_ms", wallMs}};
        out["config"] = {{"window_size", batchCfg.windowSize},
                         {"window_stride", batchCfg.windowStride},
                         {"gn_steps", batchCfg.gnSteps},
                         {"joint_limit_margin_deg", batchCfg.jointLimitMarginDeg},
                         {"fast", fast},
                         {"ceiling", ceiling},
                         {"enable_foot_penalties", batchCfg.enableFootPenalties}};

        nlohmann::json qFrames = nlohmann::json::array();
        for (const auto& q : qBatch) {
            std::vector<double> row(q.data(), q.data() + q.size());
            qFrames.push_back(row);
        }
        out["qpos_frames"] = qFrames;

        std::ofstream ofs(outJson);
        ofs << out.dump(2) << std::endl;

        std::cout << "Saved batch TO to: " << outJson << std::endl;
        if (benchmark || true) {
            std::cout << "[batch-to-cpp] frames=" << frameCount << " optimize=" << prof.optimizeMs << "ms"
                      << " total=" << prof.totalMs << "ms (" << prof.msPerFrame() << " ms/f, " << prof.effectiveFps()
                      << " FPS)" << std::endl;
            std::cout << "  bootstrap=" << prof.bootstrapMs << "ms prepare=" << prof.prepareMs
                      << "ms finalize=" << prof.finalizeMs << "ms wall=" << wallMs << "ms" << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gmr_batch_to_cli] Error: " << e.what() << std::endl;
        return 1;
    }
}
