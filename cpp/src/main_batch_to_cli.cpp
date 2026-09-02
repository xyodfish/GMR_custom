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
#include "gmr/retarget/robot_trajectory_pipeline.h"

namespace {

    void applyRobotToRobotBatchTuning(gmr::BatchTrajectoryConfig& batchCfg, bool fast) {
        batchCfg.finalizeContact = false;
        batchCfg.wFootHeight     = 1000.0;
        batchCfg.gnMaxStep       = 0.12;
        batchCfg.gnSteps         = fast ? 2 : 4;
    }

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

    nlohmann::json qposJson(const std::vector<Eigen::VectorXd>& frames) {
        nlohmann::json out = nlohmann::json::array();
        for (const Eigen::VectorXd& q : frames) {
            out.push_back(std::vector<double>(q.data(), q.data() + q.size()));
        }

        return out;
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
                  << " [--torque_limit_weight 0]"
                  << " [--torque_limit_margin 0.1]"
                  << " [--torque_limit_scope upper|all]"
                  << " [--torque_limit_gate_mode off|soft|hard]"
                  << " [--torque_limit_gate_r_on 0.85]"
                  << " [--torque_limit_gate_r_full 0.95]"
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
                  << " [--no_finalize_contact]"
                  << " [--no_g1_bridge]"
                  << " [--dump_g1_bridge_json <path>]"
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
        const std::filesystem::path dumpG1BridgeJson(getArg(argc, argv, "--dump_g1_bridge_json"));

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
        const bool noFinalizeContact = hasFlag(argc, argv, "--no_finalize_contact");
        if (noFinalizeContact) {
            batchCfg.finalizeContact = false;
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
        if (!getArg(argc, argv, "--torque_limit_weight").empty()) {
            batchCfg.torqueLimitConstraint = true;
            batchCfg.torqueLimitWeight     = std::stod(getArg(argc, argv, "--torque_limit_weight"));
        }
        if (!getArg(argc, argv, "--torque_limit_margin").empty()) {
            batchCfg.torqueLimitMargin = std::stod(getArg(argc, argv, "--torque_limit_margin"));
        }
        batchCfg.torqueLimitScope = getArg(argc, argv, "--torque_limit_scope", batchCfg.torqueLimitScope);
        batchCfg.torqueLimitGateMode = getArg(argc, argv, "--torque_limit_gate_mode", batchCfg.torqueLimitGateMode);
        if (!getArg(argc, argv, "--torque_limit_gate_r_on").empty()) {
            batchCfg.torqueLimitGateROn = std::stod(getArg(argc, argv, "--torque_limit_gate_r_on"));
        }
        if (!getArg(argc, argv, "--torque_limit_gate_r_full").empty()) {
            batchCfg.torqueLimitGateRFull = std::stod(getArg(argc, argv, "--torque_limit_gate_r_full"));
        }
        if (sequence.fps > 0.0) {
            batchCfg.motionDt = 1.0 / sequence.fps;
        }

        std::size_t frameCount = sequence.frames.size();
        if (maxFramesInput > 0) {
            frameCount = std::min(frameCount, static_cast<std::size_t>(maxFramesInput));
        }

        const auto t0 = std::chrono::steady_clock::now();
        bool usedG1Bridge = false;
        gmr::BatchTrajectoryProfile g1Profile;
        gmr::BatchTrackingQuality g1TrackingQuality;
        std::vector<Eigen::VectorXd> g1BridgeQpos;
        std::vector<Eigen::VectorXd> compatibleBridgeQpos;
        std::vector<gmr::HumanFrame> frames(sequence.frames.begin(), sequence.frames.begin() + frameCount);
        gmr::BatchTrajectoryRetargeter::FootContactSchedule footContacts;
        if (!sequence.footContacts.empty()) {
            footContacts.assign(sequence.footContacts.begin(), sequence.footContacts.begin() + frameCount);
        }

        const bool h2SmplxG1Bridge = robot == "unitree_h2" && srcHuman == "smplx" &&
            !hasFlag(argc, argv, "--no_g1_bridge");
        if (h2SmplxG1Bridge) {
            const std::filesystem::path g1Xml = gmr::resolveRobotXml(gmrRoot, "unitree_g1");
            const std::filesystem::path g1IkPath =
                gmr::resolveIkConfig(gmrRoot, srcHuman, "unitree_g1");
            const gmr::IkConfig g1IkConfig = gmr::loadIkConfig(g1IkPath, actualHumanHeight);

            gmr::RetargetOptions g1IkOpts = ikOpts;
            g1IkOpts.contactGround = gmr::buildContactGroundConfig(
                gmrRoot,
                "unitree_g1",
                g1IkPath,
                g1IkConfig.humanRootName,
                cgCli);
            std::unique_ptr<gmr::Retargeter> g1Retargeter =
                gmr::createRetargeter(backend, g1Xml, g1IkConfig, g1IkOpts);
            if (sequence.fps > 0.0) {
                g1Retargeter->setMotionFps(sequence.fps);
            }

            gmr::BatchTrajectoryConfig g1BatchCfg = batchCfg;
            g1BatchCfg.qInitJsonPath.clear();
            gmr::BatchTrajectoryRetargeter g1Batch(g1Xml, g1IkConfig, g1BatchCfg);
            gmr::BatchIkBootstrapContext g1Bootstrap{
                backend,
                g1IkOpts,
                g1IkOpts.contactGround};
            const auto* sourceContacts = footContacts.empty() ? nullptr : &footContacts;
            g1BridgeQpos = g1Batch.retargetBatch(
                frames,
                *g1Retargeter,
                offsetToGround,
                &g1Bootstrap,
                sourceContacts);
            g1Profile = g1Batch.lastProfile();
            g1TrackingQuality = g1Batch.lastTrackingQuality();
            compatibleBridgeQpos = gmr::mapCompatibleRobotTrajectory(
                g1BridgeQpos,
                g1Xml,
                robotXml);
            usedG1Bridge = true;
        } else if (robot == "unitree_h2") {
            applyRobotToRobotBatchTuning(batchCfg, fast);
        }

        if (!dumpG1BridgeJson.empty() && !usedG1Bridge) {
            throw std::runtime_error("--dump_g1_bridge_json requires SMPL-X input and robot=unitree_h2.");
        }

        std::vector<Eigen::VectorXd> qBatch;
        gmr::BatchTrajectoryProfile prof;
        double lastTorqueGate = 0.0;
        double meanTorqueGate = 0.0;
        double lastTorquePeakRatio = 0.0;
        double maxTorquePeakRatio = 0.0;
        if (usedG1Bridge) {
            qBatch = std::move(compatibleBridgeQpos);
            prof.nFrames = static_cast<int>(qBatch.size());
        } else {
            ikOpts.contactGround =
                gmr::buildContactGroundConfig(gmrRoot, robot, ikPath, ikConfig.humanRootName, cgCli);
            std::unique_ptr<gmr::Retargeter> ikRetargeter =
                gmr::createRetargeter(backend, robotXml, ikConfig, ikOpts);
            gmr::BatchTrajectoryRetargeter batchTo(robotXml, ikConfig, batchCfg);
            if (sequence.fps > 0) {
                ikRetargeter->setMotionFps(sequence.fps);
                ikOpts.motionFps = sequence.fps;
            }

            gmr::BatchIkBootstrapContext ikBootstrap{backend, ikOpts, ikOpts.contactGround};
            const auto* footContactSchedule = footContacts.empty() ? nullptr : &footContacts;
            qBatch = batchTo.retargetBatch(
                frames,
                *ikRetargeter,
                offsetToGround,
                &ikBootstrap,
                footContactSchedule);
            prof = batchTo.lastProfile();
            lastTorqueGate = batchTo.lastTorqueGate();
            meanTorqueGate = batchTo.meanTorqueGate();
            lastTorquePeakRatio = batchTo.lastTorquePeakRatio();
            maxTorquePeakRatio = batchTo.maxTorquePeakRatio();
        }

        const double wallMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (usedG1Bridge) {
            prof.totalMs = wallMs;
        }

        nlohmann::json out;
        out["robot"]      = robot;
        out["src_human"]  = srcHuman;
        out["method"]     = usedG1Bridge
            ? "batch_trajectory_optimization_gn_cpp_smplx_via_g1_joint_map"
            : "batch_trajectory_optimization_gn_cpp";
        out["g1_bridge"] = usedG1Bridge;
        out["num_frames"] = frameCount;
        out["fps"]        = sequence.fps;
        out["nq"]         = qBatch.front().size();
        out["profile"]    = {{"prepare_ms", prof.prepareMs},
                             {"bootstrap_ms", prof.bootstrapMs},
                             {"optimize_ms", prof.optimizeMs},
                             {"finalize_ms", prof.finalizeMs},
                             {"total_ms", prof.totalMs},
                             {"ms_per_frame", prof.msPerFrame()},
                             {"effective_fps", prof.effectiveFps()},
                             {"wall_ms", wallMs}};
        if (usedG1Bridge) {
            out["g1_bridge_profile"] = {
                {"prepare_ms", g1Profile.prepareMs},
                {"bootstrap_ms", g1Profile.bootstrapMs},
                {"optimize_ms", g1Profile.optimizeMs},
                {"finalize_ms", g1Profile.finalizeMs},
                {"total_ms", g1Profile.totalMs}};
            out["g1_tracking_quality"] = {
                {"position_mean_m", g1TrackingQuality.positionMeanM},
                {"position_p95_m", g1TrackingQuality.positionP95M},
                {"position_max_m", g1TrackingQuality.positionMaxM},
                {"rotation_mean_deg", g1TrackingQuality.rotationMeanDeg},
                {"rotation_p95_deg", g1TrackingQuality.rotationP95Deg},
                {"rotation_max_deg", g1TrackingQuality.rotationMaxDeg},
                {"position_samples", g1TrackingQuality.positionSamples},
                {"rotation_samples", g1TrackingQuality.rotationSamples},
                {"worst_position_frame", g1TrackingQuality.worstPositionFrame},
                {"worst_rotation_frame", g1TrackingQuality.worstRotationFrame},
                {"worst_position_body", g1TrackingQuality.worstPositionBody},
                {"worst_rotation_body", g1TrackingQuality.worstRotationBody}};
        }

        if (!dumpG1BridgeJson.empty()) {
            nlohmann::json g1Out;
            g1Out["robot"] = "unitree_g1";
            g1Out["src_human"] = srcHuman;
            g1Out["method"] = "batch_trajectory_optimization_gn_cpp_h2_bridge_source";
            g1Out["num_frames"] = g1BridgeQpos.size();
            g1Out["fps"] = sequence.fps;
            g1Out["nq"] = g1BridgeQpos.front().size();
            g1Out["tracking_quality"] = out["g1_tracking_quality"];
            g1Out["qpos_frames"] = qposJson(g1BridgeQpos);

            if (!dumpG1BridgeJson.parent_path().empty()) {
                std::filesystem::create_directories(dumpG1BridgeJson.parent_path());
            }

            std::ofstream g1File(dumpG1BridgeJson);
            if (!g1File) {
                throw std::runtime_error("Failed to open G1 bridge JSON: " + dumpG1BridgeJson.string());
            }

            g1File << g1Out.dump(2) << '\n';
        }

        out["torque_gate"] = {{"last_gate", lastTorqueGate},
                              {"mean_gate", meanTorqueGate},
                              {"last_peak_ratio", lastTorquePeakRatio},
                              {"max_peak_ratio", maxTorquePeakRatio}};
        out["config"] = {{"window_size", batchCfg.windowSize},
                         {"window_stride", batchCfg.windowStride},
                         {"gn_steps", batchCfg.gnSteps},
                         {"joint_limit_margin_deg", batchCfg.jointLimitMarginDeg},
                         {"fast", fast},
                         {"ceiling", ceiling},
                         {"enable_foot_penalties", batchCfg.enableFootPenalties}};

        out["qpos_frames"] = qposJson(qBatch);

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
