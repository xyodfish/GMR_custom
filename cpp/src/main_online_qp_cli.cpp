#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gmr/retarget/contact_ground_config.h"
#include "gmr/retarget/human_frame_io.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/online_qp_config.h"
#include "gmr/retarget/online_qp_retarget.h"
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
                  << "  gmr_online_qp_cli"
                  << " --gmr_root <path>"
                  << " --robot <robot_name>"
                  << " --human_frame_json <json>"
                  << " --out_json <path>"
                  << " [--preset default|smooth|anti_slip]"
                  << " [--mode lookahead|causal]"
                  << " [--horizon 3]"
                  << " [--sqp_iters 3]"
                  << " [--w_foot_slip 2000]"
                  << " [--w_gmr 0.4]"
                  << " [--dq_max 4]"
                  << " [--joint_limit_margin_deg 0]"
                  << " [--torque_limit_weight 0]"
                  << " [--torque_limit_margin 0.1]"
                  << " [--torque_limit_scope upper|all]"
                  << " [--torque_limit_gate_mode off|soft|hard]"
                  << " [--torque_limit_gate_r_on 0.85]"
                  << " [--torque_limit_gate_r_full 0.95]"
                  << " [--src_human smplx|bvh_lafan1]"
                  << " [--actual_human_height <m>]"
                  << " [--contact_ground]"
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
        const std::string robot = getArg(argc, argv, "--robot");
        const std::filesystem::path humanJson(getArg(argc, argv, "--human_frame_json"));
        const std::string outJson = getArg(argc, argv, "--out_json");
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
        ikOpts.solverName    = getArg(argc, argv, "--ik_solver", "daqp");

        const gmr::RetargetBackend backend = gmr::parseRetargetBackend(backendName);
        const std::filesystem::path robotXml = gmr::resolveRobotXml(gmrRoot, robot);
        const std::filesystem::path ikPath =
            gmr::resolveIkConfig(gmrRoot, getArg(argc, argv, "--src_human", "smplx"), robot);
        gmr::IkConfig ikConfig = gmr::loadIkConfig(ikPath, actualHumanHeight);

        gmr::ContactGroundCliOverrides cgCli;
        if (hasFlag(argc, argv, "--contact_ground")) {
            cgCli.enabled = true;
        }
        if (hasFlag(argc, argv, "--no_contact_ground")) {
            cgCli.enabled = false;
        }
        ikOpts.contactGround =
            gmr::buildContactGroundConfig(gmrRoot, robot, ikPath, ikConfig.humanRootName, cgCli);

        gmr::OnlineQpConfig qpCfg =
            gmr::OnlineQpConfig::fromPresetName(getArg(argc, argv, "--preset", "anti_slip"));
        const std::string mode = getArg(argc, argv, "--mode", "lookahead");
        qpCfg.useLookahead     = (mode != "causal");
        if (!getArg(argc, argv, "--horizon").empty()) {
            qpCfg.horizon = std::stoi(getArg(argc, argv, "--horizon"));
        }
        if (!getArg(argc, argv, "--sqp_iters").empty()) {
            qpCfg.sqpIters = std::stoi(getArg(argc, argv, "--sqp_iters"));
        }
        if (!getArg(argc, argv, "--w_foot_slip").empty()) {
            qpCfg.wFootSlip = std::stod(getArg(argc, argv, "--w_foot_slip"));
        }
        if (!getArg(argc, argv, "--w_gmr").empty()) {
            qpCfg.wGmr = std::stod(getArg(argc, argv, "--w_gmr"));
        }
        if (!getArg(argc, argv, "--dq_max").empty()) {
            qpCfg.dqMax = std::stod(getArg(argc, argv, "--dq_max"));
        }
        if (!getArg(argc, argv, "--light_ik_iters").empty()) {
            qpCfg.lightIkIters = std::stoi(getArg(argc, argv, "--light_ik_iters"));
        }
        if (!getArg(argc, argv, "--joint_limit_margin_deg").empty()) {
            qpCfg.jointLimitMarginDeg = std::stod(getArg(argc, argv, "--joint_limit_margin_deg"));
        }
        if (!getArg(argc, argv, "--torque_limit_weight").empty()) {
            qpCfg.torqueLimitConstraint = true;
            qpCfg.torqueLimitWeight     = std::stod(getArg(argc, argv, "--torque_limit_weight"));
        }
        if (!getArg(argc, argv, "--torque_limit_margin").empty()) {
            qpCfg.torqueLimitMargin = std::stod(getArg(argc, argv, "--torque_limit_margin"));
        }
        qpCfg.torqueLimitScope = getArg(argc, argv, "--torque_limit_scope", qpCfg.torqueLimitScope);
        qpCfg.torqueLimitGateMode = getArg(argc, argv, "--torque_limit_gate_mode", qpCfg.torqueLimitGateMode);
        if (!getArg(argc, argv, "--torque_limit_gate_r_on").empty()) {
            qpCfg.torqueLimitGateROn = std::stod(getArg(argc, argv, "--torque_limit_gate_r_on"));
        }
        if (!getArg(argc, argv, "--torque_limit_gate_r_full").empty()) {
            qpCfg.torqueLimitGateRFull = std::stod(getArg(argc, argv, "--torque_limit_gate_r_full"));
        }
        qpCfg.qpBackend = getArg(argc, argv, "--qp_solver", "daqp");

        std::unique_ptr<gmr::Retargeter> retargeter =
            gmr::createRetargeter(backend, robotXml, ikConfig, ikOpts);
        gmr::OnlineQpRetargeter onlineQp(robotXml, ikConfig, qpCfg);
        onlineQp.applyContactGroundConfig(ikOpts.contactGround);

        const gmr::HumanFrameSequence sequence = gmr::loadHumanFrameSequence(humanJson);
        if (sequence.frames.empty()) {
            throw std::runtime_error("Empty human frame sequence.");
        }
        if (sequence.fps > 0.0) {
            retargeter->setMotionFps(sequence.fps);
            onlineQp.setMotionFps(sequence.fps);
            ikOpts.motionFps = sequence.fps;
        }

        std::size_t frameCount = sequence.frames.size();
        if (maxFramesInput > 0) {
            frameCount = std::min(frameCount, static_cast<std::size_t>(maxFramesInput));
        }
        std::vector<gmr::HumanFrame> frames(sequence.frames.begin(),
                                            sequence.frames.begin() + static_cast<std::ptrdiff_t>(frameCount));

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Eigen::VectorXd> qFrames;
        qFrames.reserve(frameCount);
        onlineQp.reset();
        for (const auto& frame : frames) {
            onlineQp.pushArrivedFrame(frame);
            while (onlineQp.canStepArrived(/*flush=*/false)) {
                qFrames.push_back(onlineQp.stepArrived(*retargeter, offsetToGround, /*flush=*/false));
            }
        }
        while (onlineQp.canStepArrived(/*flush=*/true)) {
            qFrames.push_back(onlineQp.stepArrived(*retargeter, offsetToGround, /*flush=*/true));
        }
        const double wallMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double msPerFrame = qFrames.empty() ? 0.0 : wallMs / static_cast<double>(qFrames.size());

        nlohmann::json out;
        out["robot"]  = robot;
        out["method"] = "online_qp_cpp";
        out["preset"] = getArg(argc, argv, "--preset", "anti_slip");
        out["mode"]   = mode;
        out["fps"]    = sequence.fps;
        out["profile"] = {
            {"total_ms", wallMs},
            {"ms_per_frame", msPerFrame},
            {"n_frames", static_cast<int>(qFrames.size())},
            {"last_frame_ms", onlineQp.lastFrameMs()},
            {"qp_fallback_count", onlineQp.qpFallbackCount()},
            {"last_qp_error", onlineQp.lastQpError()},
        };
        out["torque_gate"] = {
            {"last_gate", onlineQp.lastTorqueGate()},
            {"mean_gate", onlineQp.meanTorqueGate()},
            {"last_peak_ratio", onlineQp.lastTorquePeakRatio()},
            {"max_peak_ratio", onlineQp.maxTorquePeakRatio()},
        };
        out["config"] = {
            {"horizon", qpCfg.horizon},
            {"sqp_iters", qpCfg.sqpIters},
            {"w_foot_slip", qpCfg.wFootSlip},
            {"w_gmr", qpCfg.wGmr},
            {"dq_max", qpCfg.dqMax},
            {"joint_limit_margin_deg", qpCfg.jointLimitMarginDeg},
            {"torque_limit_constraint", qpCfg.torqueLimitConstraint},
            {"torque_limit_weight", qpCfg.torqueLimitWeight},
            {"torque_limit_margin", qpCfg.torqueLimitMargin},
            {"torque_limit_scope", qpCfg.torqueLimitScope},
            {"torque_limit_gate_mode", qpCfg.torqueLimitGateMode},
            {"torque_limit_gate_r_on", qpCfg.torqueLimitGateROn},
            {"torque_limit_gate_r_full", qpCfg.torqueLimitGateRFull},
            {"finalize_contact", qpCfg.finalizeContact},
            {"qp_backend", qpCfg.qpBackend},
        };

        nlohmann::json qposFrames = nlohmann::json::array();
        for (const auto& q : qFrames) {
            nlohmann::json row = nlohmann::json::array();
            for (int i = 0; i < q.size(); ++i) {
                row.push_back(q[i]);
            }
            qposFrames.push_back(row);
        }
        out["qpos_frames"] = qposFrames;

        std::ofstream ofs(outJson);
        if (!ofs) {
            throw std::runtime_error("Failed to write " + outJson);
        }
        ofs << out.dump(2);

        if (benchmark) {
            std::cout << "[online-qp-cpp] frames=" << frameCount << " ms/f=" << msPerFrame
                      << " total_ms=" << wallMs << " realtime@30=" << (msPerFrame <= 1000.0 / 30.0) << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gmr_online_qp_cli error: " << e.what() << "\n";
        return 1;
    }
}
