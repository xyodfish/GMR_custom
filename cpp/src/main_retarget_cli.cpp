#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

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
            << "  gmr_retarget_cli"
            << " --gmr_root <path_to_GMR_root>"
            << " --robot <robot_name>"
            << " [--src_human <smplx|bvh_lafan1|bvh_nokov>]"
            << " --human_frame_json <single_or_multi_frame_json>"
            << " [--actual_human_height <float>]"
            << " [--damping <float>]"
            << " [--max_iter <int>]"
            << " [--use_velocity_limit]"
            << " [--offset_to_ground]"
            << " [--out_json <path>]"
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
    const std::string srcHuman = getArg(argc, argv, "--src_human", "smplx");
    const std::filesystem::path humanFrameJson(getArg(argc, argv, "--human_frame_json"));

    if (gmrRoot.empty() || robot.empty() || humanFrameJson.empty()) {
      throw std::runtime_error("Missing required args. Use --help.");
    }

    const double actualHumanHeight = std::stod(getArg(argc, argv, "--actual_human_height", "0"));

    gmr::RetargetOptions opts;
    opts.damping = std::stod(getArg(argc, argv, "--damping", "0.5"));
    opts.maxIterations = std::stoi(getArg(argc, argv, "--max_iter", "10"));
    opts.useVelocityLimit = hasFlag(argc, argv, "--use_velocity_limit");

    const std::filesystem::path xmlPath = gmr::resolveRobotXml(gmrRoot, robot);
    const std::filesystem::path ikPath = gmr::resolveIkConfig(gmrRoot, srcHuman, robot);
    gmr::IkConfig ikConfig = gmr::loadIkConfig(ikPath, actualHumanHeight);

    gmr::MujocoRetargeter retargeter(xmlPath, std::move(ikConfig), opts);

    const gmr::HumanFrameSequence sequence = gmr::loadHumanFrameSequence(humanFrameJson);
    const gmr::HumanFrame& frame = sequence.frames.front();
    const bool offsetToGround = hasFlag(argc, argv, "--offset_to_ground");
    const Eigen::VectorXd qpos = retargeter.retargetFrame(frame, offsetToGround);

    nlohmann::json out;
    out["robot"] = robot;
    out["nq"] = qpos.size();
    out["qpos"] = std::vector<double>(qpos.data(), qpos.data() + qpos.size());

    const std::string outJsonPath = getArg(argc, argv, "--out_json", "");
    if (outJsonPath.empty() == false) {
      std::ofstream ofs(outJsonPath);
      ofs << out.dump(2) << std::endl;
      std::cout << "Saved to: " << outJsonPath << std::endl;
    } else {
      std::cout << out.dump(2) << std::endl;
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[gmr_retarget_cli] Error: " << e.what() << std::endl;
    return 1;
  }
}
