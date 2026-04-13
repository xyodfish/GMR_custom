#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include "gmr/retarget/human_frame_io.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/repo_paths.h"
#include "gmr/retarget/retargeter.h"

namespace {

std::string getArg(int argc, char **argv, const std::string &name,
                   const std::string &defaultValue = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) {
      return argv[i + 1];
    }
  }
  return defaultValue;
}

bool hasFlag(int argc, char **argv, const std::string &flag) {
  for (int i = 1; i < argc; ++i) {
    if (flag == argv[i]) {
      return true;
    }
  }
  return false;
}

void printUsage() {
  std::cout << "Usage:\n"
            << "  gmr_retarget_viewer"
            << " --gmr_root <path_to_GMR_root>"
            << " --robot <robot_name>"
            << " [--src_human <smplx|bvh_lafan1|bvh_nokov>]"
            << " --human_frame_json <single_or_multi_frame_json>"
            << " [--actual_human_height <float>]"
            << " [--damping <float>]"
            << " [--max_iter <int>]"
            << " [--use_velocity_limit]"
            << " [--offset_to_ground]"
            << " [--loop]"
            << " [--realtime]"
            << " [--window_width <int>]"
            << " [--window_height <int>]"
            << "\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 7 || hasFlag(argc, argv, "--help")) {
      printUsage();
      return 0;
    }

    const std::filesystem::path gmrRoot(getArg(argc, argv, "--gmr_root"));
    const std::string robot = getArg(argc, argv, "--robot");
    const std::string srcHuman = getArg(argc, argv, "--src_human", "smplx");
    const std::filesystem::path humanFrameJson(
        getArg(argc, argv, "--human_frame_json"));

    if (gmrRoot.empty() || robot.empty() || humanFrameJson.empty()) {
      throw std::runtime_error("Missing required args. Use --help.");
    }

    const double actualHumanHeight =
        std::stod(getArg(argc, argv, "--actual_human_height", "0"));

    gmr::RetargetOptions opts;
    opts.damping = std::stod(getArg(argc, argv, "--damping", "0.5"));
    opts.maxIterations = std::stoi(getArg(argc, argv, "--max_iter", "10"));
    opts.useVelocityLimit = hasFlag(argc, argv, "--use_velocity_limit");

    const std::filesystem::path xmlPath = gmr::resolveRobotXml(gmrRoot, robot);
    const std::filesystem::path ikPath =
        gmr::resolveIkConfig(gmrRoot, srcHuman, robot);
    gmr::IkConfig ikConfig = gmr::loadIkConfig(ikPath, actualHumanHeight);

    const gmr::HumanFrameSequence sequence =
        gmr::loadHumanFrameSequence(humanFrameJson);
    if (sequence.frames.empty()) {
      throw std::runtime_error("No frames to render.");
    }

    gmr::MujocoRetargeter retargeter(xmlPath, ikConfig, opts);

    if (glfwInit() == GLFW_FALSE) {
      throw std::runtime_error("glfwInit failed.");
    }

    const int windowWidth =
        std::stoi(getArg(argc, argv, "--window_width", "1280"));
    const int windowHeight =
        std::stoi(getArg(argc, argv, "--window_height", "720"));

    GLFWwindow *window = glfwCreateWindow(
        windowWidth, windowHeight, "GMR C++ Retarget Viewer", nullptr, nullptr);
    if (window == nullptr) {
      glfwTerminate();
      throw std::runtime_error(
          "Failed to create GLFW window. Check DISPLAY / OpenGL context.");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    const mjModel *model = retargeter.model();
    const mjData *data = retargeter.data();

    mjv_makeScene(model, &scn, 5000);
    mjr_makeContext(model, &con, mjFONTSCALE_150);

    const int rootBodyId =
        mj_name2id(model, mjOBJ_BODY, ikConfig.robotRootName.c_str());
    cam.distance = 2.5;
    cam.azimuth = 120.0;
    cam.elevation = -15.0;
    if (rootBodyId >= 0) {
      cam.lookat[0] = data->xpos[3 * rootBodyId + 0];
      cam.lookat[1] = data->xpos[3 * rootBodyId + 1];
      cam.lookat[2] = data->xpos[3 * rootBodyId + 2];
    }

    const bool loop = hasFlag(argc, argv, "--loop");
    const bool offsetToGround = hasFlag(argc, argv, "--offset_to_ground");
    const bool realtime = hasFlag(argc, argv, "--realtime");
    const double frameDt = 1.0 / static_cast<double>(sequence.fps);
    const auto frameDtDuration =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(frameDt));

    std::vector<Eigen::VectorXd> precomputedQpos;
    if (!realtime) {
      precomputedQpos.reserve(sequence.frames.size());
      std::cout << "Precomputing retarget trajectory ("
                << sequence.frames.size() << " frames)..." << std::endl;
      for (std::size_t i = 0; i < sequence.frames.size(); ++i) {
        precomputedQpos.push_back(
            retargeter.retargetFrame(sequence.frames[i], offsetToGround));
        if ((i + 1) % 300 == 0 || (i + 1) == sequence.frames.size()) {
          std::cout << "\r  progress: " << (i + 1) << "/"
                    << sequence.frames.size() << std::flush;
        }
      }
      std::cout << std::endl;
      retargeter.setQpos(precomputedQpos.front());
    }

    std::cout << "Starting playback at " << sequence.fps << " FPS ("
              << (realtime ? "realtime IK" : "precomputed") << ")" << std::endl;

    std::size_t frameIdx = 0;
    auto lastStep = std::chrono::steady_clock::now();
    auto playbackStart = std::chrono::steady_clock::now();
    std::size_t shownIdx = std::numeric_limits<std::size_t>::max();
    bool playbackFinished = false;

    while (!glfwWindowShouldClose(window)) {
      auto now = std::chrono::steady_clock::now();
      if (realtime) {
        while (now - lastStep >= frameDtDuration) {
          retargeter.retargetFrame(sequence.frames[frameIdx], offsetToGround);
          frameIdx++;
          if (frameIdx >= sequence.frames.size()) {
            if (loop) {
              frameIdx = 0;
            } else {
              frameIdx = sequence.frames.size() - 1;
              playbackFinished = true;
              break;
            }
          }
          lastStep += frameDtDuration;
        }
      } else {
        const double elapsedSec =
            std::chrono::duration<double>(now - playbackStart).count();
        std::size_t desiredIdx = static_cast<std::size_t>(elapsedSec / frameDt);
        if (loop) {
          desiredIdx %= precomputedQpos.size();
        } else if (desiredIdx >= precomputedQpos.size()) {
          break;
        }
        if (desiredIdx != shownIdx) {
          retargeter.setQpos(precomputedQpos[desiredIdx]);
          shownIdx = desiredIdx;
        }
      }

      if (rootBodyId >= 0) {
        const mjData *latestData = retargeter.data();
        cam.lookat[0] = latestData->xpos[3 * rootBodyId + 0];
        cam.lookat[1] = latestData->xpos[3 * rootBodyId + 1];
        cam.lookat[2] = latestData->xpos[3 * rootBodyId + 2];
      }

      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

      mjv_updateScene(model, retargeter.mutableData(), &opt, nullptr, &cam,
                      mjCAT_ALL, &scn);
      mjr_render(viewport, &scn, &con);

      glfwSwapBuffers(window);
      glfwPollEvents();

      if (playbackFinished) {
        break;
      }
    }

    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[gmr_retarget_viewer] Error: " << e.what() << std::endl;
    return 1;
  }
}
