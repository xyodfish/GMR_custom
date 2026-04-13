#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

#include "gmr/retarget/human_frame_io.h"
#include "gmr/retarget/ik_config.h"
#include "gmr/retarget/repo_paths.h"
#include "gmr/retarget/retargeter.h"

namespace {

struct ViewerConfig {
  std::string gmrRoot;
  std::string robot;
  std::string srcHuman = "smplx";
  std::string humanFrameJson;
  double actualHumanHeight = 0.0;

  double damping = 0.5;
  int maxIter = 10;
  bool useVelocityLimit = false;

  bool offsetToGround = false;
  bool loop = false;
  bool realtime = true;

  int transparentRobot = 0;
  bool showHumanOverlay = true;
  int windowWidth = 1280;
  int windowHeight = 720;
};

std::string getArg(int argc, char **argv, const std::string &name,
                   const std::string &defaultValue = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) {
      return argv[i + 1];
    }
  }
  return defaultValue;
}

bool hasArg(int argc, char **argv, const std::string &name) {
  for (int i = 1; i < argc; ++i) {
    if (name == argv[i]) {
      return true;
    }
  }
  return false;
}

bool hasFlag(int argc, char **argv, const std::string &flag) {
  return hasArg(argc, argv, flag);
}

template <typename T>
void setIfPresent(const YAML::Node &root, const std::string &key, T *value) {
  const YAML::Node node = root[key];
  if (node) {
    *value = node.as<T>();
  }
}

const std::unordered_map<std::string, std::string> &robotBaseMap() {
  static const std::unordered_map<std::string, std::string> kMap = {
      {"unitree_g1", "pelvis"},
      {"unitree_g1_with_hands", "pelvis"},
      {"unitree_h1", "pelvis"},
      {"unitree_h1_2", "pelvis"},
      {"booster_t1", "Waist"},
      {"booster_t1_29dof", "Waist"},
      {"stanford_toddy", "waist_link"},
      {"fourier_n1", "base_link"},
      {"engineai_pm01", "LINK_BASE"},
      {"kuavo_s45", "base_link"},
      {"hightorque_hi", "base_link"},
      {"galaxea_r1pro", "torso_link4"},
      {"berkeley_humanoid_lite", "imu_2"},
      {"booster_k1", "Trunk"},
      {"pnd_adam_lite", "pelvis"},
      {"tienkung", "Base_link"},
      {"pal_talos", "base_link"},
      {"fourier_gr3", "base_link"},
  };
  return kMap;
}

const std::unordered_map<std::string, double> &viewerCamDistanceMap() {
  static const std::unordered_map<std::string, double> kMap = {
      {"unitree_g1", 2.0},
      {"unitree_g1_with_hands", 2.0},
      {"unitree_h1", 3.0},
      {"unitree_h1_2", 3.0},
      {"booster_t1", 2.0},
      {"booster_t1_29dof", 2.0},
      {"stanford_toddy", 1.0},
      {"fourier_n1", 2.0},
      {"engineai_pm01", 2.0},
      {"kuavo_s45", 3.0},
      {"hightorque_hi", 2.0},
      {"galaxea_r1pro", 3.0},
      {"berkeley_humanoid_lite", 2.0},
      {"booster_k1", 2.0},
      {"pnd_adam_lite", 3.0},
      {"tienkung", 3.0},
      {"pal_talos", 3.0},
      {"fourier_gr3", 2.0},
  };
  return kMap;
}

std::string resolveCameraBodyName(const std::string &robot,
                                  const std::string &fallback) {
  const auto &map = robotBaseMap();
  auto it = map.find(robot);
  if (it != map.end()) {
    return it->second;
  }
  return fallback;
}

double resolveCameraDistance(const std::string &robot, double fallback) {
  const auto &map = viewerCamDistanceMap();
  auto it = map.find(robot);
  if (it != map.end()) {
    return it->second;
  }
  return fallback;
}

void loadConfigYaml(const std::filesystem::path &configPath,
                    ViewerConfig *config) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(configPath.string());
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load YAML config: " +
                             configPath.string() + " error=" + e.what());
  }

  if (!root || !root.IsMap()) {
    throw std::runtime_error("YAML config must be a map/object.");
  }

  setIfPresent(root, "gmr_root", &config->gmrRoot);
  setIfPresent(root, "robot", &config->robot);
  setIfPresent(root, "src_human", &config->srcHuman);
  setIfPresent(root, "human_frame_json", &config->humanFrameJson);
  setIfPresent(root, "actual_human_height", &config->actualHumanHeight);

  setIfPresent(root, "damping", &config->damping);
  setIfPresent(root, "max_iter", &config->maxIter);
  setIfPresent(root, "use_velocity_limit", &config->useVelocityLimit);

  setIfPresent(root, "offset_to_ground", &config->offsetToGround);
  setIfPresent(root, "loop", &config->loop);

  bool precompute = false;
  if (root["precompute"]) {
    precompute = root["precompute"].as<bool>();
    config->realtime = !precompute;
  }
  setIfPresent(root, "realtime", &config->realtime);

  setIfPresent(root, "transparent_robot", &config->transparentRobot);
  setIfPresent(root, "show_human_overlay", &config->showHumanOverlay);
  setIfPresent(root, "window_width", &config->windowWidth);
  setIfPresent(root, "window_height", &config->windowHeight);
}

void applyCliOverrides(int argc, char **argv, ViewerConfig *config) {
  if (hasArg(argc, argv, "--gmr_root")) {
    config->gmrRoot = getArg(argc, argv, "--gmr_root");
  }
  if (hasArg(argc, argv, "--robot")) {
    config->robot = getArg(argc, argv, "--robot");
  }
  if (hasArg(argc, argv, "--src_human")) {
    config->srcHuman = getArg(argc, argv, "--src_human");
  }
  if (hasArg(argc, argv, "--human_frame_json")) {
    config->humanFrameJson = getArg(argc, argv, "--human_frame_json");
  }
  if (hasArg(argc, argv, "--actual_human_height")) {
    config->actualHumanHeight =
        std::stod(getArg(argc, argv, "--actual_human_height"));
  }

  if (hasArg(argc, argv, "--damping")) {
    config->damping = std::stod(getArg(argc, argv, "--damping"));
  }
  if (hasArg(argc, argv, "--max_iter")) {
    config->maxIter = std::stoi(getArg(argc, argv, "--max_iter"));
  }

  if (hasFlag(argc, argv, "--use_velocity_limit")) {
    config->useVelocityLimit = true;
  }
  if (hasFlag(argc, argv, "--no_use_velocity_limit")) {
    config->useVelocityLimit = false;
  }

  if (hasFlag(argc, argv, "--offset_to_ground")) {
    config->offsetToGround = true;
  }
  if (hasFlag(argc, argv, "--no_offset_to_ground")) {
    config->offsetToGround = false;
  }

  if (hasFlag(argc, argv, "--loop")) {
    config->loop = true;
  }
  if (hasFlag(argc, argv, "--no_loop")) {
    config->loop = false;
  }

  if (hasFlag(argc, argv, "--realtime")) {
    config->realtime = true;
  }
  if (hasFlag(argc, argv, "--precompute")) {
    config->realtime = false;
  }

  if (hasArg(argc, argv, "--transparent_robot")) {
    config->transparentRobot =
        std::stoi(getArg(argc, argv, "--transparent_robot"));
  }

  if (hasFlag(argc, argv, "--hide_human_overlay")) {
    config->showHumanOverlay = false;
  }
  if (hasFlag(argc, argv, "--show_human_overlay")) {
    config->showHumanOverlay = true;
  }

  if (hasArg(argc, argv, "--window_width")) {
    config->windowWidth = std::stoi(getArg(argc, argv, "--window_width"));
  }
  if (hasArg(argc, argv, "--window_height")) {
    config->windowHeight = std::stoi(getArg(argc, argv, "--window_height"));
  }
}

void validateConfig(const ViewerConfig &config) {
  if (config.gmrRoot.empty() || config.robot.empty() ||
      config.humanFrameJson.empty()) {
    throw std::runtime_error(
        "Missing required fields: gmr_root, robot, human_frame_json.");
  }
}

void drawHumanFrameAxes(const Eigen::Vector3d &pos, const Eigen::Matrix3d &mat,
                        mjvScene *scene, double size,
                        const Eigen::Vector3d &posOffset) {
  static const std::array<std::array<float, 4>, 3> kColors = {
      std::array<float, 4>{1.0f, 0.0f, 0.0f, 1.0f},
      std::array<float, 4>{0.0f, 1.0f, 0.0f, 1.0f},
      std::array<float, 4>{0.0f, 0.0f, 1.0f, 1.0f},
  };

  for (int i = 0; i < 3; ++i) {
    if (scene->ngeom >= scene->maxgeom) {
      return;
    }

    mjvGeom *geom = &scene->geoms[scene->ngeom];
    const Eigen::Vector3d p = pos + posOffset;
    const Eigen::Vector3d to = p + size * mat.col(i);

    mjtNum mjPos[3] = {p.x(), p.y(), p.z()};
    mjtNum mjTo[3] = {to.x(), to.y(), to.z()};
    mjtNum mjSize[3] = {0.01, 0.01, 0.01};
    mjtNum mjMat[9] = {mat(0, 0), mat(0, 1), mat(0, 2), mat(1, 0), mat(1, 1),
                       mat(1, 2), mat(2, 0), mat(2, 1), mat(2, 2)};
    float mjRgba[4] = {kColors[i][0], kColors[i][1], kColors[i][2],
                       kColors[i][3]};

    mjv_initGeom(geom, mjGEOM_ARROW, mjSize, mjPos, mjMat, mjRgba);
    mjv_connector(geom, mjGEOM_ARROW, 0.005, mjPos, mjTo);
    scene->ngeom += 1;
  }
}

void drawHumanOverlay(const gmr::HumanFrame &humanFrame, mjvScene *scene,
                      double pointScale,
                      const Eigen::Vector3d &humanPosOffset) {
  for (const auto &[name, body] : humanFrame) {
    (void)name;
    drawHumanFrameAxes(body.position,
                       body.orientation.normalized().toRotationMatrix(), scene,
                       pointScale, humanPosOffset);
  }
}

void printUsage() {
  std::cout
      << "Usage:\n"
      << "  gmr_retarget_viewer"
      << " [--config <viewer_config.yaml>]"
      << " [--gmr_root <path_to_GMR_root>]"
      << " [--robot <robot_name>]"
      << " [--src_human <smplx|bvh_lafan1|bvh_nokov>]"
      << " [--human_frame_json <single_or_multi_frame_json>]"
      << " [--actual_human_height <float>]"
      << " [--damping <float>]"
      << " [--max_iter <int>]"
      << " [--use_velocity_limit|--no_use_velocity_limit]"
      << " [--offset_to_ground|--no_offset_to_ground]"
      << " [--loop|--no_loop]"
      << " [--realtime|--precompute]"
      << " [--transparent_robot <0|1>]"
      << " [--hide_human_overlay|--show_human_overlay]"
      << " [--window_width <int>]"
      << " [--window_height <int>]"
      << "\n\n"
      << "Defaults: realtime=true (no precompute), loop=false, "
         "show_human_overlay=true.\n"
      << "CLI options override config file values.\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1 || hasFlag(argc, argv, "--help")) {
      printUsage();
      return 0;
    }

    ViewerConfig config;
    if (hasArg(argc, argv, "--config")) {
      loadConfigYaml(getArg(argc, argv, "--config"), &config);
    }
    applyCliOverrides(argc, argv, &config);
    validateConfig(config);

    const std::filesystem::path gmrRoot(config.gmrRoot);
    const std::string robot = config.robot;

    gmr::RetargetOptions opts;
    opts.damping = config.damping;
    opts.maxIterations = config.maxIter;
    opts.useVelocityLimit = config.useVelocityLimit;

    const std::filesystem::path xmlPath = gmr::resolveRobotXml(gmrRoot, robot);
    const std::filesystem::path ikPath =
        gmr::resolveIkConfig(gmrRoot, config.srcHuman, robot);
    gmr::IkConfig ikConfig =
        gmr::loadIkConfig(ikPath, config.actualHumanHeight);

    const gmr::HumanFrameSequence sequence =
        gmr::loadHumanFrameSequence(config.humanFrameJson);
    if (sequence.frames.empty()) {
      throw std::runtime_error("No frames to render.");
    }

    gmr::MujocoRetargeter retargeter(xmlPath, ikConfig, opts);

    if (glfwInit() == GLFW_FALSE) {
      throw std::runtime_error("glfwInit failed.");
    }

    GLFWwindow *window =
        glfwCreateWindow(config.windowWidth, config.windowHeight,
                         "GMR C++ Retarget Viewer", nullptr, nullptr);
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

    const double humanPointScale = 0.1;
    const Eigen::Vector3d humanPosOffset = Eigen::Vector3d::Zero();
    opt.flags[mjVIS_TRANSPARENT] = config.transparentRobot;

    mjv_makeScene(model, &scn, 5000);
    mjr_makeContext(model, &con, mjFONTSCALE_150);

    const std::string cameraBodyName =
        resolveCameraBodyName(robot, ikConfig.robotRootName);
    int cameraBodyId = mj_name2id(model, mjOBJ_BODY, cameraBodyName.c_str());
    if (cameraBodyId < 0) {
      cameraBodyId =
          mj_name2id(model, mjOBJ_BODY, ikConfig.robotRootName.c_str());
    }
    const double cameraDistance = resolveCameraDistance(robot, 2.5);
    cam.distance = cameraDistance;
    cam.elevation = -10.0;
    if (cameraBodyId >= 0) {
      cam.lookat[0] = data->xpos[3 * cameraBodyId + 0];
      cam.lookat[1] = data->xpos[3 * cameraBodyId + 1];
      cam.lookat[2] = data->xpos[3 * cameraBodyId + 2];
    }

    const double frameDt = 1.0 / static_cast<double>(sequence.fps);
    const auto frameDtDuration =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(frameDt));

    std::vector<Eigen::VectorXd> precomputedQpos;
    if (!config.realtime) {
      precomputedQpos.reserve(sequence.frames.size());
      std::cout << "Precomputing retarget trajectory ("
                << sequence.frames.size() << " frames)..." << std::endl;
      for (std::size_t i = 0; i < sequence.frames.size(); ++i) {
        precomputedQpos.push_back(
            retargeter.retargetFrame(sequence.frames[i], config.offsetToGround));
        if ((i + 1) % 300 == 0 || (i + 1) == sequence.frames.size()) {
          std::cout << "\r  progress: " << (i + 1) << "/"
                    << sequence.frames.size() << std::flush;
        }
      }
      std::cout << std::endl;
      retargeter.setQpos(precomputedQpos.front());
    }

    std::cout << "Starting playback at " << sequence.fps << " FPS ("
              << (config.realtime ? "realtime IK" : "precomputed") << ")"
              << std::endl;

    std::size_t frameIdx = 0;
    auto lastStep = std::chrono::steady_clock::now();
    auto playbackStart = std::chrono::steady_clock::now();
    std::size_t shownIdx = std::numeric_limits<std::size_t>::max();
    bool playbackFinished = false;
    gmr::HumanFrame visualHumanFrame =
        retargeter.prepareHumanFrame(sequence.frames.front(),
                                     config.offsetToGround);

    while (!glfwWindowShouldClose(window)) {
      auto now = std::chrono::steady_clock::now();
      if (config.realtime) {
        while (now - lastStep >= frameDtDuration) {
          visualHumanFrame = retargeter.prepareHumanFrame(
              sequence.frames[frameIdx], config.offsetToGround);
          retargeter.retargetFrame(sequence.frames[frameIdx],
                                   config.offsetToGround);
          frameIdx++;
          if (frameIdx >= sequence.frames.size()) {
            if (config.loop) {
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
        if (config.loop) {
          desiredIdx %= precomputedQpos.size();
        } else if (desiredIdx >= precomputedQpos.size()) {
          break;
        }
        if (desiredIdx != shownIdx) {
          retargeter.setQpos(precomputedQpos[desiredIdx]);
          visualHumanFrame = retargeter.prepareHumanFrame(
              sequence.frames[desiredIdx], config.offsetToGround);
          shownIdx = desiredIdx;
        }
      }

      if (cameraBodyId >= 0) {
        const mjData *latestData = retargeter.data();
        cam.lookat[0] = latestData->xpos[3 * cameraBodyId + 0];
        cam.lookat[1] = latestData->xpos[3 * cameraBodyId + 1];
        cam.lookat[2] = latestData->xpos[3 * cameraBodyId + 2];
        cam.distance = cameraDistance;
        cam.elevation = -10.0;
      }

      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

      mjv_updateScene(model, retargeter.mutableData(), &opt, nullptr, &cam,
                      mjCAT_ALL, &scn);
      if (config.showHumanOverlay) {
        drawHumanOverlay(visualHumanFrame, &scn, humanPointScale,
                         humanPosOffset);
      }
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
