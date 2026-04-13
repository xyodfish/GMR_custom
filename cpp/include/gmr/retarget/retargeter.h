#pragma once

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <mujoco/mujoco.h>

#include "gmr/retarget/ik_config.h"

namespace gmr {

struct HumanBodyState {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation =
      Eigen::Quaterniond::Identity(); // world frame, wxyz semantics
};

using HumanFrame = std::unordered_map<std::string, HumanBodyState>;

struct RetargetOptions {
  std::string solverName = "qpoases";
  double damping = 5e-1;
  int maxIterations = 100;
  bool useVelocityLimit = false;
  double velocityLimit = 3.0 * M_PI;
  double progressThreshold = 1e-3;
};

class MujocoRetargeter {
public:
  MujocoRetargeter(const std::filesystem::path &robotXmlPath, IkConfig ikConfig,
                   RetargetOptions options = {});
  ~MujocoRetargeter();

  MujocoRetargeter(const MujocoRetargeter &) = delete;
  MujocoRetargeter &operator=(const MujocoRetargeter &) = delete;

  Eigen::VectorXd retargetFrame(const HumanFrame &humanFrame,
                                bool offsetToGround = false);
  HumanFrame prepareHumanFrame(const HumanFrame &humanFrame,
                               bool offsetToGround = false) const;
  void setQpos(const Eigen::VectorXd &qpos);

  const Eigen::VectorXd &currentQpos() const { return qpos_; }
  const mjModel *model() const { return model_.get(); }
  const mjData *data() const { return data_.get(); }
  mjData *mutableData() { return data_.get(); }

private:
  struct TaskRuntime {
    int bodyId = -1;
    std::string humanBodyName;
    double posWeight = 0.0;
    double rotWeight = 0.0;
    Eigen::Vector3d posOffset = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotOffset = Eigen::Quaterniond::Identity();

    Eigen::Vector3d targetPos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond targetRot = Eigen::Quaterniond::Identity();
  };

  struct ModelDeleter {
    void operator()(mjModel *p) const {
      if (p != nullptr) {
        mj_deleteModel(p);
      }
    }
  };

  struct DataDeleter {
    void operator()(mjData *p) const {
      if (p != nullptr) {
        mj_deleteData(p);
      }
    }
  };

  std::unique_ptr<mjModel, ModelDeleter> model_;
  std::unique_ptr<mjData, DataDeleter> data_;

  IkConfig ikConfig_;
  RetargetOptions options_;

  std::vector<TaskRuntime> tasks1_;
  std::vector<TaskRuntime> tasks2_;
  std::unordered_map<std::string, Eigen::Vector3d> table1PosOffsets_;
  std::unordered_map<std::string, Eigen::Quaterniond> table1RotOffsets_;

  Eigen::VectorXd qpos_;
  Eigen::VectorXd qvel_;

  HumanFrame scaleAndOffsetHumanFrame(const HumanFrame &frame,
                                      bool offsetToGround) const;
  void updateTaskTargets(const HumanFrame &frame);
  double computeTaskError(const std::vector<TaskRuntime> &tasks) const;
  void solveTaskSet(const std::vector<TaskRuntime> &tasks);
  Eigen::Vector3d
  computeOrientationErrorWorld(const Eigen::Quaterniond &current,
                               const Eigen::Quaterniond &target) const;
  void syncDataFromQpos();
};

} // namespace gmr
