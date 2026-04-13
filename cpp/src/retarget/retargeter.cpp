#include "gmr/retarget/retargeter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "gmr/solver/qp_solver.h"

namespace gmr {
namespace {

Eigen::Quaterniond quatFromWxyz(const mjtNum *q) {
  return Eigen::Quaterniond(q[0], q[1], q[2], q[3]);
}

} // namespace

MujocoRetargeter::MujocoRetargeter(const std::filesystem::path &robotXmlPath,
                                   IkConfig ikConfig, RetargetOptions options)
    : ikConfig_(std::move(ikConfig)), options_(std::move(options)) {
  std::array<char, 1024> error{};
  mjModel *rawModel = mj_loadXML(robotXmlPath.string().c_str(), nullptr,
                                 error.data(), error.size());
  if (rawModel == nullptr) {
    throw std::runtime_error(
        "Failed to load MuJoCo XML: " + robotXmlPath.string() +
        " error=" + std::string(error.data()));
  }
  model_.reset(rawModel);

  mjData *rawData = mj_makeData(model_.get());
  if (rawData == nullptr) {
    throw std::runtime_error("Failed to allocate MuJoCo data.");
  }
  data_.reset(rawData);

  qpos_ = Eigen::Map<Eigen::VectorXd>(data_->qpos, model_->nq);
  qvel_ = Eigen::VectorXd::Zero(model_->nv);

  auto buildTasks = [this](const std::vector<IkTaskEntry> &src,
                           std::vector<TaskRuntime> *dst) {
    dst->clear();
    dst->reserve(src.size());
    for (const auto &entry : src) {
      const int bodyId =
          mj_name2id(model_.get(), mjOBJ_BODY, entry.robotBodyName.c_str());
      if (bodyId < 0) {
        throw std::runtime_error("Body not found in MuJoCo model: " +
                                 entry.robotBodyName);
      }

      TaskRuntime task;
      task.bodyId = bodyId;
      task.humanBodyName = entry.humanBodyName;
      task.posWeight = entry.posWeight;
      task.rotWeight = entry.rotWeight;
      task.posOffset = entry.posOffset;
      task.rotOffset = entry.rotOffset;
      dst->push_back(std::move(task));
    }
  };

  buildTasks(ikConfig_.tasksTable1, &tasks1_);
  buildTasks(ikConfig_.tasksTable2, &tasks2_);

  for (const auto &task : tasks1_) {
    table1PosOffsets_[task.humanBodyName] =
        task.posOffset - Eigen::Vector3d(0.0, 0.0, ikConfig_.groundHeight);
    table1RotOffsets_[task.humanBodyName] = task.rotOffset;
  }

  syncDataFromQpos();
}

MujocoRetargeter::~MujocoRetargeter() = default;

HumanFrame
MujocoRetargeter::scaleAndOffsetHumanFrame(const HumanFrame &frame,
                                           bool offsetToGround) const {
  auto rootIt = frame.find(ikConfig_.humanRootName);
  if (rootIt == frame.end()) {
    throw std::runtime_error("Human frame misses root body: " +
                             ikConfig_.humanRootName);
  }

  HumanFrame result;
  const Eigen::Vector3d rootPos = rootIt->second.position;
  const Eigen::Quaterniond rootRot = rootIt->second.orientation;

  const auto rootScaleIt =
      ikConfig_.humanScaleTable.find(ikConfig_.humanRootName);
  const double rootScale = rootScaleIt == ikConfig_.humanScaleTable.end()
                               ? 1.0
                               : rootScaleIt->second;
  const Eigen::Vector3d scaledRootPos = rootScale * rootPos;
  result[ikConfig_.humanRootName] = HumanBodyState{scaledRootPos, rootRot};

  for (const auto &[bodyName, scale] : ikConfig_.humanScaleTable) {
    if (bodyName == ikConfig_.humanRootName) {
      continue;
    }
    auto bodyIt = frame.find(bodyName);
    if (bodyIt == frame.end()) {
      continue;
    }

    const Eigen::Vector3d localPos =
        (bodyIt->second.position - rootPos) * scale;
    result[bodyName] =
        HumanBodyState{localPos + scaledRootPos, bodyIt->second.orientation};
  }

  for (auto &[bodyName, body] : result) {
    auto posIt = table1PosOffsets_.find(bodyName);
    auto rotIt = table1RotOffsets_.find(bodyName);
    if (posIt == table1PosOffsets_.end() || rotIt == table1RotOffsets_.end()) {
      continue;
    }

    body.orientation = body.orientation * rotIt->second;
    body.position += body.orientation * posIt->second;
  }

  for (auto &[bodyName, body] : result) {
    (void)bodyName;
    body.position[2] -= 0.0;
  }

  if (offsetToGround) {
    double lowest = std::numeric_limits<double>::infinity();
    for (const auto &[bodyName, body] : result) {
      if (bodyName.find("Foot") == std::string::npos &&
          bodyName.find("foot") == std::string::npos) {
        continue;
      }
      lowest = std::min(lowest, body.position[2]);
    }

    if (std::isfinite(lowest)) {
      for (auto &[bodyName, body] : result) {
        (void)bodyName;
        body.position[2] = body.position[2] - lowest + 0.1;
      }
    }
  }

  return result;
}

void MujocoRetargeter::updateTaskTargets(const HumanFrame &frame) {
  auto fill = [&frame](std::vector<TaskRuntime> *tasks) {
    for (auto &task : *tasks) {
      auto it = frame.find(task.humanBodyName);
      if (it == frame.end()) {
        continue;
      }
      task.targetPos = it->second.position;
      task.targetRot = it->second.orientation;
    }
  };

  fill(&tasks1_);
  fill(&tasks2_);
}

Eigen::Vector3d MujocoRetargeter::computeOrientationErrorWorld(
    const Eigen::Quaterniond &current, const Eigen::Quaterniond &target) const {
  Eigen::Quaterniond qErr = target * current.conjugate();
  if (qErr.w() < 0.0) {
    qErr.coeffs() *= -1.0;
  }

  const Eigen::Vector3d vec = qErr.vec();
  const double vecNorm = vec.norm();
  if (vecNorm < 1e-12) {
    return Eigen::Vector3d::Zero();
  }

  const double angle = 2.0 * std::atan2(vecNorm, qErr.w());
  return vec / vecNorm * angle;
}

double MujocoRetargeter::computeTaskError(
    const std::vector<TaskRuntime> &tasks) const {
  double sqErr = 0.0;
  for (const auto &task : tasks) {
    if (task.bodyId < 0) {
      continue;
    }

    const mjtNum *xpos = data_->xpos + 3 * task.bodyId;
    const mjtNum *xquat = data_->xquat + 4 * task.bodyId;

    const Eigen::Vector3d currPos(xpos[0], xpos[1], xpos[2]);
    const Eigen::Quaterniond currRot = quatFromWxyz(xquat);

    const Eigen::Vector3d posErr = task.targetPos - currPos;
    const Eigen::Vector3d rotErr =
        computeOrientationErrorWorld(currRot, task.targetRot);
    sqErr += posErr.squaredNorm() + rotErr.squaredNorm();
  }
  return std::sqrt(sqErr);
}

void MujocoRetargeter::solveTaskSet(const std::vector<TaskRuntime> &tasks) {
  if (tasks.empty()) {
    return;
  }

  const int nv = model_->nv;
  const double dt = model_->opt.timestep;
  const double invDt = dt > 1e-12 ? 1.0 / dt : 1.0;

  double currError = computeTaskError(tasks);
  solver::QPSolver solver;

  for (int iter = 0; iter < options_.maxIterations; ++iter) {
    solver::QPData qp;
    qp.reset(nv, nv);

    qp.CI.setIdentity();
    qp.ciLb.setConstant(-1e9);
    qp.ciUb.setConstant(1e9);

    if (options_.useVelocityLimit) {
      qp.ciLb.setConstant(-options_.velocityLimit);
      qp.ciUb.setConstant(options_.velocityLimit);
    }

    for (int j = 0; j < model_->njnt; ++j) {
      const int jointType = model_->jnt_type[j];
      const int limited = model_->jnt_limited[j];
      if ((jointType == mjJNT_HINGE || jointType == mjJNT_SLIDE) &&
          limited > 0) {
        const int qadr = model_->jnt_qposadr[j];
        const int vadr = model_->jnt_dofadr[j];
        const double qmin = model_->jnt_range[2 * j + 0];
        const double qmax = model_->jnt_range[2 * j + 1];

        qp.ciLb[vadr] = std::max(qp.ciLb[vadr], (qmin - qpos_[qadr]) / dt);
        qp.ciUb[vadr] = std::min(qp.ciUb[vadr], (qmax - qpos_[qadr]) / dt);
      }
    }

    std::vector<mjtNum> jacp(3 * nv, 0.0);
    std::vector<mjtNum> jacr(3 * nv, 0.0);

    for (const auto &task : tasks) {
      if (task.bodyId < 0) {
        continue;
      }

      mj_jacBody(model_.get(), data_.get(), jacp.data(), jacr.data(),
                 task.bodyId);

      const Eigen::Map<
          const Eigen::Matrix<mjtNum, 3, Eigen::Dynamic, Eigen::RowMajor>>
          Jp(jacp.data(), 3, nv);
      const Eigen::Map<
          const Eigen::Matrix<mjtNum, 3, Eigen::Dynamic, Eigen::RowMajor>>
          Jr(jacr.data(), 3, nv);

      const mjtNum *xpos = data_->xpos + 3 * task.bodyId;
      const mjtNum *xquat = data_->xquat + 4 * task.bodyId;
      const Eigen::Vector3d currPos(xpos[0], xpos[1], xpos[2]);
      const Eigen::Quaterniond currRot = quatFromWxyz(xquat);

      const Eigen::Vector3d posErr = task.targetPos - currPos;
      const Eigen::Vector3d rotErr =
          computeOrientationErrorWorld(currRot, task.targetRot);
      const Eigen::Vector3d posTargetVel = posErr * invDt;
      const Eigen::Vector3d rotTargetVel = rotErr * invDt;

      if (task.posWeight > 0.0) {
        qp.H.noalias() += task.posWeight * (Jp.transpose() * Jp);
        qp.g.noalias() += -task.posWeight * (Jp.transpose() * posTargetVel);
      }
      if (task.rotWeight > 0.0) {
        qp.H.noalias() += task.rotWeight * (Jr.transpose() * Jr);
        qp.g.noalias() += -task.rotWeight * (Jr.transpose() * rotTargetVel);
      }
    }

    qp.H.diagonal().array() += options_.damping;

    const solver::QPOutput &out = solver.solve(qp);
    if (out.status == solver::QPStatus::kOptimal) {
      qvel_ = out.x;
    } else {
      throw std::runtime_error("QP solver failed while retargeting.");
    }

    mj_integratePos(model_.get(), qpos_.data(), qvel_.data(), dt);
    syncDataFromQpos();

    const double nextError = computeTaskError(tasks);
    if (currError - nextError <= options_.progressThreshold) {
      break;
    }
    currError = nextError;
  }
}

void MujocoRetargeter::syncDataFromQpos() {
  std::copy(qpos_.data(), qpos_.data() + model_->nq, data_->qpos);
  std::copy(qvel_.data(), qvel_.data() + model_->nv, data_->qvel);
  mj_forward(model_.get(), data_.get());
}

Eigen::VectorXd MujocoRetargeter::retargetFrame(const HumanFrame &humanFrame,
                                                bool offsetToGround) {
  HumanFrame prepared = scaleAndOffsetHumanFrame(humanFrame, offsetToGround);
  updateTaskTargets(prepared);

  if (ikConfig_.useTable1) {
    solveTaskSet(tasks1_);
  }
  if (ikConfig_.useTable2) {
    solveTaskSet(tasks2_);
  }

  return qpos_;
}

HumanFrame MujocoRetargeter::prepareHumanFrame(const HumanFrame &humanFrame,
                                               bool offsetToGround) const {
  return scaleAndOffsetHumanFrame(humanFrame, offsetToGround);
}

void MujocoRetargeter::setQpos(const Eigen::VectorXd &qpos) {
  if (qpos.size() != qpos_.size()) {
    throw std::runtime_error("setQpos size mismatch.");
  }
  qpos_ = qpos;
  qvel_.setZero();
  syncDataFromQpos();
}

} // namespace gmr
