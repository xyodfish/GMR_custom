#include "gmr/retarget/batch_trajectory_retarget.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>

#include <pinocchio/spatial/explog.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "batch_trajectory_banded_solver.h"
#include "gmr/solver/qp_solver.h"
#include "retargeter_internal_utils.h"

namespace gmr {
    namespace {

        using Clock = std::chrono::steady_clock;

        double elapsedMs(const Clock::time_point& t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }

        Eigen::Vector3d quatRotError(const Eigen::Quaterniond& target, const Eigen::Quaterniond& current) {
            const Eigen::Matrix3d Rbody = current.toRotationMatrix();
            const Eigen::Matrix3d Rt    = target.toRotationMatrix();
            // Match scipy Rotation.inv() * R_body; as_rotvec() used in Python batch TO.
            return pinocchio::log3(Rt.transpose() * Rbody);
        }

        Eigen::Vector3d taskPosOffset(const IkTaskEntry& task, double groundHeight) {
            return task.posOffset - Eigen::Vector3d(0.0, 0.0, groundHeight);
        }

        Eigen::Matrix3d bodyRotation(const mjData* data, int bodyId) {
            const double* xmat = &data->xmat[9 * bodyId];
            Eigen::Matrix3d R;
            R << xmat[0], xmat[1], xmat[2], xmat[3], xmat[4], xmat[5], xmat[6], xmat[7], xmat[8];
            return R;
        }

        int gnScalarBandwidth(int m) {
            return 2 * m;
        }

        Eigen::Vector3d mjJacColumn(const double* jac, int nv, int v) {
            return Eigen::Vector3d(jac[v], jac[nv + v], jac[2 * nv + v]);
        }

    }  // namespace

    struct BatchTrajectoryRetargeter::GnWorkspace {
        batch_internal::SymmetricBandedMatrix Hband;
        Eigen::MatrixXd Hdense;
        Eigen::VectorXd g;
        Eigen::VectorXd dqFlat;
        std::vector<double> jacp;
        std::vector<double> jacr;
        std::vector<double> dq;
        std::vector<std::vector<Eigen::Vector3d>> footPos;
        std::vector<std::vector<Eigen::MatrixXd>> footJxy;
        std::vector<std::vector<Eigen::RowVectorXd>> footJz;
        bool useDense = false;
    };

    struct BatchTrajectoryRetargeter::Impl {
        struct ModelDeleter {
            void operator()(mjModel* p) const {
                if (p != nullptr) {
                    mj_deleteModel(p);
                }
            }
        };
        struct DataDeleter {
            void operator()(mjData* p) const {
                if (p != nullptr) {
                    mj_deleteData(p);
                }
            }
        };

        std::unique_ptr<mjModel, ModelDeleter> model;
        std::unique_ptr<mjData, DataDeleter> data;
    };

    BatchTrajectoryRetargeter::BatchTrajectoryRetargeter(const std::filesystem::path& robotModelPath, IkConfig ikConfig,
                                                         BatchTrajectoryConfig config)
        : config_(std::move(config)),
          ikConfig_(std::move(ikConfig)),
          impl_(std::make_unique<Impl>()),
          gnWs_(std::make_unique<GnWorkspace>()),
          robotModelPath_(robotModelPath) {
        std::array<char, 1024> error{};
        mjModel* rawModel = mj_loadXML(robotModelPath.string().c_str(), nullptr, error.data(), error.size());
        if (rawModel == nullptr) {
            throw std::runtime_error("Failed to load MuJoCo model: " + robotModelPath.string() + " error=" + std::string(error.data()));
        }
        impl_->model.reset(rawModel);
        impl_->data.reset(mj_makeData(impl_->model.get()));
        if (!impl_->data) {
            throw std::runtime_error("Failed to allocate MuJoCo data.");
        }
        nq_ = impl_->model->nq;

        for (const auto& task : ikConfig_.tasksTable1) {
            table1PosOffsets_[task.humanBodyName] = task.posOffset - Eigen::Vector3d(0.0, 0.0, ikConfig_.groundHeight);
            table1RotOffsets_[task.humanBodyName] = task.rotOffset;
        }

        buildTrackEntries();
        buildOptIndices();
        buildSmoothMappings();
        resolveFootBodyIds();
    }

    BatchTrajectoryRetargeter::~BatchTrajectoryRetargeter() = default;

    void BatchTrajectoryRetargeter::buildTrackEntries() {
        trackEntries_.clear();
        std::unordered_map<std::string, TrackEntry> merged;

        auto mergeTasks = [&](const std::vector<IkTaskEntry>& tasks) {
            for (const auto& task : tasks) {
                if (task.posWeight <= 0.0 && task.rotWeight <= 0.0) {
                    continue;
                }
                const int bodyId = mj_name2id(impl_->model.get(), mjOBJ_BODY, task.robotBodyName.c_str());
                if (bodyId < 0) {
                    continue;
                }
                auto it = merged.find(task.robotBodyName);
                if (it == merged.end()) {
                    merged[task.robotBodyName] = TrackEntry{bodyId, task.posWeight, task.rotWeight};
                } else {
                    it->second.posWeight = std::max(it->second.posWeight, task.posWeight);
                    it->second.rotWeight = std::max(it->second.rotWeight, task.rotWeight);
                }
            }
        };

        if (ikConfig_.useTable1) {
            mergeTasks(ikConfig_.tasksTable1);
        }
        if (ikConfig_.useTable2) {
            mergeTasks(ikConfig_.tasksTable2);
        }

        trackEntries_.reserve(merged.size());
        for (const auto& [name, entry] : merged) {
            (void)name;
            trackEntries_.push_back(entry);
        }
    }

    void BatchTrajectoryRetargeter::buildOptIndices() {
        optVidx_.clear();
        smoothQidx_.clear();

        mjModel* model = impl_->model.get();
        if (model->njnt > 0 && model->jnt_type[0] == mjJNT_FREE) {
            const int nvFree = std::min(6, static_cast<int>(model->nv));
            for (int v = 0; v < nvFree; ++v) {
                optVidx_.push_back(v);
            }
            const int nSmoothRoot = std::min(3, static_cast<int>(model->nq));
            for (int q = 0; q < nSmoothRoot; ++q) {
                smoothQidx_.push_back(q);
            }
        }

        for (int j = 0; j < model->njnt; ++j) {
            const int jtype = model->jnt_type[j];
            if (jtype != mjJNT_HINGE && jtype != mjJNT_SLIDE) {
                continue;
            }
            optVidx_.push_back(model->jnt_dofadr[j]);
            smoothQidx_.push_back(model->jnt_qposadr[j]);
        }

        if (!config_.smoothRootXyz) {
            smoothQidx_.erase(std::remove_if(smoothQidx_.begin(), smoothQidx_.end(), [](int q) { return q < 3; }), smoothQidx_.end());
        }
    }

    void BatchTrajectoryRetargeter::buildSmoothMappings() {
        smoothV_.clear();
        smoothQ_.clear();
        qToOptV_.clear();

        mjModel* model = impl_->model.get();
        const int m    = static_cast<int>(optVidx_.size());
        for (int i = 0; i < m; ++i) {
            const int v                     = optVidx_[i];
            const int j                     = model->dof_jntid[v];
            qToOptV_[model->jnt_qposadr[j]] = i;
            auto it                         = std::find(smoothQidx_.begin(), smoothQidx_.end(), model->jnt_qposadr[j]);
            if (it != smoothQidx_.end()) {
                smoothQ_.push_back(*it);
                smoothV_.push_back(i);
            }
        }
    }

    void BatchTrajectoryRetargeter::ensureGnWorkspace(int nFrames) const {
        mjModel* model = impl_->model.get();
        const int m    = static_cast<int>(optVidx_.size());
        const int nvar = nFrames * m;
        const int bw   = gnScalarBandwidth(m);

        gnWs_->g.resize(nvar);
        gnWs_->dqFlat.resize(nvar);
        gnWs_->jacp.assign(3 * model->nv, 0.0);
        gnWs_->jacr.assign(3 * model->nv, 0.0);
        gnWs_->dq.assign(model->nv, 0.0);

        gnWs_->footPos.assign(nFrames, std::vector<Eigen::Vector3d>(footBodyIds_.size()));
        gnWs_->footJxy.assign(nFrames, std::vector<Eigen::MatrixXd>(footBodyIds_.size()));
        gnWs_->footJz.assign(nFrames, std::vector<Eigen::RowVectorXd>(footBodyIds_.size()));

        gnWs_->useDense = true;
        gnWs_->Hdense.resize(nvar, nvar);
        if (config_.useBandedSolver) {
            gnWs_->Hband.resize(nvar, bw);
        }
    }

    void BatchTrajectoryRetargeter::resolveFootBodyIds() {
        footBodyIds_.clear();
        const char* names[] = {"left_ankle_roll_link", "right_ankle_roll_link"};
        for (const char* name : names) {
            const int id = mj_name2id(impl_->model.get(), mjOBJ_BODY, name);
            if (id >= 0) {
                footBodyIds_.push_back(id);
            }
        }
    }

    void BatchTrajectoryRetargeter::applyContactGroundConfig(const ContactGroundConfig& contactGround) {
        groundZ_ = contactGround.groundZ;
        footBodyIds_.clear();
        mjModel* model = impl_->model.get();

        auto resolveNames = [&](const std::vector<std::string>& names) {
            for (const auto& name : names) {
                const int id = mj_name2id(model, mjOBJ_BODY, name.c_str());
                if (id >= 0) {
                    footBodyIds_.push_back(id);
                }
            }
        };

        if (!contactGround.robotFootBodies.empty()) {
            resolveNames(contactGround.robotFootBodies);
        } else if (!contactGround.footBodies.empty()) {
            resolveNames(contactGround.footBodies);
        }
        if (footBodyIds_.empty()) {
            resolveFootBodyIds();
        }

        if (contactGround.enabled) {
            contactGroundPipeline_ = std::make_unique<ContactGroundPipeline>(contactGround, model);
        } else {
            contactGroundPipeline_.reset();
        }
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::loadQInitFromJson(const std::filesystem::path& path,
                                                                              std::size_t expectedFrames) const {
        std::ifstream ifs(path);
        if (!ifs) {
            throw std::runtime_error("Failed to open q_init json: " + path.string());
        }
        nlohmann::json root;
        ifs >> root;
        if (!root.contains("qpos_frames")) {
            throw std::runtime_error("q_init json must contain qpos_frames array.");
        }
        std::vector<Eigen::VectorXd> out;
        out.reserve(root["qpos_frames"].size());
        for (const auto& row : root["qpos_frames"]) {
            Eigen::VectorXd q(row.size());
            for (std::size_t i = 0; i < row.size(); ++i) {
                q[static_cast<Eigen::Index>(i)] = row[i].get<double>();
            }
            out.push_back(q);
        }
        if (expectedFrames > 0 && out.size() < expectedFrames) {
            throw std::runtime_error("q_init json has fewer frames than human sequence.");
        }
        if (expectedFrames > 0) {
            out.resize(expectedFrames);
        }
        return out;
    }

    HumanFrame BatchTrajectoryRetargeter::prepareHumanFrame(const HumanFrame& frame, bool offsetToGround) const {
        const bool useOffsetToGround = offsetToGround && contactGroundPipeline_ == nullptr;
        HumanFrame scaled            = retarget_internal::scaleHumanFrameOnly(frame, ikConfig_, useOffsetToGround);
        if (contactGroundPipeline_ != nullptr) {
            return contactGroundPipeline_->processHumanFrame(scaled);
        }
        return scaled;
    }

    BatchTrajectoryRetargeter::FrameTargets BatchTrajectoryRetargeter::targetsForPrepared(const HumanFrame& prepared) const {
        FrameTargets out;

        auto fillFromTasks = [&](const std::vector<IkTaskEntry>& tasks) {
            for (const auto& task : tasks) {
                auto bodyIt = prepared.find(task.humanBodyName);
                if (bodyIt == prepared.end()) {
                    continue;
                }
                const int bodyId = mj_name2id(impl_->model.get(), mjOBJ_BODY, task.robotBodyName.c_str());
                if (bodyId < 0) {
                    continue;
                }
                const auto [targetPos, targetRot] = retarget_internal::applyBodyOffset(
                    bodyIt->second.position, bodyIt->second.orientation, taskPosOffset(task, ikConfig_.groundHeight), task.rotOffset);
                FrameTaskTarget tgt;
                tgt.bodyId    = bodyId;
                tgt.targetPos = targetPos;
                tgt.targetRot = targetRot;
                tgt.targetRot.normalize();
                out[bodyId] = tgt;
            }
        };

        if (ikConfig_.useTable1) {
            fillFromTasks(ikConfig_.tasksTable1);
        }
        if (ikConfig_.useTable2) {
            fillFromTasks(ikConfig_.tasksTable2);
        }
        return out;
    }

    void BatchTrajectoryRetargeter::clipHingeQpos(Eigen::VectorXd& q) const {
        mjModel* model = impl_->model.get();
        for (int j = 0; j < model->njnt; ++j) {
            const int jtype = model->jnt_type[j];
            if ((jtype == mjJNT_HINGE || jtype == mjJNT_SLIDE) && model->jnt_limited[j] > 0) {
                const int qadr    = model->jnt_qposadr[j];
                const double qmin = model->jnt_range[2 * j + 0];
                const double qmax = model->jnt_range[2 * j + 1];
                q[qadr]           = std::min(std::max(q[qadr], qmin), qmax);
            }
        }
    }

    void BatchTrajectoryRetargeter::clipHingeQposMargin(Eigen::VectorXd& q, double marginDeg) const {
        if (marginDeg <= 0.0) {
            clipHingeQpos(q);
            return;
        }
        mjModel* model         = impl_->model.get();
        const double marginRad = marginDeg * 0.017453292519943295;
        for (int j = 0; j < model->njnt; ++j) {
            const int jtype = model->jnt_type[j];
            if ((jtype != mjJNT_HINGE && jtype != mjJNT_SLIDE) || model->jnt_limited[j] <= 0) {
                continue;
            }
            const int qadr = model->jnt_qposadr[j];
            double qmin    = model->jnt_range[2 * j + 0];
            double qmax    = model->jnt_range[2 * j + 1];
            if (jtype == mjJNT_HINGE && (qmax - qmin) > 2.0 * marginRad) {
                qmin += marginRad;
                qmax -= marginRad;
            }
            q[qadr] = std::min(std::max(q[qadr], qmin), qmax);
        }
    }

    void BatchTrajectoryRetargeter::applyGnStepToWindow(std::vector<Eigen::VectorXd>& qWin, const Eigen::VectorXd& dqFlat,
                                                        double alpha) const {
        mjModel* model    = impl_->model.get();
        const int nFrames = static_cast<int>(qWin.size());
        const int m       = static_cast<int>(optVidx_.size());
        std::vector<double> dq(model->nv, 0.0);
        for (int t = 0; t < nFrames; ++t) {
            std::fill(dq.begin(), dq.end(), 0.0);
            for (int vi = 0; vi < m; ++vi) {
                dq[optVidx_[vi]] = -alpha * dqFlat[t * m + vi];
            }
            mj_integratePos(model, qWin[t].data(), dq.data(), 1.0);
            clipHingeQpos(qWin[t]);
        }
    }

    double BatchTrajectoryRetargeter::windowCost(const std::vector<Eigen::VectorXd>& qWin, const std::vector<FrameTargets>& targets,
                                                 const Eigen::VectorXd& anchor, const std::vector<Eigen::VectorXd>& qRef, int frameOffset,
                                                 double anchorWeight) const {
        const int nFrames = static_cast<int>(qWin.size());
        double cost       = 0.0;
        double fkCost     = 0.0;
        double velCost    = 0.0;
        double accCost    = 0.0;
        double anchorCost = 0.0;
        double footCost   = 0.0;

        mjModel* model = impl_->model.get();
        mjData* data   = impl_->data.get();

        const bool footActive = config_.enableFootPenalties && !footBodyIds_.empty() &&
                                (config_.wFootHeight > 0.0 || config_.wFootSlip > 0.0 || config_.wFootIkAnchor > 0.0 ||
                                 config_.wRootXyContact > 0.0 || config_.wContactJointAnchor > 0.0);
        const int nFeet = static_cast<int>(footBodyIds_.size());
        std::vector<std::vector<Eigen::Vector3d>> footPos;
        if (footActive) {
            footPos.assign(nFrames, std::vector<Eigen::Vector3d>(nFeet));
        }

        for (int t = 0; t < nFrames; ++t) {
            mju_copy(data->qpos, qWin[t].data(), model->nq);
            mj_forward(model, data);

            for (const auto& entry : trackEntries_) {
                auto tgtIt = targets[t].find(entry.bodyId);
                if (tgtIt == targets[t].end()) {
                    continue;
                }
                const FrameTaskTarget& tgt = tgtIt->second;
                const double* xpos         = &data->xpos[3 * entry.bodyId];
                if (entry.posWeight > 0.0) {
                    const Eigen::Vector3d e(xpos[0] - tgt.targetPos.x(), xpos[1] - tgt.targetPos.y(), xpos[2] - tgt.targetPos.z());
                    fkCost += entry.posWeight * e.squaredNorm();
                }
                if (entry.rotWeight > 0.0) {
                    const Eigen::Matrix3d Rbody = bodyRotation(data, entry.bodyId);
                    Eigen::Quaterniond qBody(Rbody);
                    qBody.normalize();
                    const Eigen::Vector3d err = quatRotError(tgt.targetRot, qBody);
                    fkCost += entry.rotWeight * err.squaredNorm();
                }
            }

            if (footActive) {
                for (int f = 0; f < nFeet; ++f) {
                    const double* xpos = &data->xpos[3 * footBodyIds_[f]];
                    footPos[t][f]      = Eigen::Vector3d(xpos[0], xpos[1], xpos[2]);
                }
            }
        }

        cost += fkCost;

        if (config_.wVelocity > 0.0 && nFrames >= 2 && !smoothQ_.empty()) {
            for (int t = 1; t < nFrames; ++t) {
                for (int qadr : smoothQ_) {
                    const double e = qWin[t][qadr] - qWin[t - 1][qadr];
                    velCost += config_.wVelocity * e * e;
                }
            }
        }

        if (config_.wAcceleration > 0.0 && nFrames >= 3 && !smoothQ_.empty()) {
            for (int t = 2; t < nFrames; ++t) {
                for (int qadr : smoothQ_) {
                    const double e = qWin[t][qadr] - 2.0 * qWin[t - 1][qadr] + qWin[t - 2][qadr];
                    accCost += config_.wAcceleration * e * e;
                }
            }
        }

        if (anchorWeight > 0.0) {
            const Eigen::VectorXd delta = qWin[0] - anchor;
            anchorCost                  = anchorWeight * delta.squaredNorm();
        }

        cost += velCost + accCost + anchorCost;

        if (!footActive) {
            if (config_.verbose && frameOffset == 0) {
                std::cerr << "[batch-to-cpp] windowCost breakdown fk=" << fkCost << " vel=" << velCost << " acc=" << accCost
                          << " foot=" << footCost << " total=" << cost << "\n";
            }
            return cost;
        }

        const bool useGlobalRefFoot = config_.wFootIkAnchor > 0.0 && !globalRefFootPos_.empty() && !qRef.empty();

        for (int t = 0; t < nFrames; ++t) {
            const int tAbs = t + frameOffset;
            for (int f = 0; f < nFeet; ++f) {
                bool contact = true;
                if (!globalRefContact_.empty()) {
                    if (tAbs >= static_cast<int>(globalRefContact_.size())) {
                        continue;
                    }
                    contact = globalRefContact_[tAbs][f];
                }
                if (!contact) {
                    continue;
                }

                if (config_.wFootHeight > 0.0) {
                    const double dz = footPos[t][f].z() - groundZ_;
                    footCost += config_.wFootHeight * dz * dz;
                }

                if (useGlobalRefFoot && tAbs < static_cast<int>(globalRefFootPos_.size())) {
                    const Eigen::Vector2d dxy = footPos[t][f].head<2>() - globalRefFootPos_[tAbs][f].head<2>();
                    footCost += config_.wFootIkAnchor * dxy.squaredNorm();
                }
            }

            bool anyContact = false;
            if (!globalRefContact_.empty() && tAbs < static_cast<int>(globalRefContact_.size())) {
                anyContact = std::any_of(globalRefContact_[tAbs].begin(), globalRefContact_[tAbs].end(), [](bool v) { return v; });
            } else if (globalRefContact_.empty()) {
                anyContact = true;
            }

            if (config_.wRootXyContact > 0.0 && anyContact && model->nq >= 2 && !qRef.empty()) {
                const Eigen::Vector2d dxy = qWin[t].head<2>() - qRef[t].head<2>();
                footCost += config_.wRootXyContact * dxy.squaredNorm();
            }

            if (config_.wContactJointAnchor > 0.0 && anyContact && !qRef.empty()) {
                for (int qadr : smoothQ_) {
                    const double e = qWin[t][qadr] - qRef[t][qadr];
                    footCost += config_.wContactJointAnchor * e * e;
                }
            }

            if (config_.wFootSlip > 0.0 && t > 0) {
                for (int f = 0; f < nFeet; ++f) {
                    bool both = true;
                    if (!globalRefContact_.empty()) {
                        const int tPrevAbs = t - 1 + frameOffset;
                        if (tAbs >= static_cast<int>(globalRefContact_.size()) || tPrevAbs >= static_cast<int>(globalRefContact_.size())) {
                            both = false;
                        } else {
                            both = globalRefContact_[tAbs][f] && globalRefContact_[tPrevAbs][f];
                        }
                    }
                    if (!both) {
                        continue;
                    }
                    const Eigen::Vector2d dxy = footPos[t][f].head<2>() - footPos[t - 1][f].head<2>();
                    footCost += config_.wFootSlip * dxy.squaredNorm();
                }
            }
        }

        cost += footCost;
        if (config_.verbose && frameOffset == 0) {
            std::cerr << "[batch-to-cpp] windowCost breakdown fk=" << fkCost << " vel=" << velCost << " acc=" << accCost
                      << " foot=" << footCost << " total=" << cost << "\n";
        }

        return cost;
    }

    void BatchTrajectoryRetargeter::buildGlobalRefFootPos(const std::vector<Eigen::VectorXd>& qRef) {
        globalRefFootPos_.clear();
        if (!config_.enableFootPenalties || config_.wFootIkAnchor <= 0.0 || footBodyIds_.empty() || qRef.empty()) {
            return;
        }

        const int nFrames = static_cast<int>(qRef.size());
        const int nFeet   = static_cast<int>(footBodyIds_.size());
        globalRefFootPos_.assign(nFrames, std::vector<Eigen::Vector3d>(nFeet));

        mjModel* model = impl_->model.get();
        mjData* data   = impl_->data.get();
        for (int t = 0; t < nFrames; ++t) {
            mju_copy(data->qpos, qRef[t].data(), model->nq);
            mj_forward(model, data);
            for (int f = 0; f < nFeet; ++f) {
                const double* xpos      = &data->xpos[3 * footBodyIds_[f]];
                globalRefFootPos_[t][f] = Eigen::Vector3d(xpos[0], xpos[1], xpos[2]);
            }
        }
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::bootstrapQ(const std::vector<HumanFrame>& humanFrames, Retargeter& retargeter,
                                                                       bool offsetToGround, const BatchIkBootstrapContext* ikBootstrap) {
        if (!config_.qInitJsonPath.empty()) {
            return loadQInitFromJson(config_.qInitJsonPath, humanFrames.size());
        }

        std::vector<Eigen::VectorXd> qInit;
        const int n = static_cast<int>(humanFrames.size());
        qInit.resize(n);
        if (!config_.useGmrInit) {
            const Eigen::VectorXd q0 = retargeter.currentQpos();
            for (int i = 0; i < n; ++i) {
                qInit[i] = q0;
            }
            return qInit;
        }

#if defined(_OPENMP)
        const bool canParallel = config_.parallelBootstrap && ikBootstrap != nullptr && n > 1;
        if (canParallel) {
            const int nThreads = config_.parallelThreads > 0 ? config_.parallelThreads : omp_get_max_threads();
            std::vector<std::unique_ptr<Retargeter>> workers(static_cast<std::size_t>(nThreads));
            for (int t = 0; t < nThreads; ++t) {
                workers[static_cast<std::size_t>(t)] =
                    createRetargeter(ikBootstrap->backend, robotModelPath_, ikConfig_, ikBootstrap->options);
            }
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int i = 0; i < n; ++i) {
                const int tid = omp_get_thread_num();
                qInit[static_cast<std::size_t>(i)] =
                    workers[static_cast<std::size_t>(tid)]->retargetFrame(humanFrames[static_cast<std::size_t>(i)], offsetToGround);
            }
            return qInit;
        }
#endif

        for (int i = 0; i < n; ++i) {
            qInit[i] = retargeter.retargetFrame(humanFrames[i], offsetToGround);
        }
        return qInit;
    }

    std::vector<int> BatchTrajectoryRetargeter::windowStarts(int nFrames) const {
        const int H = config_.windowSize;
        const int S = config_.windowStride;
        if (nFrames <= H) {
            return {0};
        }
        std::vector<int> starts;
        for (int s = 0; s < nFrames; s += S) {
            starts.push_back(s);
        }
        const int last = nFrames - H;
        if (starts.back() != last) {
            starts.push_back(last);
        }
        return starts;
    }

    std::vector<std::vector<bool>> BatchTrajectoryRetargeter::batchContactMask(const std::vector<Eigen::VectorXd>& qRef) const {
        const int nFeet   = static_cast<int>(footBodyIds_.size());
        const int nFrames = static_cast<int>(qRef.size());
        std::vector<std::vector<bool>> contact(nFrames, std::vector<bool>(nFeet, false));
        if (nFeet == 0 || nFrames == 0) {
            return contact;
        }

        std::vector<double> zMin(nFeet, std::numeric_limits<double>::infinity());
        std::vector<std::vector<double>> footZ(nFrames, std::vector<double>(nFeet, 0.0));

        mjModel* model = impl_->model.get();
        mjData* data   = impl_->data.get();
        for (int t = 0; t < nFrames; ++t) {
            mju_copy(data->qpos, qRef[t].data(), model->nq);
            mj_forward(model, data);
            for (int f = 0; f < nFeet; ++f) {
                const double z = data->xpos[3 * footBodyIds_[f] + 2];
                footZ[t][f]    = z;
                zMin[f]        = std::min(zMin[f], z);
            }
        }

        for (int t = 0; t < nFrames; ++t) {
            for (int f = 0; f < nFeet; ++f) {
                contact[t][f] = footZ[t][f] <= zMin[f] + config_.footContactMargin;
            }
        }
        return contact;
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::optimizeSlidingWindows(const std::vector<Eigen::VectorXd>& qInit,
                                                                                   const std::vector<FrameTargets>& targets) {
        const int n = static_cast<int>(qInit.size());
        const int H = config_.windowSize;

        if (n <= H) {
            return optimizeGnWindow(qInit, targets, qInit.front(), qInit, 0, config_.wAnchor);
        }

        std::vector<Eigen::VectorXd> qOut = qInit;
        const std::vector<int> starts     = windowStarts(n);
        for (std::size_t wi = 0; wi < starts.size(); ++wi) {
            const int start = starts[wi];
            const int end   = std::min(start + H, n);
            if (end - start < 2) {
                continue;
            }

            std::vector<Eigen::VectorXd> qWin;
            std::vector<FrameTargets> tgtWin;
            std::vector<Eigen::VectorXd> refWin;
            qWin.reserve(end - start);
            tgtWin.reserve(end - start);
            refWin.reserve(end - start);
            for (int i = start; i < end; ++i) {
                qWin.push_back(qOut[i]);
                tgtWin.push_back(targets[i]);
                refWin.push_back(qInit[i]);
            }

            double anchorW = config_.wAnchor;
            if (start > 0) {
                anchorW = std::max(anchorW, config_.windowAnchorWeight);
            }

            const std::vector<Eigen::VectorXd> qOpt = optimizeGnWindow(qWin, tgtWin, qOut[start], refWin, start, anchorW);

            int commitEnd = start + config_.windowStride;
            if (wi == 0) {
                commitEnd = std::min(start + config_.windowStride, n);
            } else if (end >= n) {
                commitEnd = n;
            }

            for (int i = start; i < commitEnd; ++i) {
                qOut[i] = qOpt[i - start];
            }
        }
        return qOut;
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::optimizeGnWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                                             const std::vector<FrameTargets>& targets,
                                                                             const Eigen::VectorXd& anchor,
                                                                             const std::vector<Eigen::VectorXd>& qRef, int frameOffset,
                                                                             double anchorWeight, const QpWindowOptions* qpOpts) {
        std::vector<Eigen::VectorXd> qWin = qInit;
        const int nFrames                 = static_cast<int>(qWin.size());
        const int m                       = static_cast<int>(optVidx_.size());
        const int nvar                    = nFrames * m;

        ensureGnWorkspace(nFrames);
        GnWorkspace& ws = *gnWs_;

        mjModel* model = impl_->model.get();
        mjData* data   = impl_->data.get();

        const bool footActive = config_.enableFootPenalties && !footBodyIds_.empty() &&
                                (config_.wFootHeight > 0.0 || config_.wFootSlip > 0.0 || config_.wFootIkAnchor > 0.0 ||
                                 config_.wRootXyContact > 0.0 || config_.wContactJointAnchor > 0.0);
        const int nFeet             = static_cast<int>(footBodyIds_.size());
        const bool useGlobalRefFoot = config_.wFootIkAnchor > 0.0 && !globalRefFootPos_.empty() && !qRef.empty();

        const bool logCost = config_.verbose;
        double costBefore  = 0.0;
        if (logCost) {
            costBefore = windowCost(qWin, targets, anchor, qRef, frameOffset, anchorWeight);
        }

        for (int step = 0; step < config_.gnSteps; ++step) {
            ws.Hdense.setZero();
            ws.g.setZero();

            for (int t = 0; t < nFrames; ++t) {
                mju_copy(data->qpos, qWin[t].data(), model->nq);
                mj_forward(model, data);
                const int off = t * m;
                const int nv  = model->nv;

                for (const auto& entry : trackEntries_) {
                    auto tgtIt = targets[t].find(entry.bodyId);
                    if (tgtIt == targets[t].end()) {
                        continue;
                    }
                    const FrameTaskTarget& tgt = tgtIt->second;
                    const double* xpos         = &data->xpos[3 * entry.bodyId];
                    Eigen::Vector3d bodyPos(xpos[0], xpos[1], xpos[2]);

                    if (entry.posWeight > 0.0) {
                        const Eigen::Vector3d err = bodyPos - tgt.targetPos;
                        mj_jac(model, data, ws.jacp.data(), nullptr, xpos, entry.bodyId);
                        Eigen::MatrixXd J(3, m);
                        for (int col = 0; col < m; ++col) {
                            J.col(col) = mjJacColumn(ws.jacp.data(), nv, optVidx_[col]);
                        }
                        const double w = entry.posWeight;
                        ws.Hdense.block(off, off, m, m).noalias() += w * J.transpose() * J;
                        ws.g.segment(off, m).noalias() += w * J.transpose() * err;
                    }

                    if (entry.rotWeight > 0.0) {
                        const Eigen::Matrix3d Rbody = bodyRotation(data, entry.bodyId);
                        Eigen::Quaterniond qBody(Rbody);
                        qBody.normalize();
                        const Eigen::Vector3d err = quatRotError(tgt.targetRot, qBody);
                        mj_jac(model, data, nullptr, ws.jacr.data(), xpos, entry.bodyId);
                        Eigen::MatrixXd J(3, m);
                        for (int col = 0; col < m; ++col) {
                            J.col(col) = mjJacColumn(ws.jacr.data(), nv, optVidx_[col]);
                        }
                        const double w = entry.rotWeight;
                        ws.Hdense.block(off, off, m, m).noalias() += w * J.transpose() * J;
                        ws.g.segment(off, m).noalias() += w * J.transpose() * err;
                    }
                }

                if (footActive) {
                    for (int f = 0; f < nFeet; ++f) {
                        const int bid      = footBodyIds_[f];
                        const double* fpos = &data->xpos[3 * bid];
                        ws.footPos[t][f]   = Eigen::Vector3d(fpos[0], fpos[1], fpos[2]);
                        mj_jac(model, data, ws.jacp.data(), nullptr, fpos, bid);
                        Eigen::MatrixXd Jxy(2, m);
                        Eigen::RowVectorXd Jz(m);
                        for (int col = 0; col < m; ++col) {
                            const Eigen::Vector3d jcol = mjJacColumn(ws.jacp.data(), nv, optVidx_[col]);
                            Jxy(0, col)                = jcol.x();
                            Jxy(1, col)                = jcol.y();
                            Jz(col)                    = jcol.z();
                        }
                        ws.footJxy[t][f] = Jxy;
                        ws.footJz[t][f]  = Jz;
                    }
                }
            }

            if (anchorWeight > 0.0) {
                for (int vi = 0; vi < m; ++vi) {
                    const int v      = optVidx_[vi];
                    const int qadr   = model->jnt_qposadr[model->dof_jntid[v]];
                    const double err = qWin[0][qadr] - anchor[qadr];
                    ws.Hdense(vi, vi) += anchorWeight;
                    ws.g[vi] += anchorWeight * err;
                }
            }

            if (config_.wVelocity > 0.0 && nFrames >= 2 && !smoothV_.empty()) {
                for (int t = 1; t < nFrames; ++t) {
                    const int offT  = t * m;
                    const int offM1 = (t - 1) * m;
                    for (std::size_t k = 0; k < smoothV_.size(); ++k) {
                        const int vi   = smoothV_[k];
                        const int qadr = smoothQ_[k];
                        const double e = qWin[t][qadr] - qWin[t - 1][qadr];
                        const double w = config_.wVelocity;
                        ws.Hdense(offT + vi, offT + vi) += w;
                        ws.Hdense(offM1 + vi, offM1 + vi) += w;
                        ws.Hdense(offT + vi, offM1 + vi) -= w;
                        ws.Hdense(offM1 + vi, offT + vi) -= w;
                        ws.g[offT + vi] += w * e;
                        ws.g[offM1 + vi] -= w * e;
                    }
                }
            }

            if (config_.wAcceleration > 0.0 && nFrames >= 3 && !smoothV_.empty()) {
                for (int t = 2; t < nFrames; ++t) {
                    for (std::size_t k = 0; k < smoothV_.size(); ++k) {
                        const int vi   = smoothV_[k];
                        const int qadr = smoothQ_[k];
                        const double e = qWin[t][qadr] - 2.0 * qWin[t - 1][qadr] + qWin[t - 2][qadr];
                        const int i0   = (t - 2) * m + vi;
                        const int i1   = (t - 1) * m + vi;
                        const int i2   = t * m + vi;
                        const double w = config_.wAcceleration;
                        ws.Hdense(i2, i2) += w;
                        ws.Hdense(i1, i1) += 4.0 * w;
                        ws.Hdense(i0, i0) += w;
                        ws.Hdense(i2, i1) -= 2.0 * w;
                        ws.Hdense(i1, i2) -= 2.0 * w;
                        ws.Hdense(i2, i0) += w;
                        ws.Hdense(i0, i2) += w;
                        ws.Hdense(i1, i0) -= 2.0 * w;
                        ws.Hdense(i0, i1) -= 2.0 * w;
                        ws.g[i2] += w * e;
                        ws.g[i1] -= 2.0 * w * e;
                        ws.g[i0] += w * e;
                    }
                }
            }

            if (footActive) {
                for (int t = 0; t < nFrames; ++t) {
                    const int offT = t * m;
                    const int tAbs = t + frameOffset;

                    for (int f = 0; f < nFeet; ++f) {
                        bool contact = true;
                        if (!globalRefContact_.empty()) {
                            if (tAbs >= static_cast<int>(globalRefContact_.size())) {
                                continue;
                            }
                            contact = globalRefContact_[tAbs][f];
                        }
                        if (!contact) {
                            continue;
                        }

                        if (config_.wFootHeight > 0.0) {
                            const double err             = ws.footPos[t][f].z() - groundZ_;
                            const Eigen::RowVectorXd& Jz = ws.footJz[t][f];
                            const double w               = config_.wFootHeight;
                            ws.Hdense.block(offT, offT, m, m).noalias() += w * Jz.transpose() * Jz;
                            ws.g.segment(offT, m).noalias() += w * Jz.transpose() * err;
                        }

                        if (useGlobalRefFoot && tAbs < static_cast<int>(globalRefFootPos_.size())) {
                            const Eigen::Vector2d err  = ws.footPos[t][f].head<2>() - globalRefFootPos_[tAbs][f].head<2>();
                            const Eigen::MatrixXd& Jxy = ws.footJxy[t][f];
                            const double w             = config_.wFootIkAnchor;
                            ws.Hdense.block(offT, offT, m, m).noalias() += w * Jxy.transpose() * Jxy;
                            ws.g.segment(offT, m).noalias() += w * Jxy.transpose() * err;
                        }
                    }

                    bool anyContact = false;
                    if (!globalRefContact_.empty() && tAbs < static_cast<int>(globalRefContact_.size())) {
                        anyContact = std::any_of(globalRefContact_[tAbs].begin(), globalRefContact_[tAbs].end(), [](bool v) { return v; });
                    } else if (globalRefContact_.empty()) {
                        anyContact = true;
                    }

                    if (config_.wRootXyContact > 0.0 && anyContact && model->nq >= 2) {
                        for (int qadr = 0; qadr < 2; ++qadr) {
                            auto it = qToOptV_.find(qadr);
                            if (it == qToOptV_.end()) {
                                continue;
                            }
                            const int vi     = it->second;
                            const double err = qWin[t][qadr] - qRef[t][qadr];
                            const double w   = config_.wRootXyContact;
                            ws.Hdense(offT + vi, offT + vi) += w;
                            ws.g[offT + vi] += w * err;
                        }
                    }

                    if (config_.wContactJointAnchor > 0.0 && anyContact) {
                        for (std::size_t k = 0; k < smoothV_.size(); ++k) {
                            const int vi     = smoothV_[k];
                            const int qadr   = smoothQ_[k];
                            const double err = qWin[t][qadr] - qRef[t][qadr];
                            const double w   = config_.wContactJointAnchor;
                            ws.Hdense(offT + vi, offT + vi) += w;
                            ws.g[offT + vi] += w * err;
                        }
                    }

                    if (config_.wFootSlip > 0.0 && t > 0) {
                        for (int f = 0; f < nFeet; ++f) {
                            bool both = true;
                            if (!globalRefContact_.empty()) {
                                const int tPrevAbs = t - 1 + frameOffset;
                                if (tAbs >= static_cast<int>(globalRefContact_.size()) ||
                                    tPrevAbs >= static_cast<int>(globalRefContact_.size())) {
                                    both = false;
                                } else {
                                    both = globalRefContact_[tAbs][f] && globalRefContact_[tPrevAbs][f];
                                }
                            }
                            if (!both) {
                                continue;
                            }
                            const Eigen::Vector2d err = ws.footPos[t][f].head<2>() - ws.footPos[t - 1][f].head<2>();
                            const int offPrev         = (t - 1) * m;
                            const Eigen::MatrixXd& Jt = ws.footJxy[t][f];
                            const Eigen::MatrixXd& Jp = ws.footJxy[t - 1][f];
                            const double w            = config_.wFootSlip;
                            ws.Hdense.block(offT, offT, m, m).noalias() += w * Jt.transpose() * Jt;
                            ws.Hdense.block(offPrev, offPrev, m, m).noalias() += w * Jp.transpose() * Jp;
                            ws.Hdense.block(offT, offPrev, m, m).noalias() -= w * Jt.transpose() * Jp;
                            ws.Hdense.block(offPrev, offT, m, m).noalias() -= w * Jp.transpose() * Jt;
                            ws.g.segment(offT, m).noalias() += w * Jt.transpose() * err;
                            ws.g.segment(offPrev, m).noalias() -= w * Jp.transpose() * err;
                        }
                    }
                }
            }

            if (qpOpts != nullptr && qpOpts->wGmr > 0.0) {
                for (int t = 0; t < nFrames; ++t) {
                    const int off = t * m;
                    for (int vi = 0; vi < m; ++vi) {
                        const int v    = optVidx_[vi];
                        const int j    = model->dof_jntid[v];
                        const int qadr = model->jnt_qposadr[j];
                        if (model->jnt_type[j] == mjJNT_FREE && qadr >= 3) {
                            continue;
                        }
                        const double err = qWin[t][qadr] - qRef[t][qadr];
                        ws.Hdense(off + vi, off + vi) += qpOpts->wGmr;
                        ws.g[off + vi] += qpOpts->wGmr * err;
                    }
                }
            }

            Eigen::MatrixXd Hreg = ws.Hdense + config_.gnDamping * Eigen::MatrixXd::Identity(nvar, nvar);
            if (qpOpts != nullptr) {
                Hreg = 0.5 * (Hreg + Hreg.transpose()).eval();

                Eigen::VectorXd lb = Eigen::VectorXd::Constant(nvar, -config_.gnMaxStep);
                Eigen::VectorXd ub = Eigen::VectorXd::Constant(nvar, config_.gnMaxStep);

                struct HingePair {
                    int localV = 0;
                    int qadr   = 0;
                };
                std::vector<HingePair> hingePairs;
                hingePairs.reserve(static_cast<std::size_t>(m));
                for (int vi = 0; vi < m; ++vi) {
                    const int v     = optVidx_[vi];
                    const int j     = model->dof_jntid[v];
                    const int jtype = model->jnt_type[j];
                    if (jtype == mjJNT_HINGE || jtype == mjJNT_SLIDE) {
                        hingePairs.push_back(HingePair{vi, model->jnt_qposadr[j]});
                    }
                }

                if (qpOpts->useJointLimits) {
                    constexpr double kDeg2Rad = 0.017453292519943295;
                    const double marginRad    = std::max(0.0, qpOpts->jointLimitMarginDeg) * kDeg2Rad;
                    for (int t = 0; t < nFrames; ++t) {
                        const int off = t * m;
                        for (const auto& hp : hingePairs) {
                            const int j = model->dof_jntid[optVidx_[hp.localV]];
                            if (model->jnt_limited[j] <= 0) {
                                continue;
                            }
                            double qmin = model->jnt_range[2 * j + 0];
                            double qmax = model->jnt_range[2 * j + 1];
                            // Keep revolute joints a safety margin away from the hard limits so the
                            // commanded trajectory stays inside the trackable range.
                            if (marginRad > 0.0 && model->jnt_type[j] == mjJNT_HINGE &&
                                (qmax - qmin) > 2.0 * marginRad) {
                                qmin += marginRad;
                                qmax -= marginRad;
                            }
                            const double q0     = qWin[t][hp.qadr];
                            lb[off + hp.localV] = std::max(lb[off + hp.localV], qmin - q0);
                            ub[off + hp.localV] = std::min(ub[off + hp.localV], qmax - q0);
                        }
                    }
                }

                const int pinFrames = std::max(0, std::min(qpOpts->pinFrames, nFrames - 1));
                if (pinFrames > 0) {
                    lb.head(pinFrames * m).setZero();
                    ub.head(pinFrames * m).setZero();
                }

                std::vector<Eigen::VectorXd> gRows;
                std::vector<double> hVals;
                const double dt    = std::max(qpOpts->motionDt, 1e-6);
                const double dqLim = qpOpts->dqMax * dt;
                if (qpOpts->useVelocityLimits && !hingePairs.empty()) {
                    for (int t = 0; t < nFrames; ++t) {
                        const int off = t * m;
                        for (const auto& hp : hingePairs) {
                            if (t == 0) {
                                if (qpOpts->qPrev == nullptr) {
                                    continue;
                                }
                                const double base     = qWin[0][hp.qadr] - (*qpOpts->qPrev)[hp.qadr];
                                Eigen::VectorXd rowP  = Eigen::VectorXd::Zero(nvar);
                                Eigen::VectorXd rowM  = Eigen::VectorXd::Zero(nvar);
                                rowP[off + hp.localV] = 1.0;
                                rowM[off + hp.localV] = -1.0;
                                gRows.push_back(rowP);
                                hVals.push_back(dqLim - base);
                                gRows.push_back(rowM);
                                hVals.push_back(dqLim + base);
                            } else {
                                const int offM         = (t - 1) * m;
                                const double base      = qWin[t][hp.qadr] - qWin[t - 1][hp.qadr];
                                Eigen::VectorXd rowP   = Eigen::VectorXd::Zero(nvar);
                                Eigen::VectorXd rowM   = Eigen::VectorXd::Zero(nvar);
                                rowP[off + hp.localV]  = 1.0;
                                rowP[offM + hp.localV] = -1.0;
                                rowM[off + hp.localV]  = -1.0;
                                rowM[offM + hp.localV] = 1.0;
                                gRows.push_back(rowP);
                                hVals.push_back(dqLim - base);
                                gRows.push_back(rowM);
                                hVals.push_back(dqLim + base);
                            }
                        }
                    }
                }

                const int nBox  = nvar;
                const int nVel  = static_cast<int>(gRows.size());
                const int nIneq = nBox + nVel;
                gmr::solver::QPData qp;
                qp.reset(nvar, nIneq);
                qp.H = Hreg;
                qp.g = ws.g;
                for (int i = 0; i < nvar; ++i) {
                    qp.CI(i, i) = 1.0;
                    qp.ciLb[i]  = lb[i];
                    qp.ciUb[i]  = ub[i];
                }
                for (int r = 0; r < nVel; ++r) {
                    qp.CI.row(nBox + r) = gRows[static_cast<std::size_t>(r)].transpose();
                    qp.ciLb[nBox + r]   = -1e20;
                    qp.ciUb[nBox + r]   = hVals[static_cast<std::size_t>(r)];
                }

                bool qpOk = false;
                try {
                    gmr::solver::QPSolver solver(qpOpts->qpBackend);
                    const gmr::solver::QPOutput& out = solver.solve(qp);
                    if (out.status == gmr::solver::QPStatus::kOptimal || out.status == gmr::solver::QPStatus::kMaxIterReached) {
                        ws.dqFlat = out.x;
                        qpOk      = true;
                    }
                } catch (...) {
                    qpOk = false;
                }
                if (!qpOk) {
                    ws.dqFlat = Hreg.ldlt().solve(ws.g);
                }
                if (pinFrames > 0) {
                    ws.dqFlat.head(pinFrames * m).setZero();
                }
            } else if (config_.useBandedSolver) {
                ws.Hband.setZero();
                const int bw = ws.Hband.bandwidth();
                for (int i = 0; i < nvar; ++i) {
                    for (int j = std::max(0, i - bw); j <= i; ++j) {
                        ws.Hband.add(i, j, ws.Hdense(i, j));
                    }
                }
                ws.Hband.solve(ws.g, config_.gnDamping, ws.dqFlat);
            } else {
                ws.dqFlat = Hreg.ldlt().solve(ws.g);
            }
            const bool solved = true;
            (void)solved;

            ws.dqFlat = ws.dqFlat.cwiseMax(-config_.gnMaxStep).cwiseMin(config_.gnMaxStep);

            const std::vector<double>& alphas = config_.gnLineSearchAlphas.empty() ? std::vector<double>{1.0} : config_.gnLineSearchAlphas;

            if (alphas.size() == 1) {
                applyGnStepToWindow(qWin, ws.dqFlat, alphas.front());
            } else if (config_.gnLineSearchMode == GnLineSearchMode::kArmijo) {
                const double cost0 = windowCost(qWin, targets, anchor, qRef, frameOffset, anchorWeight);
                bool improved      = false;
                for (double alpha : alphas) {
                    std::vector<Eigen::VectorXd> trial = qWin;
                    applyGnStepToWindow(trial, ws.dqFlat, alpha);
                    const double trialCost = windowCost(trial, targets, anchor, qRef, frameOffset, anchorWeight);
                    if (trialCost < cost0) {
                        qWin     = trial;
                        improved = true;
                        break;
                    }
                }
                if (!improved) {
                    applyGnStepToWindow(qWin, ws.dqFlat, alphas.front());
                }
            } else {
                std::vector<Eigen::VectorXd> bestQ = qWin;
                double bestCost                    = windowCost(qWin, targets, anchor, qRef, frameOffset, anchorWeight);
                for (double alpha : alphas) {
                    std::vector<Eigen::VectorXd> trial = qWin;
                    applyGnStepToWindow(trial, ws.dqFlat, alpha);
                    const double trialCost = windowCost(trial, targets, anchor, qRef, frameOffset, anchorWeight);
                    if (trialCost < bestCost) {
                        bestCost = trialCost;
                        bestQ    = trial;
                    }
                }
                qWin = bestQ;
            }
        }

        if (logCost) {
            const double costAfter = windowCost(qWin, targets, anchor, qRef, frameOffset, anchorWeight);
            std::cerr << "[batch-to-cpp] GN window offset=" << frameOffset << " frames=" << nFrames << " cost " << costBefore << " -> "
                      << costAfter << " track=" << trackEntries_.size() << " targets0=" << (targets.empty() ? 0 : targets.front().size())
                      << " smoothV=" << smoothV_.size() << " smoothQ=" << smoothQ_.size() << " footBodies=" << footBodyIds_.size()
                      << " refContact=" << globalRefContact_.size() << "\n";
        }

        return qWin;
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::optimizeQpWindow(const std::vector<Eigen::VectorXd>& qInit,
                                                                             const std::vector<FrameTargets>& targets,
                                                                             const Eigen::VectorXd& anchor,
                                                                             const std::vector<Eigen::VectorXd>& qRef, int frameOffset,
                                                                             double anchorWeight, const QpWindowOptions& qpOpts) {
        return optimizeGnWindow(qInit, targets, anchor, qRef, frameOffset, anchorWeight, &qpOpts);
    }

    void BatchTrajectoryRetargeter::clearFootContactSchedule() {
        globalRefContact_.clear();
        globalRefFootPos_.clear();
    }

    void BatchTrajectoryRetargeter::setFootContactFromQRef(const std::vector<Eigen::VectorXd>& qRef) {
        if (qRef.empty()) {
            clearFootContactSchedule();
            return;
        }
        globalRefContact_ = batchContactMask(qRef);
        buildGlobalRefFootPos(qRef);
    }

    Eigen::VectorXd BatchTrajectoryRetargeter::finalizeQpos(const Eigen::VectorXd& qpos, Retargeter& retargeter, const HumanFrame& prepared,
                                                            bool offsetToGround) {
        (void)prepared;
        (void)offsetToGround;
        retargeter.setQpos(qpos);
        retargeter.finalizeContact();
        return retargeter.currentQpos();
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::retargetBatch(const std::vector<HumanFrame>& humanFrames,
                                                                          Retargeter& retargeter, bool offsetToGround,
                                                                          const BatchIkBootstrapContext* ikBootstrap) {
        lastProfile_ = {};
        if (humanFrames.empty()) {
            return {};
        }

        if (ikBootstrap != nullptr) {
            applyContactGroundConfig(ikBootstrap->contactGround);
        }

        double motionFps = 30.0;
        if (ikBootstrap != nullptr && ikBootstrap->options.motionFps > 0.0) {
            motionFps = ikBootstrap->options.motionFps;
        }
        if (contactGroundPipeline_ != nullptr) {
            contactGroundPipeline_->setFps(motionFps);
        }

        const auto tTotal = Clock::now();
        std::vector<HumanFrame> prepared;
        std::vector<FrameTargets> targets;
        prepared.reserve(humanFrames.size());
        targets.reserve(humanFrames.size());

        auto t0 = Clock::now();
        for (const auto& frame : humanFrames) {
            // Match Python batch: same contact-ground state machine as bootstrap IK (one shared pipeline).
            HumanFrame prep = retargeter.prepareRetargetInput(frame, offsetToGround);
            prepared.push_back(prep);
            targets.push_back(targetsForPrepared(prep));
        }
        lastProfile_.prepareMs = elapsedMs(t0);

        t0                                 = Clock::now();
        std::vector<Eigen::VectorXd> qInit = bootstrapQ(humanFrames, retargeter, offsetToGround, ikBootstrap);
        lastProfile_.bootstrapMs           = elapsedMs(t0);

        if (config_.enableFootPenalties && !footBodyIds_.empty()) {
            globalRefContact_ = batchContactMask(qInit);
        } else {
            globalRefContact_.clear();
        }
        buildGlobalRefFootPos(qInit);

        t0                                = Clock::now();
        std::vector<Eigen::VectorXd> qOpt = optimizeSlidingWindows(qInit, targets);
        lastProfile_.optimizeMs           = elapsedMs(t0);

        t0 = Clock::now();
        std::vector<Eigen::VectorXd> qOut;
        qOut.resize(qOpt.size());

#if defined(_OPENMP)
        const bool canParallelFinalize = config_.parallelFinalize && ikBootstrap != nullptr && static_cast<int>(qOpt.size()) > 1;
        if (canParallelFinalize) {
            const int n        = static_cast<int>(qOpt.size());
            const int nThreads = config_.parallelThreads > 0 ? config_.parallelThreads : omp_get_max_threads();
            std::vector<std::unique_ptr<Retargeter>> workers(static_cast<std::size_t>(nThreads));
            for (int t = 0; t < nThreads; ++t) {
                workers[static_cast<std::size_t>(t)] =
                    createRetargeter(ikBootstrap->backend, robotModelPath_, ikConfig_, ikBootstrap->options);
            }
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int i = 0; i < n; ++i) {
                const int tid = omp_get_thread_num();
                if (config_.finalizeContact) {
                    workers[static_cast<std::size_t>(tid)]->setQpos(qOpt[static_cast<std::size_t>(i)]);
                    workers[static_cast<std::size_t>(tid)]->finalizeContact();
                    qOut[static_cast<std::size_t>(i)] = workers[static_cast<std::size_t>(tid)]->currentQpos();
                } else {
                    qOut[static_cast<std::size_t>(i)] = qOpt[static_cast<std::size_t>(i)];
                }
            }
        } else
#endif
        {
            if (config_.finalizeContact) {
                for (std::size_t i = 0; i < qOpt.size(); ++i) {
                    qOut[i] = finalizeQpos(qOpt[i], retargeter, prepared[i], offsetToGround);
                }
            } else {
                qOut = std::move(qOpt);
            }
        }
        lastProfile_.finalizeMs = elapsedMs(t0);

        // Keep committed hinge joints a safety margin off the hard limits (Python _apply_margin_clip).
        // Applied once on the final pose so it does not disturb the GN optimizer.
        if (config_.jointLimitMarginDeg > 0.0) {
            for (auto& q : qOut) {
                clipHingeQposMargin(q, config_.jointLimitMarginDeg);
            }
        }

        lastProfile_.nFrames = static_cast<int>(humanFrames.size());
        lastProfile_.totalMs = elapsedMs(tTotal);
        return qOut;
    }

}  // namespace gmr
