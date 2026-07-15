#include "gmr/retarget/batch_trajectory_retarget.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include <mujoco/mujoco.h>

#include "retargeter_internal_utils.h"

namespace gmr {
    namespace {

        using Clock = std::chrono::steady_clock;

        double elapsedMs(const Clock::time_point& t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }

        Eigen::Vector3d quatRotError(const Eigen::Quaterniond& target, const Eigen::Quaterniond& current) {
            Eigen::Quaterniond q = target.conjugate() * current;
            if (q.w() < 0.0) {
                q.coeffs() *= -1.0;
            }
            const double n = q.vec().norm();
            if (n < 1e-12) {
                return Eigen::Vector3d::Zero();
            }
            return 2.0 * std::atan2(n, q.w()) * q.vec() / n;
        }

        Eigen::Matrix3d bodyRotation(const mjData* data, int bodyId) {
            const double* xmat = &data->xmat[9 * bodyId];
            Eigen::Matrix3d R;
            R << xmat[0], xmat[1], xmat[2], xmat[3], xmat[4], xmat[5], xmat[6], xmat[7], xmat[8];
            return R;
        }

    }  // namespace

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
        : config_(std::move(config)), ikConfig_(std::move(ikConfig)), impl_(std::make_unique<Impl>()) {
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
            smoothQidx_.erase(std::remove_if(smoothQidx_.begin(), smoothQidx_.end(), [](int q) { return q < 3; }),
                              smoothQidx_.end());
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

    HumanFrame BatchTrajectoryRetargeter::prepareHumanFrame(const HumanFrame& frame, bool offsetToGround) const {
        return retarget_internal::scaleAndOffsetHumanFrameImpl(frame, ikConfig_, table1PosOffsets_, table1RotOffsets_,
                                                               offsetToGround);
    }

    std::vector<BatchTrajectoryRetargeter::FrameTaskTarget> BatchTrajectoryRetargeter::targetsForPrepared(
        const HumanFrame& prepared) const {
        std::vector<FrameTaskTarget> out;
        out.reserve(trackEntries_.size());

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
                FrameTaskTarget tgt;
                tgt.bodyId     = bodyId;
                tgt.posWeight  = task.posWeight;
                tgt.rotWeight  = task.rotWeight;
                tgt.targetPos  = bodyIt->second.position;
                tgt.targetRot  = bodyIt->second.orientation;
                tgt.targetRot.normalize();
                out.push_back(tgt);
            }
        };

        if (ikConfig_.useTable1) {
            fillFromTasks(ikConfig_.tasksTable1);
        }
        if (ikConfig_.useTable2) {
            fillFromTasks(ikConfig_.tasksTable2);
        }

        std::sort(out.begin(), out.end(), [](const FrameTaskTarget& a, const FrameTaskTarget& b) {
            if (a.bodyId != b.bodyId) {
                return a.bodyId < b.bodyId;
            }
            return a.posWeight + a.rotWeight > b.posWeight + b.rotWeight;
        });
        out.erase(std::unique(out.begin(), out.end(),
                              [](const FrameTaskTarget& a, const FrameTaskTarget& b) { return a.bodyId == b.bodyId; }),
                  out.end());
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

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::bootstrapQ(const std::vector<HumanFrame>& humanFrames,
                                                                       Retargeter& retargeter, bool offsetToGround) {
        std::vector<Eigen::VectorXd> qInit;
        qInit.reserve(humanFrames.size());
        if (!config_.useGmrInit) {
            const Eigen::VectorXd q0 = retargeter.currentQpos();
            for (std::size_t i = 0; i < humanFrames.size(); ++i) {
                (void)i;
                qInit.push_back(q0);
            }
            return qInit;
        }

        for (const auto& frame : humanFrames) {
            qInit.push_back(retargeter.retargetFrame(frame, offsetToGround));
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
        const int nFeet = static_cast<int>(footBodyIds_.size());
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

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::optimizeSlidingWindows(
        const std::vector<Eigen::VectorXd>& qInit, const std::vector<std::vector<FrameTaskTarget>>& targets) {
        const int n = static_cast<int>(qInit.size());
        const int H = config_.windowSize;
        const int S = config_.windowStride;

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
            std::vector<std::vector<FrameTaskTarget>> tgtWin;
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

            const std::vector<Eigen::VectorXd> qOpt =
                optimizeGnWindow(qWin, tgtWin, qOut[start], refWin, start, anchorW);

            int commitEnd = start + S;
            if (wi == 0) {
                commitEnd = std::min(start + S, n);
            } else if (end >= n) {
                commitEnd = n;
            } else {
                commitEnd = start + S;
            }

            for (int i = start; i < commitEnd; ++i) {
                qOut[i] = qOpt[i - start];
            }
        }
        return qOut;
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::optimizeGnWindow(
        const std::vector<Eigen::VectorXd>& qInit, const std::vector<std::vector<FrameTaskTarget>>& targets,
        const Eigen::VectorXd& anchor, const std::vector<Eigen::VectorXd>& qRef, int frameOffset, double anchorWeight) {
        (void)frameOffset;
        std::vector<Eigen::VectorXd> qWin = qInit;
        const int nFrames                 = static_cast<int>(qWin.size());
        const int m                       = static_cast<int>(optVidx_.size());
        const int nvar                    = nFrames * m;

        std::vector<int> smoothV;
        std::vector<int> smoothQ;
        smoothV.reserve(smoothQidx_.size());
        smoothQ.reserve(smoothQidx_.size());
        std::unordered_map<int, int> qToOptV;
        for (int i = 0; i < m; ++i) {
            const int v = optVidx_[i];
            const int j = impl_->model->dof_jntid[v];
            qToOptV[impl_->model->jnt_qposadr[j]] = i;
            auto it = std::find(smoothQidx_.begin(), smoothQidx_.end(), impl_->model->jnt_qposadr[j]);
            if (it != smoothQidx_.end()) {
                smoothQ.push_back(*it);
                smoothV.push_back(i);
            }
        }

        mjModel* model = impl_->model.get();
        mjData* data   = impl_->data.get();
        std::vector<double> jacp(3 * model->nv, 0.0);
        std::vector<double> jacr(3 * model->nv, 0.0);
        std::vector<double> dq(model->nv, 0.0);

        const bool footActive = config_.enableFootPenalties && !footBodyIds_.empty() &&
                                (config_.wFootHeight > 0.0 || config_.wFootSlip > 0.0 || config_.wFootIkAnchor > 0.0 ||
                                 config_.wRootXyContact > 0.0 || config_.wContactJointAnchor > 0.0);

        for (int step = 0; step < config_.gnSteps; ++step) {
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
            Eigen::VectorXd g = Eigen::VectorXd::Zero(nvar);

            for (int t = 0; t < nFrames; ++t) {
                mju_copy(data->qpos, qWin[t].data(), model->nq);
                mj_forward(model, data);
                const int off = t * m;

                for (const auto& tgt : targets[t]) {
                    const double* xpos = &data->xpos[3 * tgt.bodyId];
                    Eigen::Vector3d bodyPos(xpos[0], xpos[1], xpos[2]);

                    if (tgt.posWeight > 0.0) {
                        const Eigen::Vector3d err = bodyPos - tgt.targetPos;
                        mj_jac(model, data, jacp.data(), nullptr, xpos, tgt.bodyId);
                        Eigen::MatrixXd J(3, m);
                        for (int col = 0; col < m; ++col) {
                            J.col(col) = Eigen::Map<const Eigen::Vector3d>(&jacp[3 * optVidx_[col]]);
                        }
                        const double w = tgt.posWeight;
                        H.block(off, off, m, m).noalias() += w * J.transpose() * J;
                        g.segment(off, m).noalias() += w * J.transpose() * err;
                    }

                    if (tgt.rotWeight > 0.0) {
                        const Eigen::Matrix3d Rbody = bodyRotation(data, tgt.bodyId);
                        Eigen::Quaterniond qBody(Rbody);
                        qBody.normalize();
                        const Eigen::Vector3d err = quatRotError(tgt.targetRot, qBody);
                        mj_jac(model, data, nullptr, jacr.data(), xpos, tgt.bodyId);
                        Eigen::MatrixXd J(3, m);
                        for (int col = 0; col < m; ++col) {
                            J.col(col) = Eigen::Map<const Eigen::Vector3d>(&jacr[3 * optVidx_[col]]);
                        }
                        const double w = tgt.rotWeight;
                        H.block(off, off, m, m).noalias() += w * J.transpose() * J;
                        g.segment(off, m).noalias() += w * J.transpose() * err;
                    }
                }
            }

            if (anchorWeight > 0.0) {
                for (int vi = 0; vi < m; ++vi) {
                    const int v   = optVidx_[vi];
                    const int qadr = model->jnt_qposadr[model->dof_jntid[v]];
                    const double err = qWin[0][qadr] - anchor[qadr];
                    H(vi, vi) += anchorWeight;
                    g[vi] += anchorWeight * err;
                }
            }

            if (config_.wVelocity > 0.0 && nFrames >= 2 && !smoothV.empty()) {
                for (int t = 1; t < nFrames; ++t) {
                    const int offT  = t * m;
                    const int offM1 = (t - 1) * m;
                    for (std::size_t k = 0; k < smoothV.size(); ++k) {
                        const int vi = smoothV[k];
                        const int qadr = smoothQ[k];
                        const double e = qWin[t][qadr] - qWin[t - 1][qadr];
                        const double w = config_.wVelocity;
                        H(offT + vi, offT + vi) += w;
                        H(offM1 + vi, offM1 + vi) += w;
                        H(offT + vi, offM1 + vi) -= w;
                        H(offM1 + vi, offT + vi) -= w;
                        g[offT + vi] += w * e;
                        g[offM1 + vi] -= w * e;
                    }
                }
            }

            if (config_.wAcceleration > 0.0 && nFrames >= 3 && !smoothV.empty()) {
                for (int t = 2; t < nFrames; ++t) {
                    for (std::size_t k = 0; k < smoothV.size(); ++k) {
                        const int vi   = smoothV[k];
                        const int qadr = smoothQ[k];
                        const double e = qWin[t][qadr] - 2.0 * qWin[t - 1][qadr] + qWin[t - 2][qadr];
                        const int i0   = (t - 2) * m + vi;
                        const int i1   = (t - 1) * m + vi;
                        const int i2   = t * m + vi;
                        const double w = config_.wAcceleration;
                        H(i2, i2) += w;
                        H(i1, i1) += 4.0 * w;
                        H(i0, i0) += w;
                        H(i2, i1) -= 2.0 * w;
                        H(i1, i2) -= 2.0 * w;
                        H(i2, i0) += w;
                        H(i0, i2) += w;
                        H(i1, i0) -= 2.0 * w;
                        H(i0, i1) -= 2.0 * w;
                        g[i2] += w * e;
                        g[i1] -= 2.0 * w * e;
                        g[i0] += w * e;
                    }
                }
            }

            if (footActive) {
                const int nFeet = static_cast<int>(footBodyIds_.size());
                std::vector<std::vector<Eigen::Vector3d>> footPos(nFrames, std::vector<Eigen::Vector3d>(nFeet));
                std::vector<std::vector<Eigen::MatrixXd>> footJxy(nFrames, std::vector<Eigen::MatrixXd>(nFeet));
                std::vector<std::vector<Eigen::RowVectorXd>> footJz(nFrames, std::vector<Eigen::RowVectorXd>(nFeet));

                for (int t = 0; t < nFrames; ++t) {
                    mju_copy(data->qpos, qWin[t].data(), model->nq);
                    mj_forward(model, data);
                    for (int f = 0; f < nFeet; ++f) {
                        const int bid = footBodyIds_[f];
                        const double* xpos = &data->xpos[3 * bid];
                        footPos[t][f] = Eigen::Vector3d(xpos[0], xpos[1], xpos[2]);
                        mj_jac(model, data, jacp.data(), nullptr, xpos, bid);
                        Eigen::MatrixXd Jxy(2, m);
                        Eigen::RowVectorXd Jz(m);
                        for (int col = 0; col < m; ++col) {
                            Jxy(0, col) = jacp[3 * optVidx_[col] + 0];
                            Jxy(1, col) = jacp[3 * optVidx_[col] + 1];
                            Jz(col)     = jacp[3 * optVidx_[col] + 2];
                        }
                        footJxy[t][f] = Jxy;
                        footJz[t][f]  = Jz;
                    }
                }

                std::vector<std::vector<Eigen::Vector3d>> refFootPos;
                if (config_.wFootIkAnchor > 0.0) {
                    refFootPos.resize(nFrames, std::vector<Eigen::Vector3d>(nFeet));
                    for (int t = 0; t < nFrames; ++t) {
                        mju_copy(data->qpos, qRef[t].data(), model->nq);
                        mj_forward(model, data);
                        for (int f = 0; f < nFeet; ++f) {
                            const double* xpos = &data->xpos[3 * footBodyIds_[f]];
                            refFootPos[t][f] = Eigen::Vector3d(xpos[0], xpos[1], xpos[2]);
                        }
                    }
                }

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
                            const double err = footPos[t][f].z() - groundZ_;
                            const Eigen::RowVectorXd& Jz = footJz[t][f];
                            const double w               = config_.wFootHeight;
                            H.block(offT, offT, m, m).noalias() += w * Jz.transpose() * Jz;
                            g.segment(offT, m).noalias() += w * Jz.transpose() * err;
                        }

                        if (config_.wFootIkAnchor > 0.0 && !refFootPos.empty()) {
                            const Eigen::Vector2d err = footPos[t][f].head<2>() - refFootPos[t][f].head<2>();
                            const Eigen::MatrixXd& Jxy = footJxy[t][f];
                            const double w               = config_.wFootIkAnchor;
                            H.block(offT, offT, m, m).noalias() += w * Jxy.transpose() * Jxy;
                            g.segment(offT, m).noalias() += w * Jxy.transpose() * err;
                        }
                    }

                    bool anyContact = false;
                    if (!globalRefContact_.empty() && tAbs < static_cast<int>(globalRefContact_.size())) {
                        anyContact = std::any_of(globalRefContact_[tAbs].begin(), globalRefContact_[tAbs].end(),
                                                 [](bool v) { return v; });
                    } else if (globalRefContact_.empty()) {
                        anyContact = true;
                    }

                    if (config_.wRootXyContact > 0.0 && anyContact && model->nq >= 2) {
                        for (int qadr = 0; qadr < 2; ++qadr) {
                            auto it = qToOptV.find(qadr);
                            if (it == qToOptV.end()) {
                                continue;
                            }
                            const int vi  = it->second;
                            const double err = qWin[t][qadr] - qRef[t][qadr];
                            const double w     = config_.wRootXyContact;
                            H(offT + vi, offT + vi) += w;
                            g[offT + vi] += w * err;
                        }
                    }

                    if (config_.wContactJointAnchor > 0.0 && anyContact) {
                        for (std::size_t k = 0; k < smoothV.size(); ++k) {
                            const int vi   = smoothV[k];
                            const int qadr = smoothQ[k];
                            const double err = qWin[t][qadr] - qRef[t][qadr];
                            const double w   = config_.wContactJointAnchor;
                            H(offT + vi, offT + vi) += w;
                            g[offT + vi] += w * err;
                        }
                    }

                    if (config_.wFootSlip > 0.0 && t > 0) {
                        for (int f = 0; f < nFeet; ++f) {
                            bool both = true;
                            if (!globalRefContact_.empty()) {
                                const int tAbs = t + frameOffset;
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
                            const Eigen::Vector2d err = footPos[t][f].head<2>() - footPos[t - 1][f].head<2>();
                            const int offPrev           = (t - 1) * m;
                            const Eigen::MatrixXd& Jt   = footJxy[t][f];
                            const Eigen::MatrixXd& Jp   = footJxy[t - 1][f];
                            const double w              = config_.wFootSlip;
                            H.block(offT, offT, m, m).noalias() += w * Jt.transpose() * Jt;
                            H.block(offPrev, offPrev, m, m).noalias() += w * Jp.transpose() * Jp;
                            H.block(offT, offPrev, m, m).noalias() -= w * Jt.transpose() * Jp;
                            H.block(offPrev, offT, m, m).noalias() -= w * Jp.transpose() * Jt;
                            g.segment(offT, m).noalias() += w * Jt.transpose() * err;
                            g.segment(offPrev, m).noalias() -= w * Jp.transpose() * err;
                        }
                    }
                }
            }

            Eigen::MatrixXd Hreg = H + config_.gnDamping * Eigen::MatrixXd::Identity(nvar, nvar);
            Eigen::VectorXd dqFlat = Hreg.ldlt().solve(g);
            dqFlat = dqFlat.cwiseMax(-config_.gnMaxStep).cwiseMin(config_.gnMaxStep);

            const std::vector<double>& alphas =
                config_.gnLineSearchAlphas.empty() ? std::vector<double>{1.0} : config_.gnLineSearchAlphas;

            if (alphas.size() == 1) {
                const double alpha = alphas.front();
                for (int t = 0; t < nFrames; ++t) {
                    std::fill(dq.begin(), dq.end(), 0.0);
                    for (int vi = 0; vi < m; ++vi) {
                        dq[optVidx_[vi]] = -alpha * dqFlat[t * m + vi];
                    }
                    mj_integratePos(model, qWin[t].data(), dq.data(), 1.0);
                    clipHingeQpos(qWin[t]);
                }
            } else {
                std::vector<Eigen::VectorXd> bestQ = qWin;
                double bestCost = std::numeric_limits<double>::infinity();
                for (double alpha : alphas) {
                    std::vector<Eigen::VectorXd> trial = qWin;
                    for (int t = 0; t < nFrames; ++t) {
                        std::fill(dq.begin(), dq.end(), 0.0);
                        for (int vi = 0; vi < m; ++vi) {
                            dq[optVidx_[vi]] = -alpha * dqFlat[t * m + vi];
                        }
                        mj_integratePos(model, trial[t].data(), dq.data(), 1.0);
                        clipHingeQpos(trial[t]);
                    }
                    double cost = 0.0;
                    for (int t = 0; t < nFrames; ++t) {
                        mju_copy(data->qpos, trial[t].data(), model->nq);
                        mj_forward(model, data);
                        for (const auto& tgt : targets[t]) {
                            const double* xpos = &data->xpos[3 * tgt.bodyId];
                            if (tgt.posWeight > 0.0) {
                                const Eigen::Vector3d e(xpos[0] - tgt.targetPos.x(), xpos[1] - tgt.targetPos.y(),
                                                         xpos[2] - tgt.targetPos.z());
                                cost += tgt.posWeight * e.squaredNorm();
                            }
                        }
                    }
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestQ    = trial;
                    }
                }
                qWin = bestQ;
            }
        }

        return qWin;
    }

    Eigen::VectorXd BatchTrajectoryRetargeter::finalizeQpos(const Eigen::VectorXd& qpos, Retargeter& retargeter,
                                                           const HumanFrame& prepared, bool offsetToGround) {
        (void)prepared;
        (void)offsetToGround;
        retargeter.setQpos(qpos);
        return retargeter.currentQpos();
    }

    std::vector<Eigen::VectorXd> BatchTrajectoryRetargeter::retargetBatch(const std::vector<HumanFrame>& humanFrames,
                                                                          Retargeter& retargeter, bool offsetToGround) {
        lastProfile_ = {};
        if (humanFrames.empty()) {
            return {};
        }

        const auto tTotal = Clock::now();
        std::vector<HumanFrame> prepared;
        std::vector<std::vector<FrameTaskTarget>> targets;
        prepared.reserve(humanFrames.size());
        targets.reserve(humanFrames.size());

        auto t0 = Clock::now();
        for (const auto& frame : humanFrames) {
            HumanFrame prep = prepareHumanFrame(frame, offsetToGround);
            prepared.push_back(prep);
            targets.push_back(targetsForPrepared(prep));
        }
        lastProfile_.prepareMs = elapsedMs(t0);

        t0 = Clock::now();
        std::vector<Eigen::VectorXd> qInit = bootstrapQ(humanFrames, retargeter, offsetToGround);
        lastProfile_.bootstrapMs           = elapsedMs(t0);

        if (config_.enableFootPenalties && !footBodyIds_.empty()) {
            globalRefContact_ = batchContactMask(qInit);
        } else {
            globalRefContact_.clear();
        }

        t0 = Clock::now();
        std::vector<Eigen::VectorXd> qOpt = optimizeSlidingWindows(qInit, targets);
        lastProfile_.optimizeMs            = elapsedMs(t0);

        t0 = Clock::now();
        std::vector<Eigen::VectorXd> qOut;
        qOut.reserve(qOpt.size());
        if (config_.finalizeContact) {
            for (std::size_t i = 0; i < qOpt.size(); ++i) {
                qOut.push_back(finalizeQpos(qOpt[i], retargeter, prepared[i], offsetToGround));
            }
        } else {
            qOut = std::move(qOpt);
        }
        lastProfile_.finalizeMs = elapsedMs(t0);

        lastProfile_.nFrames = static_cast<int>(humanFrames.size());
        lastProfile_.totalMs = elapsedMs(tTotal);
        return qOut;
    }

}  // namespace gmr
