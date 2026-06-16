#include "gmr/retarget/mujoco_collision_limit.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include <glog/logging.h>

namespace gmr {

    namespace {

        bool isWeldedTogether(const mjModel* m, int geom1, int geom2) {
            const int body1 = m->geom_bodyid[geom1];
            const int body2 = m->geom_bodyid[geom2];
            return m->body_weldid[body1] == m->body_weldid[body2];
        }

        bool areGeomBodiesParentChild(const mjModel* m, int geom1, int geom2) {
            const int body_id1     = m->geom_bodyid[geom1];
            const int body_id2     = m->geom_bodyid[geom2];
            const int body_weldid1 = m->body_weldid[body_id1];
            const int body_weldid2 = m->body_weldid[body_id2];

            const int weld_parent_id1     = m->body_parentid[body_weldid1];
            const int weld_parent_id2     = m->body_parentid[body_weldid2];
            const int weld_parent_weldid1 = m->body_weldid[weld_parent_id1];
            const int weld_parent_weldid2 = m->body_weldid[weld_parent_id2];

            const bool cond1 = body_weldid1 == weld_parent_weldid2;
            const bool cond2 = body_weldid2 == weld_parent_weldid1;
            return cond1 || cond2;
        }

        bool passContypeConaffinity(const mjModel* m, int geom1, int geom2) {
            const bool cond1 = (m->geom_contype[geom1] & m->geom_conaffinity[geom2]) != 0;
            const bool cond2 = (m->geom_contype[geom2] & m->geom_conaffinity[geom1]) != 0;
            return cond1 || cond2;
        }

        std::vector<int> homogenizeGeomIdList(const mjModel* m, const std::vector<std::string>& names) {
            std::vector<int> ids;
            ids.reserve(names.size());
            for (const auto& name : names) {
                const int gid = mj_name2id(m, mjOBJ_GEOM, name.c_str());
                if (gid >= 0) {
                    ids.push_back(gid);
                }
            }
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            return ids;
        }

        std::vector<std::pair<int, int>> constructGeomIdPairs(
            const mjModel* m, const std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>& pairs) {
            std::set<std::pair<int, int>> unique;
            for (const auto& p : pairs) {
                const std::vector<int> idsA = homogenizeGeomIdList(m, p.first);
                const std::vector<int> idsB = homogenizeGeomIdList(m, p.second);
                if (idsA.empty() || idsB.empty()) {
                    continue;
                }
                for (int geom_a : idsA) {
                    for (int geom_b : idsB) {
                        if (!isWeldedTogether(m, geom_a, geom_b) && !areGeomBodiesParentChild(m, geom_a, geom_b) &&
                            passContypeConaffinity(m, geom_a, geom_b)) {
                            const int lo = std::min(geom_a, geom_b);
                            const int hi = std::max(geom_a, geom_b);
                            unique.emplace(lo, hi);
                        }
                    }
                }
            }
            return {unique.begin(), unique.end()};
        }

        Eigen::RowVectorXd contactNormalJacobianRow(const mjModel* m, mjData* d, int geom1, int geom2, const mjtNum fromto[6],
                                                    std::vector<mjtNum>& jac1, std::vector<mjtNum>& jac2, int nv) {
            mjtNum normal[3];
            normal[0] = fromto[3] - fromto[0];
            normal[1] = fromto[4] - fromto[1];
            normal[2] = fromto[5] - fromto[2];
            mju_normalize3(normal);

            const int b1 = m->geom_bodyid[geom1];
            const int b2 = m->geom_bodyid[geom2];
            mjtNum p1[3] = {fromto[0], fromto[1], fromto[2]};
            mjtNum p2[3] = {fromto[3], fromto[4], fromto[5]};
            mj_jac(m, d, jac1.data(), nullptr, p1, b1);
            mj_jac(m, d, jac2.data(), nullptr, p2, b2);

            Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> J1(jac1.data(), 3, nv);
            Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> J2(jac2.data(), 3, nv);
            const Eigen::Vector3d n(normal[0], normal[1], normal[2]);
            return n.transpose() * (J2 - J1);
        }

    }  // namespace

    MujocoCollisionLimit::MujocoCollisionLimit(const mjModel* model, const CollisionAvoidanceConfig& cfg)
        : model_(model), cfg_(cfg), nv_(model ? model->nv : 0) {
        if (!model_ || !cfg_.enabled) {
            return;
        }
        geomIdPairs_ = constructGeomIdPairs(model_, cfg_.selfCollisionPairs);
        jac1_.assign(static_cast<size_t>(3 * nv_), 0.0);
        jac2_.assign(static_cast<size_t>(3 * nv_), 0.0);

        LOG(INFO) << "Constructing mujoco collision limit with " << geomIdPairs_.size() << " pairs";
    }

    void MujocoCollisionLimit::fillRows(mjData* data, double dt, double inequalityScale, Eigen::Ref<RowMajorMatrixXd> CI,
                                        Eigen::Ref<Eigen::VectorXd> ciLb, Eigen::Ref<Eigen::VectorXd> ciUb, int rowOffset) const {
        if (geomIdPairs_.empty() || dt <= 1e-15) {
            return;
        }
        const double distmax    = cfg_.detectionDistance;
        const double minDist    = cfg_.minDistance;
        const double gain       = cfg_.gain;
        const double relaxation = cfg_.boundRelaxation;
        mjtNum fromto[6]        = {0};

        for (int idx = 0; idx < static_cast<int>(geomIdPairs_.size()); ++idx) {
            const int r = rowOffset + idx;
            CI.row(r).setZero();
            ciLb[r] = -1e9;
            ciUb[r] = 1e9;

            const int geom1   = geomIdPairs_[static_cast<size_t>(idx)].first;
            const int geom2   = geomIdPairs_[static_cast<size_t>(idx)].second;
            const mjtNum dist = mj_geomDistance(model_, data, geom1, geom2, static_cast<mjtNum>(distmax), fromto);
            if (std::abs(static_cast<double>(dist) - distmax) < 1e-12) {
                continue;
            }

            const Eigen::RowVectorXd row = contactNormalJacobianRow(model_, data, geom1, geom2, fromto, jac1_, jac2_, nv_);
            double h                     = relaxation;
            if (static_cast<double>(dist) > minDist) {
                h = gain * (static_cast<double>(dist) - minDist) / dt + relaxation;
            }
            h *= inequalityScale;
            const double sign = (static_cast<double>(dist) >= 0.0) ? -1.0 : 1.0;
            CI.row(r)         = sign * row;
            ciUb[r]           = h;
        }
    }

}  // namespace gmr
