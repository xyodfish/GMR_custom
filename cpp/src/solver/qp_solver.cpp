#include "gmr/solver/qp_solver.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

extern "C" {
#include "api.h"
#include "types.h"
}

namespace gmr::solver {

namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

}  // namespace

struct QPSolver::DaqpState {
  DAQPSettings settings{};
  std::vector<double> H;
  std::vector<double> f;
  std::vector<double> A;
  std::vector<double> bupper;
  std::vector<double> blower;
  std::vector<int> sense;
};

QPSolver::QPSolver(const std::string& backend) : backend_(toLower(backend)) {
  qpoasesOptions_.setToMPC();
  qpoasesOptions_.printLevel = qpOASES::PL_NONE;
  qpoasesOptions_.enableEqualities = qpOASES::BT_TRUE;
  qpoasesOptions_.enableRegularisation = qpOASES::BT_TRUE;
  qpoasesOptions_.epsRegularisation = 1e-8;

  if (backend_ == "daqp") {
    daqpState_ = std::make_unique<DaqpState>();
    daqp_default_settings(&daqpState_->settings);
    daqpState_->settings.iter_limit = 1000;
    daqpState_->settings.primal_tol = 1e-6;
    daqpState_->settings.dual_tol = 1e-6;
  }
}

QPSolver::~QPSolver() = default;

const QPOutput& QPSolver::solve(const QPData& qpData) {
  if (backend_ == "daqp") {
    return solveDaqp(qpData);
  }
  if (backend_ == "qpoases" || backend_ == "qpOASES") {
    return solveQpoases(qpData);
  }
  throw std::runtime_error("Unsupported IK QP backend: " + backend_ + " (use daqp or qpoases).");
}

const QPOutput& QPSolver::solveQpoases(const QPData& qpData) {
  const int nv = qpData.H.rows();
  const int nc = qpData.CI.rows();
  output_.resize(nv, 0, nc);
  output_.lambda.resize(nc);

  if (!qpoasesInitialized_ || qpoasesNv_ != nv || qpoasesNc_ != nc) {
    qpoasesSolver_ = std::make_shared<qpOASES::SQProblem>(nv, nc);
    qpoasesSolver_->setOptions(qpoasesOptions_);
    qpoasesInitialized_ = true;
    qpoasesNv_ = nv;
    qpoasesNc_ = nc;
  }

  int nWSR = 100;
  const qpOASES::returnValue ret = qpoasesSolver_->init(qpData.H.data(), qpData.g.data(), qpData.CI.data(), nullptr,
                                                        nullptr, qpData.ciLb.data(), qpData.ciUb.data(), nWSR);

  if (ret == qpOASES::SUCCESSFUL_RETURN || ret == qpOASES::RET_MAX_NWSR_REACHED) {
    output_.status = ret == qpOASES::SUCCESSFUL_RETURN ? QPStatus::kOptimal : QPStatus::kMaxIterReached;
    if (qpoasesSolver_->getPrimalSolution(output_.x.data()) != qpOASES::SUCCESSFUL_RETURN) {
      output_.status = QPStatus::kError;
      return output_;
    }

    output_.iterations = nWSR;
  } else {
    output_.status = QPStatus::kError;
    if (ret == qpOASES::RET_INIT_FAILED_INFEASIBILITY) {
      output_.status = QPStatus::kInfeasible;
    }
  }

  return output_;
}

const QPOutput& QPSolver::solveDaqp(const QPData& qpData) {
  if (daqpState_ == nullptr) {
    throw std::runtime_error("DAQP backend is not initialized.");
  }

  const int nv = qpData.H.rows();
  const int nc = qpData.CI.rows();
  output_.resize(nv, 0, nc);
  output_.lambda.resize(nc);

  DaqpState& st = *daqpState_;
  st.H.resize(static_cast<std::size_t>(nv * nv));
  st.f.resize(static_cast<std::size_t>(nv));
  st.A.resize(static_cast<std::size_t>(nc * nv));
  st.bupper.resize(static_cast<std::size_t>(nc));
  st.blower.resize(static_cast<std::size_t>(nc));
  st.sense.assign(static_cast<std::size_t>(nc), 0);

  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Hmap(st.H.data(), nv, nv);
  Hmap = qpData.H;
  Eigen::Map<Eigen::VectorXd> fmap(st.f.data(), nv);
  fmap = qpData.g;
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Amap(st.A.data(), nc, nv);
  Amap = qpData.CI;
  Eigen::Map<Eigen::VectorXd> buMap(st.bupper.data(), nc);
  buMap = qpData.ciUb;
  Eigen::Map<Eigen::VectorXd> blMap(st.blower.data(), nc);
  blMap = qpData.ciLb;

  DAQPProblem problem{};
  problem.n = nv;
  problem.m = nc;
  problem.ms = 0;
  problem.H = st.H.data();
  problem.f = st.f.data();
  problem.A = st.A.data();
  problem.bupper = st.bupper.data();
  problem.blower = st.blower.data();
  problem.sense = st.sense.data();

  DAQPResult result{};
  result.x = output_.x.data();
  result.lam = output_.lambda.data();

  daqp_quadprog(&result, &problem, &st.settings);

  if (result.exitflag > 0) {
    output_.status = QPStatus::kOptimal;
    output_.iterations = result.iter;
  } else {
    output_.status = QPStatus::kError;
    if (result.exitflag == 0) {
      output_.status = QPStatus::kMaxIterReached;
    }
  }

  return output_;
}

}  // namespace gmr::solver
