#pragma once

#include <memory>
#include <string>

#include <qpOASES.hpp>

#include "gmr/solver/qp_data.h"

namespace gmr::solver {

class QPSolver {
 public:
  explicit QPSolver(const std::string& backend = "daqp");
  ~QPSolver();

  QPSolver(const QPSolver&) = delete;
  QPSolver& operator=(const QPSolver&) = delete;

  const QPOutput& solve(const QPData& qpData);

 private:
  struct DaqpState;

  const QPOutput& solveQpoases(const QPData& qpData);
  const QPOutput& solveDaqp(const QPData& qpData);

  std::string backend_;

  std::shared_ptr<qpOASES::SQProblem> qpoasesSolver_;
  qpOASES::Options qpoasesOptions_;
  bool qpoasesInitialized_ = false;
  int qpoasesNv_ = 0;
  int qpoasesNc_ = 0;

  std::unique_ptr<DaqpState> daqpState_;

  QPOutput output_;
};

}  // namespace gmr::solver
