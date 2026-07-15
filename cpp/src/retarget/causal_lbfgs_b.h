#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include <Eigen/Core>

namespace gmr {
    namespace causal_lbfgs {

        struct BoxBounds {
            Eigen::VectorXd lower;
            Eigen::VectorXd upper;
        };

        struct Options {
            int maxIter     = 5;
            int historySize = 8;
            double ftol     = 1e-4;
            double gtol     = 1e-6;
            double minStep  = 1e-12;
            double gradEps  = 1e-8;
        };

        struct Result {
            Eigen::VectorXd x;
            double finalCost = 0.0;
            int iterations = 0;
            bool success   = false;
        };

        using CostGradFn = std::function<void(const Eigen::VectorXd& x, double* cost, Eigen::VectorXd* grad)>;

        inline void projectToBounds(Eigen::VectorXd* x, const BoxBounds& bounds) {
            for (int i = 0; i < x->size(); ++i) {
                if (std::isfinite(bounds.lower[i])) {
                    (*x)[i] = std::max((*x)[i], bounds.lower[i]);
                }
                if (std::isfinite(bounds.upper[i])) {
                    (*x)[i] = std::min((*x)[i], bounds.upper[i]);
                }
            }
        }

        inline Eigen::VectorXd numericalGrad(const std::function<double(const Eigen::VectorXd&)>& cost,
                                             const Eigen::VectorXd& x, double eps) {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
            Eigen::VectorXd xp   = x;
            for (int i = 0; i < x.size(); ++i) {
                const double h = eps * std::max(1.0, std::abs(x[i]));
                xp[i]          = x[i] + h;
                const double fp = cost(xp);
                xp[i]           = x[i] - h;
                const double fm = cost(xp);
                xp[i]           = x[i];
                grad[i]         = (fp - fm) / (2.0 * h);
            }
            return grad;
        }

        inline Result minimizeCore(const std::function<double(const Eigen::VectorXd&)>& evalCost,
                                   const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& evalGrad,
                                   const Eigen::VectorXd& x0, const BoxBounds& bounds, const Options& opts) {
            Result out;
            out.x = x0;

            if (x0.size() == 0) {
                out.success = true;
                return out;
            }

            projectToBounds(&out.x, bounds);

            double f           = evalCost(out.x);
            Eigen::VectorXd grad = evalGrad(out.x);
            Eigen::VectorXd xPrev;
            Eigen::VectorXd gPrev;

            std::vector<Eigen::VectorXd> sHist;
            std::vector<Eigen::VectorXd> yHist;
            sHist.reserve(static_cast<std::size_t>(opts.historySize));
            yHist.reserve(static_cast<std::size_t>(opts.historySize));

            for (int iter = 0; iter < opts.maxIter; ++iter) {
                out.iterations = iter + 1;

                if (grad.norm() < opts.gtol) {
                    out.success = true;
                    break;
                }

                Eigen::VectorXd q = -grad;
                const int m       = static_cast<int>(sHist.size());
                for (int i = m - 1; i >= 0; --i) {
                    const double rho = 1.0 / std::max(sHist[static_cast<std::size_t>(i)].dot(yHist[static_cast<std::size_t>(i)]), 1e-12);
                    const double a   = rho * sHist[static_cast<std::size_t>(i)].dot(q);
                    q.noalias() -= a * yHist[static_cast<std::size_t>(i)];
                }

                if (m > 0) {
                    const double gamma = sHist.back().dot(yHist.back()) / std::max(yHist.back().squaredNorm(), 1e-12);
                    q *= gamma;
                }

                for (int i = 0; i < m; ++i) {
                    const double rho = 1.0 / std::max(sHist[static_cast<std::size_t>(i)].dot(yHist[static_cast<std::size_t>(i)]), 1e-12);
                    const double b   = rho * yHist[static_cast<std::size_t>(i)].dot(q);
                    q.noalias() += (sHist[static_cast<std::size_t>(i)].dot(grad) - b) * sHist[static_cast<std::size_t>(i)];
                }

                Eigen::VectorXd step = q;
                if (step.norm() < opts.minStep) {
                    out.success = true;
                    break;
                }

                const double c1       = 1e-4;
                const double dirDeriv = grad.dot(step);
                double alpha          = 1.0;
                double fPrevIter      = f;
                Eigen::VectorXd xNew  = out.x;
                double fNew           = f;
                bool accepted         = false;

                for (int ls = 0; ls < 20; ++ls) {
                    xNew = out.x + alpha * step;
                    projectToBounds(&xNew, bounds);
                    fNew = evalCost(xNew);
                    if (fNew <= f + c1 * alpha * dirDeriv) {
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }

                if (!accepted) {
                    break;
                }

                xPrev = out.x;
                gPrev = grad;
                out.x = xNew;
                f     = fNew;
                grad  = evalGrad(out.x);

                if (xPrev.size() == out.x.size()) {
                    Eigen::VectorXd s = out.x - xPrev;
                    Eigen::VectorXd y = grad - gPrev;
                    if (s.dot(y) > 1e-12) {
                        if (static_cast<int>(sHist.size()) >= opts.historySize) {
                            sHist.erase(sHist.begin());
                            yHist.erase(yHist.begin());
                        }
                        sHist.push_back(s);
                        yHist.push_back(y);
                    }
                }

                if (iter > 0 && std::abs(f - fPrevIter) < opts.ftol * std::max(1.0, std::abs(fPrevIter))) {
                    out.success = true;
                    break;
                }
            }

            out.finalCost = f;
            if (!out.success && out.iterations >= opts.maxIter) {
                out.success = true;
            }
            return out;
        }

        /// Box-constrained L-BFGS with numerical gradient.
        inline Result minimize(const std::function<double(const Eigen::VectorXd&)>& cost, const Eigen::VectorXd& x0,
                               const BoxBounds& bounds, const Options& opts = {}) {
            auto evalCost = [&](const Eigen::VectorXd& x) -> double {
                Eigen::VectorXd xc = x;
                projectToBounds(&xc, bounds);
                return cost(xc);
            };
            auto evalGrad = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return numericalGrad(evalCost, x, opts.gradEps);
            };
            return minimizeCore(evalCost, evalGrad, x0, bounds, opts);
        }

        /// Box-constrained L-BFGS with analytic cost/gradient callback.
        inline Result minimizeWithCostGrad(CostGradFn costGrad, const Eigen::VectorXd& x0, const BoxBounds& bounds,
                                           const Options& opts = {}) {
            auto evalCost = [&](const Eigen::VectorXd& x) -> double {
                Eigen::VectorXd xc = x;
                projectToBounds(&xc, bounds);
                double cost = 0.0;
                Eigen::VectorXd grad;
                costGrad(xc, &cost, &grad);
                return cost;
            };
            auto evalGrad = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                Eigen::VectorXd xc = x;
                projectToBounds(&xc, bounds);
                double cost = 0.0;
                Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
                costGrad(xc, &cost, &grad);
                return grad;
            };
            return minimizeCore(evalCost, evalGrad, x0, bounds, opts);
        }

    }  // namespace causal_lbfgs
}  // namespace gmr
