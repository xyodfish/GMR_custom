#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

namespace gmr {
    namespace batch_internal {

        /// Lower-triangular band storage for symmetric ``n x n`` matrix (bandwidth ``bw``).
        class SymmetricBandedMatrix {
           public:
            void resize(int n, int bw) {
                n_   = n;
                bw_  = std::max(0, bw);
                data_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(bw_ + 1), 0.0);
                droppedEntries_ = 0;
            }

            void setZero() {
                std::fill(data_.begin(), data_.end(), 0.0);
                droppedEntries_ = 0;
            }

            int size() const { return n_; }
            int bandwidth() const { return bw_; }
            int droppedEntries() const { return droppedEntries_; }

            void add(int i, int j, double value) {
                if (value == 0.0 || i < 0 || j < 0 || i >= n_ || j >= n_) {
                    return;
                }
                if (i < j) {
                    std::swap(i, j);
                }
                const int off = i - j;
                if (off > bw_) {
                    ++droppedEntries_;
                    return;
                }
                data_[static_cast<std::size_t>(i) * static_cast<std::size_t>(bw_ + 1) + static_cast<std::size_t>(bw_ - off)] +=
                    value;
            }

            void addBlock(int rowOffset, int colOffset, int blockSize, const Eigen::MatrixXd& blockIn) {
                const Eigen::MatrixXd block = blockIn;
                for (int r = 0; r < blockSize; ++r) {
                    for (int c = 0; c < blockSize; ++c) {
                        add(rowOffset + r, colOffset + c, block(r, c));
                    }
                }
            }

            void addDiagonalBlock(int offset, int blockSize, const Eigen::MatrixXd& blockIn) {
                addBlock(offset, offset, blockSize, blockIn);
            }

            double get(int i, int j) const {
                if (i < j) {
                    std::swap(i, j);
                }
                const int off = i - j;
                if (off > bw_) {
                    return 0.0;
                }
                return data_[static_cast<std::size_t>(i) * static_cast<std::size_t>(bw_ + 1) + static_cast<std::size_t>(bw_ - off)];
            }

            void addDamping(double lambda) {
                for (int i = 0; i < n_; ++i) {
                    add(i, i, lambda);
                }
            }

            /// Solve ``(A + damping I) x = b`` for SPD banded ``A`` (lower storage).
            /// Expands to dense + LDLT for numerical parity with the dense GN path.
            bool solve(const Eigen::VectorXd& b, double damping, Eigen::VectorXd& x) const {
                if (n_ <= 0) {
                    x.resize(0);
                    return true;
                }
                if (b.size() != n_) {
                    throw std::runtime_error("Banded solve: rhs size mismatch.");
                }

                Eigen::MatrixXd Hdense = Eigen::MatrixXd::Zero(n_, n_);
                for (int i = 0; i < n_; ++i) {
                    for (int j = std::max(0, i - bw_); j <= i; ++j) {
                        Hdense(i, j) = get(i, j);
                        if (i != j) {
                            Hdense(j, i) = Hdense(i, j);
                        }
                    }
                }
                const Eigen::MatrixXd Hreg =
                    Hdense + damping * Eigen::MatrixXd::Identity(n_, n_);
                const Eigen::LDLT<Eigen::MatrixXd> ldlt(Hreg);
                if (ldlt.info() != Eigen::Success) {
                    return false;
                }
                x = ldlt.solve(b);
                return true;
            }

           private:
            int n_  = 0;
            int bw_ = 0;
            int droppedEntries_ = 0;
            std::vector<double> data_;
        };

    }  // namespace batch_internal
}  // namespace gmr
