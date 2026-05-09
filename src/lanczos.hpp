#ifndef LANCZOS_HPP
#define LANCZOS_HPP

#include <armadillo>
#include <vector>
#include <cmath>
#include <future>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include "mapped_matrix.hpp"

namespace rbp {

    void blas_set_num_threads(int n_threads);
    int blas_get_num_threads();
/**
 * Lanczos 算法的结果结构体
 */
struct LanczosResult {
    arma::vec eigenvalues;   // 特征值
    arma::mat eigenvectors;  // 特征向量矩阵 (按列存储)
    int iterations;          // 实际消耗的迭代步数
};

/**
 * 支持所有 MappedMatrix 模式（完整映射/分块加载/子集映射）的现代化 Lanczos 求解器
 * @tparam T: MappedMatrix 的数据类型 (float 或 double)
 * @param A: 必须是对称矩阵
 * @param m: 需要的最大特征值个数
 * @param lm: 需要的最小特征值个数 (若 lm < 0，则计算绝对值最大的 m 个特征值)
 * @param tol: 收敛容差 (例如 1e-6)
 * @param ncpus: 并行线程数
 * @return LanczosResult 结构体
 */
template<typename T>
LanczosResult lanczos_eigen(const MappedMatrix<T>& A, int m, int lm, double tol, int ncpus = 12) {
    if (A.nrow() != A.ncol()) {
        throw std::runtime_error("Matrix A must be a square symmetric matrix.");
    }

    size_t n = A.nrow();
    bool biggest = (lm < 0);
    if (biggest) lm = 0;

    int f_check = std::max(10, (m + lm) / 2);
    int kk = std::max(1, static_cast<int>(n / 10));
    if (kk < f_check) f_check = kk;

    // Lanczos 向量组和三对角矩阵的对角线/次对角线
    std::vector<arma::vec> q;
    std::vector<double> a, b;
    
    // 随机初始化第一个 q 向量 (保持为 double 精度，确保数学稳定性)
    arma::vec q0(n, arma::fill::randn);
    q0 /= arma::norm(q0);
    q.push_back(q0);

    arma::vec d_final;
    arma::mat v_final;
    int final_j = 0;
    int blas_ncpus = blas_get_num_threads();
    
    // 主循环，最多进行 n 步
    for (int j = 0; j < n; ++j) {
        
        // ------------------------------------------------------------------
        // 核心步骤 1：动态模式计算 z = A * q[j]
        // ------------------------------------------------------------------
        arma::vec z(n, arma::fill::zeros);
        
        if (!A.isChunkedMode()) {
            // ==============================================================
            // 分支 A: 完整映射模式 或 子集映射模式
            // 数据已存在于连续内存，直接获取完整视图进行极速矩阵-向量相乘
            // ==============================================================
            arma::Mat<T> A_full = A.to_arma_mat();
            arma::Col<T> q_j = arma::conv_to<arma::Col<T>>::from(q[j]);
            if (A.nrow() == A.ncol()) {
                z = arma::conv_to<arma::vec>::from(A_full * q_j);
            } else {
                z = arma::conv_to<arma::vec>::from(A_full * (A_full.t() * q_j));
            }
            
        } else {
            // ==============================================================
            // 分支 B: 分块加载模式
            // ==============================================================
            if (A.nrow() == A.ncol()) {
                arma::Row<T> q_j = arma::conv_to<arma::Row<T>>::from(q[j].t());
                arma::Row<T> z_T(n, arma::fill::zeros);

                arma::uword total_cols = n;
                arma::uword chunk_size = 512;

                if (ncpus > 1) {rbp::blas_set_num_threads(1);}
                #pragma omp parallel for schedule(dynamic) num_threads(ncpus)
                for (arma::uword col_start = 0; col_start < total_cols; col_start += chunk_size) {

                    arma::uword col_end = std::min(col_start + chunk_size - 1, total_cols - 1);
                    auto view = A.get_chunk(col_start, col_end);

                    arma::Mat<T>& view_mat = view.mat();
                    z_T.subvec(col_start, col_end) = q_j * view_mat;
                }
                if (ncpus > 1) {rbp::blas_set_num_threads(blas_ncpus);}
                z = arma::conv_to<arma::vec>::from(z_T.t());
            } else {
                arma::Col<T> q_j = arma::conv_to<arma::Col<T>>::from(q[j]);
                arma::Col<T> z_T(n, arma::fill::zeros);
                arma::uword total_cols = A.ncol();
                arma::uword chunk_size = 512;

                if (ncpus > 1) {rbp::blas_set_num_threads(1);}
                #pragma omp parallel for schedule(dynamic) num_threads(ncpus)
                for (arma::uword col_start = 0; col_start < total_cols; col_start += chunk_size) {

                    arma::uword col_end = std::min(col_start + chunk_size - 1, total_cols - 1);
                    auto view = A.get_chunk(col_start, col_end);

                    arma::Mat<T>& view_mat = view.mat();
                    arma::Col<T> z_T_i = view_mat * (view_mat.t() * q_j);
                    #pragma omp critical
                    {
                        z_T += z_T_i;
                    }
                }
                if (ncpus > 1) {rbp::blas_set_num_threads(blas_ncpus);}
                z = arma::conv_to<arma::vec>::from(z_T);
            }
            
        }

        // ------------------------------------------------------------------
        // 核心步骤 2：Lanczos 递推与完全重正交化 (Full Re-orthogonalization)
        // ------------------------------------------------------------------
        double alpha = arma::dot(q[j], z);
        a.push_back(alpha);

        if (j == 0) {
            z -= alpha * q[j];
        } else {
            z -= alpha * q[j] + b.back() * q[j-1];
            
            // 两次完全重正交化，确保绝对的数值稳定
            for (int r = 0; r < 2; ++r) {
                for (int i = 0; i <= j; ++i) {
                    double proj = arma::dot(z, q[i]);
                    z -= proj * q[i];
                }
            }
        }

        double beta = arma::norm(z);
        b.push_back(beta);

        if (j < n - 1) {
            if (beta < 1e-12) {
                final_j = j;
                break; 
            }
            q.push_back(z / beta);
        }

        // ------------------------------------------------------------------
        // 核心步骤 3：每 f_check 步求解三对角矩阵的特征值，检查收敛
        // ------------------------------------------------------------------
        if (((j >= m + lm) && (j % f_check == 0)) || (j == n - 1)) {
            
            arma::mat Tj(j + 1, j + 1, arma::fill::zeros);
            for (int i = 0; i <= j; ++i) {
                Tj(i, i) = a[i];
                if (i < j) {
                    Tj(i, i+1) = b[i];
                    Tj(i+1, i) = b[i];
                }
            }

            arma::vec eigval_asc;
            arma::mat eigvec_asc;
            arma::eig_sym(eigval_asc, eigvec_asc, Tj);
            
            // 转换为降序
            arma::vec d = arma::reverse(eigval_asc);
            arma::mat v = arma::fliplr(eigvec_asc);

            double normTj = std::max(std::abs(d(0)), std::abs(d(j)));
            double max_err = normTj * tol;
            
            std::vector<double> err(j + 1);
            for (int k = 0; k <= j; ++k) {
                err[k] = std::abs(beta * v(j, k)); 
            }

            bool converged = false;
            
            if (biggest) {
                int pi = 0, ni = 0;
                converged = true;
                while (pi + ni < m) {
                    if (std::abs(d(pi)) >= std::abs(d(j - ni))) {
                        if (err[pi] > max_err) { converged = false; break; }
                        else pi++;
                    } else {
                        if (err[j - ni] > max_err) { converged = false; break; }
                        else ni++;
                    }
                }
                if (converged) {
                    m = pi;
                    lm = ni;
                }
            } else {
                converged = true;
                for (int i = 0; i < m; ++i) {
                    if (err[i] > max_err) converged = false;
                }
                for (int i = j; i > j - lm; --i) {
                    if (err[i] > max_err) converged = false;
                }
            }

            if (converged) {
                d_final = d;
                v_final = v;
                final_j = j;
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // 核心步骤 4：提取所需的 Ritz 向量 (特征向量)
    // ------------------------------------------------------------------
    LanczosResult res;
    res.iterations = final_j + 1;
    res.eigenvalues.set_size(m + lm);
    res.eigenvectors.set_size(n, m + lm);

    arma::mat Q(n, final_j + 1);
    for (int i = 0; i <= final_j; ++i) {
        Q.col(i) = q[i];
    }

    // 上端
    for (int k = 0; k < m; ++k) {
        res.eigenvalues(k) = d_final(k);
        res.eigenvectors.col(k) = Q * v_final.col(k); 
    }

    // 下端
    for (int k = m; k < m + lm; ++k) {
        int kk = final_j - (lm + m - k - 1); 
        res.eigenvalues(k) = d_final(kk);
        res.eigenvectors.col(k) = Q * v_final.col(kk);
    }

    return res;
}

//使用mat作为输入
template<typename T>
LanczosResult lanczos_eigen_mat(const arma::Mat<T>& A, int m, int lm, double tol, int ncpus = 12) {
    if (A.n_rows != A.n_cols) {
        throw std::runtime_error("Matrix A must be a square symmetric matrix.");
    }

    size_t n = A.n_rows;
    bool biggest = (lm < 0);
    if (biggest) lm = 0;

    int f_check = std::max(10, (m + lm) / 2);
    int kk = std::max(1, static_cast<int>(n / 10));
    if (kk < f_check) f_check = kk;

    // Lanczos 向量组和三对角矩阵的对角线/次对角线
    std::vector<arma::vec> q;
    std::vector<double> a, b;
    
    // 随机初始化第一个 q 向量 (保持为 double 精度，确保数学稳定性)
    arma::vec q0(n, arma::fill::randn);
    q0 /= arma::norm(q0);
    q.push_back(q0);

    arma::vec d_final;
    arma::mat v_final;
    int final_j = 0;
    int blas_ncpus = blas_get_num_threads();
    
    // 主循环，最多进行 n 步
    for (int j = 0; j < n; ++j) {
        
        // ------------------------------------------------------------------
        // 核心步骤 1：动态模式计算 z = A * q[j]
        // ------------------------------------------------------------------
        arma::Col<T> q_j = arma::conv_to<arma::Col<T>>::from(q[j]);
        arma::vec z = arma::conv_to<arma::vec>::from(A * q_j);
       
        // ------------------------------------------------------------------
        // 核心步骤 2：Lanczos 递推与完全重正交化 (Full Re-orthogonalization)
        // ------------------------------------------------------------------
        double alpha = arma::dot(q[j], z);
        a.push_back(alpha);

        if (j == 0) {
            z -= alpha * q[j];
        } else {
            z -= alpha * q[j] + b.back() * q[j-1];
            
            // 两次完全重正交化，确保绝对的数值稳定
            for (int r = 0; r < 2; ++r) {
                for (int i = 0; i <= j; ++i) {
                    double proj = arma::dot(z, q[i]);
                    z -= proj * q[i];
                }
            }
        }

        double beta = arma::norm(z);
        b.push_back(beta);

        if (j < n - 1) {
            if (beta < 1e-12) {
                final_j = j;
                break; 
            }
            q.push_back(z / beta);
        }

        // ------------------------------------------------------------------
        // 核心步骤 3：每 f_check 步求解三对角矩阵的特征值，检查收敛
        // ------------------------------------------------------------------
        if (((j >= m + lm) && (j % f_check == 0)) || (j == n - 1)) {
            
            arma::mat Tj(j + 1, j + 1, arma::fill::zeros);
            for (int i = 0; i <= j; ++i) {
                Tj(i, i) = a[i];
                if (i < j) {
                    Tj(i, i+1) = b[i];
                    Tj(i+1, i) = b[i];
                }
            }

            arma::vec eigval_asc;
            arma::mat eigvec_asc;
            arma::eig_sym(eigval_asc, eigvec_asc, Tj);
            
            // 转换为降序
            arma::vec d = arma::reverse(eigval_asc);
            arma::mat v = arma::fliplr(eigvec_asc);

            double normTj = std::max(std::abs(d(0)), std::abs(d(j)));
            double max_err = normTj * tol;
            
            std::vector<double> err(j + 1);
            for (int k = 0; k <= j; ++k) {
                err[k] = std::abs(beta * v(j, k)); 
            }

            bool converged = false;
            
            if (biggest) {
                int pi = 0, ni = 0;
                converged = true;
                while (pi + ni < m) {
                    if (std::abs(d(pi)) >= std::abs(d(j - ni))) {
                        if (err[pi] > max_err) { converged = false; break; }
                        else pi++;
                    } else {
                        if (err[j - ni] > max_err) { converged = false; break; }
                        else ni++;
                    }
                }
                if (converged) {
                    m = pi;
                    lm = ni;
                }
            } else {
                converged = true;
                for (int i = 0; i < m; ++i) {
                    if (err[i] > max_err) converged = false;
                }
                for (int i = j; i > j - lm; --i) {
                    if (err[i] > max_err) converged = false;
                }
            }

            if (converged) {
                d_final = d;
                v_final = v;
                final_j = j;
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // 核心步骤 4：提取所需的 Ritz 向量 (特征向量)
    // ------------------------------------------------------------------
    LanczosResult res;
    res.iterations = final_j + 1;
    res.eigenvalues.set_size(m + lm);
    res.eigenvectors.set_size(n, m + lm);

    arma::mat Q(n, final_j + 1);
    for (int i = 0; i <= final_j; ++i) {
        Q.col(i) = q[i];
    }

    // 上端
    for (int k = 0; k < m; ++k) {
        res.eigenvalues(k) = d_final(k);
        res.eigenvectors.col(k) = Q * v_final.col(k); 
    }

    // 下端
    for (int k = m; k < m + lm; ++k) {
        int kk = final_j - (lm + m - k - 1); 
        res.eigenvalues(k) = d_final(kk);
        res.eigenvectors.col(k) = Q * v_final.col(kk);
    }

    return res;
}


} // namespace rbp

#endif // LANCZOS_HPP