#include "rbp.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

namespace rbp {

// ============================================================================
// 计算 sum Z K Z' * x 
// ============================================================================
template<typename T>
arma::Mat<T> compute_Ax(
    const std::vector<arma::umat>& Z, // 0-based: 两列元素依次表示为非0(1)所在的行号和列号
    const std::vector<ObjectK<T>>& K, // 已知K为ObjectK结构体
    const arma::Col<T>& scale,           // 方差组分向量 
    const arma::Mat<T>& x,            // 输入向量纯粹化为 T
    int ncpus = 1
) {
    arma::uword nk = Z.size();
    arma::uword nx = x.n_cols;
    arma::uword n = x.n_rows;
  
    arma::Mat<T> Ax(n, nx, arma::fill::zeros);
  
    for (arma::uword i = 0; i < nk; ++i) {
        const arma::umat& Zi = Z[i];
        const ObjectK<T>& Ki = K[i];
        const T si = scale.at(i); 
  
        if (Zi.n_elem == 1 && Ki.type == KType::IDENTITY) {
            // 1. Zi和Ki都是单位阵
            Ax += si * x;
  
        } else if (Zi.n_elem == 1) {
            // 2. 仅Zi是单位阵
            Ax += si * block_Kx(Ki, x, ncpus);
  
        } else if (Ki.type == KType::IDENTITY) {
            // 3. 仅Ki是单位阵
            arma::uword ncol_z = arma::max(Zi.col(1)) + 1;
            arma::Mat<T> zx(ncol_z, nx, arma::fill::zeros);
            
            for (arma::uword j = 0; j < Zi.n_rows; ++j) {
                zx.row(Zi(j, 1)) += x.row(Zi(j, 0));
            }
  
            arma::Mat<T> zkzx(n, nx, arma::fill::zeros);
            zkzx.rows(Zi.col(0)) = zx.rows(Zi.col(1));
            Ax += si * zkzx;
  
        } else {
            // 4. Zi和Ki都不是单位阵
            arma::uword ncol_z = (Ki.type == KType::KINSHIP) ? Ki.kinship.nrow() : Ki.compress.nrow();
            
            arma::Mat<T> zx(ncol_z, nx, arma::fill::zeros);
            for (arma::uword j = 0; j < Zi.n_rows; ++j) {
                zx.row(Zi(j, 1)) += x.row(Zi(j, 0));
            }
            
            // 核心矩阵乘法 (直接传递 T 类型)
            arma::Mat<T> kzx = block_Kx(Ki, zx, ncpus); 
            
            arma::Mat<T> zkzx(n, nx, arma::fill::zeros);
            zkzx.rows(Zi.col(0)) = kzx.rows(Zi.col(1));
            Ax += si * zkzx;
        }
    }
  
    return Ax;	
}

// ============================================================================
// 共轭梯度法 (Conjugate Gradient) 主求解器
// ============================================================================
template<typename T>
arma::Mat<T> cg_solve_cpp(
    const std::vector<arma::umat>& Z, 
    const std::vector<ObjectK<T>>& K,
    const arma::Col<T>& scale, 
    const arma::Mat<T>& b, 
    const arma::Mat<T>& x0,
    int ncpus, 
    double tol, 
    int maxit,
    bool verbose
) {
    //关闭迭代信息输出
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  
    arma::uword n = b.n_rows;
    arma::uword nb = b.n_cols;
  
    // 所有的计算容器全部提升为泛型 T
    arma::Mat<T> x = x0.is_zero() ? arma::Mat<T>(n, nb, arma::fill::zeros) : x0;
    arma::Mat<T> r = x0.is_zero() ? b : (b - compute_Ax(Z, K, scale, x0, ncpus));
    arma::Mat<T> p = r;
    
    arma::Col<T> rsold = arma::sum(arma::square(r), 0).t();
    
    // tol 比较时对齐类型
    arma::uvec break_cond = arma::conv_to<arma::uvec>::from(arma::sqrt(rsold) >= static_cast<T>(tol));
    
    arma::uvec ind = arma::find(break_cond == 1); 
    bool cong = false;
    int cong_it = 0;
  
    // 共轭梯度主循环
    for (int nit = 1; nit <= maxit; ++nit) {
        if (arma::sum(break_cond) == 0) {
            cong = true;
            break;
        }
      
        arma::Mat<T> ppi = p.cols(ind);
        arma::Mat<T> Ap = compute_Ax(Z, K, scale, ppi, ncpus);
        arma::Row<T> alpha = rsold.elem(ind).t() / arma::sum(ppi % Ap, 0);
      
        ppi.each_row() %= alpha;
        Ap.each_row() %= alpha;
        x.cols(ind) += ppi;
        r.cols(ind) -= Ap;
        
        arma::Col<T> rsnew = arma::sum(arma::square(r), 0).t();
      
        // 收敛判断
        break_cond = arma::conv_to<arma::uvec>::from(arma::sqrt(rsnew) >= static_cast<T>(tol));
        ind = arma::find(break_cond == 1); 
      
        /*
        if (nit % 40 == 0 || nit == 1 || nit == maxit) {
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            std::cout << "\r" << std::setw(90) << std::left << "" << std::flush;
            std::cout << "\rCGiter: " << nit << "/" << maxit
                      << " | CongNum: " << nb - arma::sum(break_cond) << "/" << nb
                      << " | RunTime: " << string_time(static_cast<double>(time_elapsed) / 1000.0)
                      << std::flush;
        }
        */
      
        if (arma::sum(break_cond) == 0) {
            cong = true;
            cong_it = nit;
            /*
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            std::cout << "\r" << std::setw(90) << std::left << "" << std::flush;
            std::cout << "\rCGiter: " << nit << "/" << maxit
                      << " | CongNum: " << nb - arma::sum(break_cond) << "/" << nb
                      << " | RunTime: " << string_time(static_cast<double>(time_elapsed) / 1000.0)
                      << std::flush;
            */
            break;
        }
  
        // 方向更新
        arma::Mat<T> p_sub = p.cols(ind);
        p_sub.each_row() %= (rsnew.elem(ind) / rsold.elem(ind)).t();
        p.cols(ind) = p_sub + r.cols(ind); 
        
        rsold = rsnew;
    }

    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    
    if (verbose) {
        if (!cong) {
            std::string strtime_perit = string_time(static_cast<double>(time_elapsed) / (1000.0 * maxit));
            std::cout << arma::sum(break_cond) << "/" << nb
                      << " equations still NOT converged after " << maxit << " CG iterations (" 
                      << strtime_perit << " per iteration)" << std::endl;
        }else {
            std::string strtime_perit = string_time(static_cast<double>(time_elapsed) / (1000.0 * cong_it));
            std::cout <<"All " << nb << " equations converged after " << cong_it << " CG iterations (" 
                      << strtime_perit << " per iteration)" << std::endl;
        }
    }
    
    return x;
}

// 显式实例化以支持 float 和 double
template arma::Mat<double> compute_Ax<double>(const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const arma::Col<double>&, const arma::Mat<double>&, int);
template arma::Mat<float> compute_Ax<float>(const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const arma::Col<float>&, const arma::Mat<float>&, int);

template arma::Mat<double> cg_solve_cpp<double>(const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const arma::Col<double>&, const arma::Mat<double>&, const arma::Mat<double>&, int, double, int, bool);
template arma::Mat<float> cg_solve_cpp<float>(const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const arma::Col<float>&, const arma::Mat<float>&, const arma::Mat<float>&, int, double, int, bool);

} // namespace rbp