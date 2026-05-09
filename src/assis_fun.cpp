#include <sstream>
#include <future>
#include <algorithm>
#include <utility>
#include "rbp.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <omp.h>
#include <iostream>

#ifdef USE_MKL
    #include <mkl.h>
#elif defined(USE_OPENBLAS)
    // OpenBLAS 线程控制函数声明
    extern "C" {
        void openblas_set_num_threads(int num_threads);
        int openblas_get_num_threads(void);
        int openblas_get_parallel(void);
    }
#endif

namespace rbp {

/**
 * 设置 BLAS 库使用的线程数
 * @param n_threads: 线程数
 */
void blas_set_num_threads(int n_threads) {
#ifdef USE_MKL
    mkl_set_num_threads(n_threads);
#elif defined(USE_OPENBLAS)
    openblas_set_num_threads(n_threads);
#else
    // 其他 BLAS 实现可能不支持动态线程控制
    (void)n_threads;
#endif
}

/**
 * 获取当前 BLAS 库使用的线程数
 * @return: 当前线程数
 */
int blas_get_num_threads() {
#ifdef USE_MKL
    return mkl_get_max_threads();
#elif defined(USE_OPENBLAS)
    return openblas_get_num_threads();
#else
    // 默认返回 1（单线程）
    return 1;
#endif
}

// ============================================================================
// 辅助函数
// ============================================================================

std::string string_time(double seconds) {
    int time = static_cast<int>(seconds);
    int hour = time / 3600;
    time %= 3600;
    int min = time / 60;
    int sec = time % 60;

    std::ostringstream oss;
    if (hour != 0) oss << hour << "h";
    if (hour != 0 || min != 0) oss << min << "m";
    oss << sec << "s";
    return oss.str();
}

template<typename T>
arma::Mat<T> sym_safe_inverse(const arma::Mat<T>& A) {
    arma::Mat<T> inv_A;
    if (arma::inv(inv_A, A)) {
        return inv_A;
    } else {
        return arma::pinv(A, static_cast<T>(1e-6));
    }
}

/**
 * 核心算法：block_Kx - 计算 K * x
 * 
 * @tparam T: K矩阵的数据类型（float或double）
 * @param K: ObjectK结构体，包含矩阵类型和数据
 * @param x_T: 输入矩阵
 * @param ncpus: CPU线程数
 * @return: K * x 的结果 (double类型)
 * 
 * 注意：
 * 1. KINSHIP类型计算 K * x
 * 2. COMPRESS类型计算 K * (K.t() * x)
 */
template<typename T>
arma::Mat<T> block_Kx(const ObjectK<T>& K, const arma::Mat<T>& x_T, int ncpus) {
    
    arma::uword nx = x_T.n_cols;
    arma::uword n = x_T.n_rows;

    int blas_ncpus = blas_get_num_threads();
    
    if (K.type == KType::COMPRESS) {
        // 创建 T 类型的累加结果矩阵
        arma::Mat<T> Kx_T(n, nx, arma::fill::zeros);
        
        if (!K.compress.isChunkedMode()) {
            // 不分块模式
            arma::Mat<T> K_mat = K.compress.to_arma_mat();
            arma::Mat<T> Kt_x = K_mat.t() * x_T;
            Kx_T = K_mat * Kt_x;
            
        } else {
            size_t total_cols = K.compress.ncol();
            size_t chunk_size = 512;

            if (ncpus > 1) {blas_set_num_threads(1);}
            #pragma omp parallel for schedule(dynamic) num_threads(ncpus)
            for (size_t col_start = 0; col_start < total_cols; col_start += chunk_size) {

                size_t col_end = std::min(col_start + chunk_size - 1, total_cols - 1);
                auto view = K.compress.get_chunk(col_start, col_end);

                arma::Mat<T>& view_mat = view.mat();
                arma::Mat<T> Kx_T_i = view_mat * (view_mat.t() * x_T);
                #pragma omp critical
                {
                    Kx_T += Kx_T_i;
                }
                
            }
            if (ncpus > 1) {blas_set_num_threads(blas_ncpus);}
        }
        return Kx_T;
        
    } else if (K.type == KType::KINSHIP) {
        
        if (!K.kinship.isChunkedMode()) {
            return K.kinship.to_arma_mat() * x_T;
            
        } else {
            arma::Mat<T> xt_T = x_T.t();
            arma::Mat<T> Kx_T(nx, n, arma::fill::zeros);
            arma::uword total_cols = n;
            arma::uword chunk_size = 512;

            if (ncpus > 1) {blas_set_num_threads(1);}
            #pragma omp parallel for schedule(dynamic) num_threads(ncpus)
            for (arma::uword col_start = 0; col_start < total_cols; col_start += chunk_size) {

                arma::uword col_end = std::min(col_start + chunk_size - 1, total_cols - 1);
                auto view = K.kinship.get_chunk(col_start, col_end);

                arma::Mat<T>& view_mat = view.mat();
                Kx_T.cols(col_start, col_end) = xt_T * view_mat;

            }
            if (ncpus > 1) {blas_set_num_threads(blas_ncpus);}
            return Kx_T.t();
        }
    }
    return arma::Mat<T>();
    
}

/**
 * 通过系统命令极速获取文件行数
 * 注意：必须在支持 POSIX 环境 (Linux/macOS) 下使用
 */
size_t get_file_lines(const std::string& filename) {
    // 使用 '<' 重定向输入，确保 wc -l 只输出数字，不输出文件名
    std::string cmd = "wc -l < " + filename;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed! Cannot execute wc -l command.");
    }
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    // std::stoull 会自动忽略前导空格和换行符
    return std::stoull(result);
}

//计算V = SUM ZKZ' 
template<typename T>
bool compute_V(
    const std::vector<arma::umat>& Z, //已知Z为imat对象，1表示单位阵，或者两列元素依次表示为元素1所在的行号和列号, 0-based
    const std::vector<ObjectK<T>>& K, //已知K为mat对象，1表示单位阵，或者由外部指针生成的非拷贝矩阵
    const arma::Col<T>& scale, 
    arma::Mat<T>& V,
    int ncpus) {
    //K必须时full mapping的，且只能是KINSHIP类型的矩阵
    try {
        
        arma::uword nk = Z.size();
    
        for (arma::uword i=0; i<nk; ++i) {
            const T si = scale.at(i);
            const arma::umat& Zi = Z[i];
            const ObjectK<T>& Ki = K[i];
                
            if (Zi.n_elem == 1 && Ki.type == KType::IDENTITY) {
                // 1. Zi和Ki都是单位阵
                V.diag() += si;
            } else if (Zi.n_elem == 1) {
                // 2. 仅Zi是单位阵
                if (Ki.type != KType::KINSHIP) {
                    throw std::runtime_error("Only KINSHIP input for GRM is supported in direct V method.");
                }

                V += si * Ki.kinship.to_arma_mat();
    
            } else if (Ki.type == KType::IDENTITY) {
                // 3. 仅Ki是单位阵
                arma::uvec z0 = Zi.col(0); 
                arma::uvec z1 = Zi.col(1); 
                std::vector<std::vector<arma::uword>> col_groups(arma::max(Zi.col(1))+1);
                for (arma::uword j = 0; j < Zi.n_rows; ++j) {
                    col_groups[z1(j)].push_back(z0(j));
                }
                #pragma omp parallel for num_threads(ncpus) schedule(static)
                for (size_t k = 0; k < col_groups.size(); ++k) { 
                    const auto& group = col_groups[k]; 
                    if (!group.empty()) { 
                        arma::uvec rows = arma::conv_to<arma::uvec>::from(group);
                        V.submat(rows, rows) += si ;
                    }
                }
            } else {
                // 4. Zi和Ki都不是单位阵
                if (Ki.type != KType::KINSHIP) {
                    throw std::runtime_error("Only KINSHIP input for GRM is supported in direct V method.");
                }
                arma::uvec z0 = Zi.col(0); 
                arma::uvec z1 = Zi.col(1); 
        
                V.submat(z0,z0) += si * Ki.kinship.to_arma_mat().submat(z1,z1); 
                
            }
            
        }
    
        return true;
    
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
        
}
template arma::Mat<double> sym_safe_inverse<double>(const arma::Mat<double>&);
template arma::Mat<float> sym_safe_inverse<float>(const arma::Mat<float>&);

template arma::Mat<double> block_Kx<double>(const ObjectK<double>&, const arma::Mat<double>&, int);
template arma::Mat<float> block_Kx<float>(const ObjectK<float>&, const arma::Mat<float>&, int);

template bool compute_V<double>(const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const arma::Col<double>&, arma::Mat<double>&, int);
template bool compute_V<float>(const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const arma::Col<float>&, arma::Mat<float>&, int);
  

} // namespace rbp