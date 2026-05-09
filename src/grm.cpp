#include "rbp.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <omp.h>

namespace rbp {

/**
 * 辅助函数：计算 GRM 的广义逆矩阵并以 3 列形式输出
 */
template<typename T>
arma::Mat<T> psdinv(MappedMatrix<T>& K, double tol = 1e-6, int unit = 5, int niter = 100, int ncpus = 12) {
    size_t n = K.nrow();
    
    std::cout << "Attempting to get inversion by Cholesky decomposition..." << std::endl;

    // 将映射矩阵载入内存供分解使用
    arma::Mat<T> P;
    if (K.isChunkedMode()) {
        // 如果是分块模式的句柄，重新建立一个完整映射来提取数据
        MappedMatrix<T> K_full(K.filename(), n, n, false, false, true);
        P = K_full.to_arma_mat();
    } else {
        P = K.to_arma_mat();
    }

    arma::Mat<T> Pinv;
    arma::Mat<T> R;

    // 1. 尝试直接 Cholesky 分解
    if (arma::chol(R, P)) {
        std::cout << "Cholesky succeeded | " << std::flush;
        
        arma::inv(Pinv, arma::trimatu(R));
        Pinv = arma::symmatu(Pinv * Pinv.t()); 
        
        std::cout << "inverse matrix calculated" << std::endl;

    } else {
        std::cout << "Cholesky failed, using Lanczos general inverse..." << std::endl;
        //岭逆减去极大特征值方向的分量
        Pinv = P;
        Pinv.diag() += 1e-4; // ridge P
        
        if (arma::chol(R, Pinv)) {

            arma::inv(Pinv, arma::trimatu(R));
            Pinv = arma::symmatu(Pinv * Pinv.t()); 
            
            bool success = false;
            arma::vec val;
            arma::mat vec;

            for (int i = 0; i < niter; ++i) {
                int m = unit * (i + 1);
                
                if (m > static_cast<int>(n / 10) || m > 200) {
                    throw std::runtime_error("Too many eigenvalues smaller than the tolerance; Lanczos stops working");
                }

                // 借助 lanczos_eigen 求解极大特征值
                LanczosResult res = lanczos_eigen_mat(Pinv, m, -1, tol, ncpus);
                val = res.eigenvalues;
                vec = res.eigenvectors;

                arma::uvec index = arma::find(val >= (1.0 / (2.0 * tol + 1e-4)));
                if (index.n_elem < val.n_elem) {
                    val = val.elem(index);
                    vec = vec.cols(index);
                    std::cout << val.n_elem << " eigenvalue(s) smaller than tolerance" << std::endl;
                    
                    success = true;
                    break;
                }
            }

            if (!success) {
                throw std::runtime_error("Too many eigenvalues smaller than the tolerance; please increase niter or unit");
            }

            arma::Mat<T> penalty = arma::conv_to<arma::Mat<T>>::from(vec * arma::diagmat(val) * vec.t());
            Pinv -= penalty;

        } else {
            throw std::runtime_error("Invalid tolerance supplied, ridged Cholesky also failed");
        }
        
        std::cout << "General inverse matrix calculated\n" << std::flush;
    }
    return Pinv;

    
}


/**
 * 构建亲缘关系矩阵 (GRM) 
 * * @tparam T: 计算精度 (float 或 double)
 * @param bed_file: BED 文件路径
 * @param out_file: 结果矩阵 K 的保存路径
 * @param ind_row: 样本索引 (0-based)
 * @param ind_col: SNP 索引 (0-based)
 * @param weights: 每个 SNP 的权重向量 (空则全默认为 1)
 * @param method: 1=Yang 加性, 2=VanRaden 加性, 3=Dominance 显性
 * @param snp_chunk_size: 分块计算时每块处理的 SNP 数量 (默认 10000)
 * @param ind_chunk_size: 分块计算时每块处理的样本数量 
 * @param is_chunk: 是否启用分块计算模式 (默认 false)，对于样本数 > 50,000 强烈建议启用以控制内存使用
 * @param calc_inverse: 是否额外计算并保存三列格式的广义逆矩阵
 * @param ncpus: 并行计算时使用的 CPU 核心数
 */
template<typename T>
void make_KIN(std::string bed_file, 
              std::string out_file, 
              const arma::uvec& ind_row, 
              const arma::uvec& ind_col,
              const arma::Col<T>& weights,
              int method,
              size_t snp_chunk_size,
              size_t ind_chunk_size,
              bool is_chunk,
              bool calc_inverse,
              int ncpus) {

    if (method == 3) {
        out_file += ".GD.bin";
    }else {
        out_file += ".GA.bin";
    }
        
    
    // ====================================================================
    // 提取并保存个体的 ID 列表 (.id 文件)
    // ====================================================================
    std::string bed_path = bed_file;
    if (bed_path.size() < 4 || bed_path.substr(bed_path.size() - 4) != ".bed") {
        bed_path += ".bed";
    }
    std::string fam_path = bed_path.substr(0, bed_path.size() - 4) + ".fam";

    std::ifstream fam_file(fam_path);
    if (!fam_file) {
        throw std::runtime_error("Cannot open FAM file: " + fam_path);
    }

    // 1. 读取整个 fam 文件，获取所有个体的 IID
    std::vector<std::string> all_iids;
    std::string line, fid, iid;
    while (std::getline(fam_file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        // PLINK .fam 前两列是 FID 和 IID
        if (iss >> fid >> iid) { 
            all_iids.push_back(iid); // 仅保留第二列 IID //all_iids.push_back(fid + "\t" + iid);
        }
    }
    fam_file.close();

    // 2. 根据 out_file 生成对应的 .id 文件名
    std::string id_file = out_file.substr(0, out_file.size() - 4) + ".id";
    std::ofstream id_out(id_file);
    if (!id_out) {
        throw std::runtime_error("Cannot create ID file: " + id_file);
    }

    // 3. 严格根据 ind_row (用户的抽样/过滤要求) 输出最终的 ID 列表
    for (size_t i = 0; i < ind_row.n_elem; ++i) {
        size_t orig_idx = ind_row[i];
        if (orig_idx < all_iids.size()) {
            id_out << all_iids[orig_idx] << "\n";
        } else {
            id_out << "UNKNOWN_ID\n"; // 防御性机制，防止越界
        }
    }
    id_out.close();
    // ====================================================================
    
    // 1. 初始化无状态访问器，强制 coding_scheme=0 提取基础的 0-1-2 编码
    BedAccessor<T> bed(bed_file, ind_row, ind_col, !is_chunk, 3, 0);
    
    size_t N = bed.nrow();
    size_t M = bed.ncol();
    bool use_weights = !weights.is_empty();

    if (use_weights) {
        if (weights.n_elem != M) {
            throw std::invalid_argument("SNP weights vector length must match the number of selected SNPs.");
        }
        // 新增检查：利用 Armadillo 的向量化操作极速排查负数
        if (arma::any(weights < 0.0)) {
            throw std::invalid_argument("SNP weights cannot contain negative values.");
        }
    }
    std::cout << N << " individuals and " << M << " markers has been selected in bed file: [" << bed_file <<"]"<< std::endl;


    // 2. 阶段 1：预计算均值和缩放因子
    std::cout << "Pass 1: Pre-calculating statistics for " << M << " SNPs..." << std::endl;
    
    arma::Row<T> p = bed.calculate_snp_stats(ncpus) / 2.0;
    arma::Row<T> scale = arma::Row<T>();
    T total_scale;
    bool use_scale = use_weights;
    std::string method_name;
    
    if (method == 1) {        // Yang
        method_name = "Yang Additive";
        use_scale = true;
        scale = 1 / arma::sqrt(2.0 * (p % (1.0 - p)));
        scale.replace(arma::datum::inf, 0.0); // 避免除以零导致的无穷大
        total_scale = static_cast<T>(1.0 / M);
        if (use_weights) {
            scale %= arma::sqrt(weights);
        } 
    } else if (method == 2) { // VanRaden
        method_name = "VanRaden Additive";
        if (use_scale) {
            scale = arma::sqrt(weights);
        } 
        total_scale = 0.5 / arma::sum(p % (1.0 - p));
    } else if (method == 3) { // Dominance
        method_name = "Dominance";
        if (use_scale) {
            scale = arma::sqrt(weights);
        } 
        total_scale = 1.0 / arma::sum(arma::square(2.0 * (p % (1.0 - p)) ));
    }

    // 3. 阶段 2：内存感知的分块与 GRM 累加
   std::cout << "Pass 2: Accumulating GRM Matrix (" << method_name << ")..." << std::endl;
   
   MappedMatrix<T> K(out_file, N, N, true, is_chunk, true); 

    if (!is_chunk) {
        // ====================================================================
        // 模式 A: 使用 MappedMatrix<T> 完整映射更新
        // ====================================================================
        std::cout << "Mode: full mapping Write." << std::endl;
        std::cout << "Step of markers is " << snp_chunk_size << std::endl;
        if (N > 50000) {
            std::cout << "The number of individuals (" << N 
                      << ") exceeds 50,000. It is highly recommended to set is_chunk=true" << std::endl;
        }
        arma::Mat<T> K_map = K.to_arma_mat();
        
        // --- 进度条预计算逻辑 ---
        size_t total_loops = (M + snp_chunk_size - 1) / snp_chunk_size; 
        size_t task_unit = std::max<size_t>(1, (total_loops + 99) / 100);
        size_t n_step = (total_loops + task_unit - 1) / task_unit; 

        ProgressBar pb(n_step, 50);

        for (size_t start = 0; start < M; start += snp_chunk_size) {
            size_t end = std::min(M - 1, start + snp_chunk_size - 1);
            size_t cols = end - start + 1;
            
            arma::Mat<T> X_chunk(N, cols);
            bed.cols(start, end, X_chunk.memptr());

            // 编码转换
            if (method == 3) { // Dominance
                #pragma omp parallel for schedule(static) num_threads(ncpus)
                for (size_t j = 0; j < cols; ++j) {
                    size_t global_j = start + j;
                    T p_j = p(global_j);
                    T q_j = 1.0 - p_j;
                    arma::Col<T> dom_lut = {
                        static_cast<T>(-2.0 * p_j * p_j), // 纯合子 0
                        static_cast<T>( 2.0 * p_j * q_j), // 杂合子 1
                        static_cast<T>(-2.0 * q_j * q_j), // 纯合子 2
                        static_cast<T>( 0.0 )                    // 缺失值 3
                    };

                    arma::uvec g = arma::conv_to<arma::uvec>::from(X_chunk.col(j));
                    X_chunk.col(j) = dom_lut.elem(g);
                    
                }
            } else { 
                X_chunk.each_row() -= 2.0 * p.subvec(start, end);
            }
             
            if (use_scale) {
                X_chunk.each_row() %= scale.subvec(start, end);
            }

            K_map += X_chunk * X_chunk.t();
            // --- 进度条按 task_unit 刷新 ---
            size_t current_loop = start / snp_chunk_size;
            if (((current_loop) + 1) % task_unit == 0) {
                ++pb;
                pb.display();
            }
        }
        
        K_map *= total_scale;
        K.sync();
        pb.done();

    } else {
        // ====================================================================
        // 模式 B: Out-of-Core 分块写回
        // ====================================================================
        std::cout << "Mode: Large-scale Out-of-Core Chunked Write." << std::endl;
        std::cout << "Step of individuals and markers is " << ind_chunk_size << " and " << snp_chunk_size << std::endl;
         
        // --- 进度条预计算逻辑 ---
        size_t total_loops = (N + ind_chunk_size - 1) / ind_chunk_size; 
        size_t task_unit = std::max<size_t>(1, (total_loops + 99) / 100);
        size_t n_step = (total_loops + task_unit - 1) / task_unit; 

        ProgressBar pb(n_step, 50);
        //int blas_ncpus = blas_get_num_threads();
        //if (ncpus > 1) {blas_set_num_threads(1);}
        //#pragma omp parallel for schedule(dynamic) num_threads(ncpus)
        for (size_t i_start = 0; i_start < N; i_start += ind_chunk_size) {

            arma::uword col_end = std::min(i_start + ind_chunk_size - 1, N - 1);
            auto view = K.get_chunk(i_start, col_end);
            arma::Mat<T>& K_local = view.mat();

            for (size_t start = 0; start < M; start += snp_chunk_size) {
                size_t end = std::min(M - 1, start + snp_chunk_size - 1);
                size_t cols = end - start + 1;
                
                arma::Mat<T> X_chunk(N, cols);
                bed.cols(start, end, X_chunk.memptr());
                // 编码转换
                if (method == 3) { // Dominance
                    //#pragma omp parallel for schedule(static) num_threads(ncpus)
                    for (size_t j = 0; j < cols; ++j) {
                        size_t global_j = start + j;
                        T p_j = p(global_j);
                        T q_j = 1.0 - p_j;
                        arma::Col<T> dom_lut = {
                            static_cast<T>(-2.0 * p_j * p_j), // 纯合子 0
                            static_cast<T>( 2.0 * p_j * q_j), // 杂合子 1
                            static_cast<T>(-2.0 * q_j * q_j), // 纯合子 2
                            static_cast<T>( 0.0 )                    // 缺失值 3
                        };
                        arma::uvec g = arma::conv_to<arma::uvec>::from(X_chunk.col(j));
                        X_chunk.col(j) = dom_lut.elem(g);
                    }
                } else { 
                    X_chunk.each_row() -= 2.0 * p.subvec(start, end);
                }
                if (use_scale) {
                    X_chunk.each_row() %= scale.subvec(start, end);
                }
            
                K_local += (X_chunk * X_chunk.rows(i_start, col_end).t()) * total_scale;
                
            }
            
            view.sync_chunk();
            
            // --- 进度条按 task_unit 刷新 ---
            size_t current_loop = i_start / ind_chunk_size;
            if (((current_loop) + 1) % task_unit == 0) {
                ++pb;
                pb.display();
            }
        }
        //if (ncpus > 1) {blas_set_num_threads(blas_ncpus);}
        pb.done();

    }
    
    std::cout << "The out matrix saved to: " << out_file << std::endl;

    if (calc_inverse) {
        arma::Mat<T> grm_inv = psdinv(K, 1e-6, 5, 100, ncpus);
        // 3. 将结果输出为 3 列格式
        std::string inv_file = out_file.substr(0, out_file.size() - 4) + ".inv.txt";
        std::cout << "Writing the inverse matrix in 3 columns type to " << inv_file << "..." << std::endl;
        
        std::ofstream out(inv_file);
        if (!out) throw std::runtime_error("Cannot open output file for inverse matrix.");

        for (size_t j = 0; j < N; ++j) {
            for (size_t i = j; i < N; ++i) {
                out << (i + 1) << "\t" << (j + 1) << "\t" << grm_inv(i, j) << "\n";
            }
        }
        out.close();
    }
}

// 显式实例化以支持 float 和 double
template arma::Mat<float> psdinv<float>(MappedMatrix<float>&, double, int, int, int);
template arma::Mat<double> psdinv<double>(MappedMatrix<double>&, double, int, int, int);

template void make_KIN<float>(std::string, std::string, const arma::uvec&, const arma::uvec&, const arma::fvec&, int, size_t, size_t, bool, bool, int);
template void make_KIN<double>(std::string, std::string, const arma::uvec&, const arma::uvec&, const arma::vec&, int, size_t, size_t, bool, bool, int);
} // namespace rbp