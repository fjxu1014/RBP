#include "rbp.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory> // 必须引入 memory 库以支持智能指针

namespace rbp {

/**
 * 基于 K 矩阵 (GRM) 计算主成分分析 (kPCA)
 * @tparam T: 计算精度 (float 或 double)
 * @param grm_file: GRM 矩阵的 .bin 文件路径
 * @param nPC: 需要提取的主成分个数
 * @param tol: Lanczos 算法的收敛容差
 * @param isChunk: 是否启用分块读取模式 (默认关闭)
 * @param ncpus: 线程数
 */
template<typename T>
void kPCA(std::string grm_file, int nPC, double tol, bool isChunk, int ncpus) {
    
    // 1. 解析文件名并推导基础路径
    if (grm_file.size() < 4 || grm_file.substr(grm_file.size() - 4) != ".bin") {
        grm_file += ".bin";
    }
    std::string base_path = grm_file.substr(0, grm_file.size() - 4);
    std::string id_file = base_path + ".id";
    
    // 2. 读取 .id 文件获取个体数 N 和对应的 ID 列表
    std::vector<std::string> iids;
    std::ifstream id_in(id_file);
    if (!id_in) {
        throw std::runtime_error("Cannot open corresponding ID file: " + id_file);
    }
    
    std::string line;
    while (std::getline(id_in, line)) {
        if (!line.empty()) {
            iids.push_back(line);
        }
    }
    id_in.close();
    
    size_t N = iids.size();
    if (N == 0) {
        throw std::runtime_error("The ID file is empty or invalid.");
    }
    
    std::cout << "Loaded " << N << " individuals from " << id_file << std::endl;

    // ------------------------------------------------------------------------
    // 警告提示：大样本量时建议开启分块
    // ------------------------------------------------------------------------
    if (N > 50000 && !isChunk) {
        std::cout << "[WARNING] The number of individuals (" << N 
                  << ") exceeds 50,000. It is highly recommended to set isChunk=true" << std::endl;
    }
    if (isChunk) {
        std::cout << "Mapping file in Chunked Mode..." << std::endl;

    } else {
        std::cout << "Mapping file in Full Memory Mode..." << std::endl;
        
    }
    size_t M;
    std::string col_file = base_path + ".col";
    std::ifstream file_check(col_file);
    if (file_check.is_open()) {
        file_check.close(); 
        M = get_file_lines(col_file); 
        if (M == 0) {
            throw std::runtime_error("The file '" + col_file + "' exists but is empty.");
        }
    } else {
        M = N;
    }
    
    MappedMatrix<T> K(grm_file, N, M, false, isChunk, true);
    std::cout << "Calculating top " << nPC << " PCs by Lanczos-max algorithm..." << std::flush;
    LanczosResult res = lanczos_eigen(K, nPC, -1, tol, ncpus);

    arma::vec val = res.eigenvalues;
    arma::mat vec = res.eigenvectors;
    
    std::cout << " | Done" << std::endl;
    
    // 计算真正的 PCs (特征向量乘以特征值的算术平方根)
    vec.each_row() %= arma::sqrt(arma::abs(val.t())); 
    
    // 5. 将特征值数值按顺序保存为 .val 文件
    std::string val_file = base_path + ".val";
    std::ofstream val_out(val_file);
    if (!val_out) {
        throw std::runtime_error("Cannot create eigenvalue output file: " + val_file);
    }
    for (arma::uword i = 0; i < val.n_elem; ++i) {
        val_out << val(i) << "\n";
    }
    val_out.close();
    std::cout << "Eigenvalues successfully saved to: " << val_file << std::endl;
    
    // 6. 将特征向量 (PCs) 按列追加到 .id 文件之后，保存为 .vec 文件
    std::string vec_file = base_path + ".vec";
    std::ofstream vec_out(vec_file);
    if (!vec_out) {
        throw std::runtime_error("Cannot create eigenvector output file: " + vec_file);
    }
    
    for (size_t i = 0; i < N; ++i) {
        vec_out << iids[i];
        for (int j = 0; j < nPC; ++j) {
            vec_out << "\t" << vec(i, j);
        }
        vec_out << "\n";
    }
    vec_out.close();
    std::cout << "PCs successfully saved to: " << vec_file << std::endl;
}

// 显式实例化以支持多精度类型
template void kPCA<float>(std::string, int, double, bool, int);
template void kPCA<double>(std::string, int, double, bool, int);

} // namespace rbp