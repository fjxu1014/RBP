#ifndef RBP_TYPES_HPP
#define RBP_TYPES_HPP

#include <armadillo>
#include <vector>
#include <string>
#include <type_traits>
#include "mapped_matrix.hpp"

namespace rbp {

/**
 * K矩阵类型枚举
 */
enum class KType {
    IDENTITY,   ///< 单位矩阵
    KINSHIP,    ///< 亲缘关系矩阵
    COMPRESS    ///< 压缩矩阵
};

/**
 * K矩阵对象结构（模板化，支持float和double）
 * @tparam T 数据类型，默认为double
 */
template<typename T = double>
struct ObjectK {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "ObjectK only supports float and double types");
    
    KType type;                    ///< 矩阵类型
    MappedMatrix<T> kinship;       ///< 亲缘关系矩阵
    MappedMatrix<T> compress;      ///< 压缩矩阵
    
};

// 为了方便使用，提供常用类型的别名
using ObjectKd = ObjectK<double>;  ///< double类型的ObjectK
using ObjectKf = ObjectK<float>;   ///< float类型的ObjectK


/**
 * AI-REML算法结果结构
 */
template<typename T>
struct AIResult {
    arma::Mat<T> AI;       ///< Average Information 矩阵
    arma::Col<T> RHS;      ///< Right Hand Side
    arma::Col<T> Py;       ///< P*y
    arma::Col<T> Beta;     ///< 固定效应估计值
    arma::Col<T> Varbeta;  ///< Beta的方差
    arma::Col<T> trPG;     ///< trace(PG)
    arma::Mat<T> VyXu;     ///< V^{-1}(yXu)
    arma::Mat<T> VGPy;
    arma::Mat<T> VX;       ///< V^{-1}X
    arma::Mat<T> XVXi;     ///< (X'V^{-1}X)^{-1}
    arma::Mat<T> Pu;       ///< Pu
};

// ============================================================================
template<typename T>
struct DataInput {
    arma::Mat<T> y;                    
    arma::Mat<T> X;                    
    std::vector<arma::umat> Z;         // 所有随机效应的设计矩阵 (0-based)
    std::vector<ObjectK<T>> K;         // 所有随机效应的协方差/亲缘关系矩阵
    arma::umat Z_geno;                 // 基因型数据的设计矩阵
    std::vector<std::string> valid_ids; // 最终保留下来的个体 ID
    std::vector<std::string> phe_names;                   // 表型名称
    std::vector<std::string> x_col_names;                 // 协变量矩阵 X 的完整列名 (Intercept, Fix1_L1, Cov1...)
    std::vector<std::vector<std::string>> grm_ids;        // 每个 GRM 对应的原始个体 ID 列表
    std::vector<std::string> grm_names;                   // 不含 .bin 的 GRM 文件前缀/名称
    std::vector<std::string> rand_names;
    std::vector<std::vector<std::string>> rand_ids;
};

struct single_result {
    arma::vec vc;                 // 方差组分 (强制转换为 double)
    arma::vec se_vc;              // 方差组分标准误 (强制转换为 double)
    arma::vec beta;               // 固定效应估计值 (强制转换为 double)
    arma::vec se_beta;            // 固定效应标准误 (强制转换为 double)
    std::vector<arma::mat> rand_eff; // 随机效应及其PEV、r2等 (强制转换为 double)
    arma::mat gwas_res;           // GWAS扫描结果矩阵 (本身即为 double)
};

struct batch_result {
    arma::vec vg;                
    arma::vec ve;
    arma::vec se_vg;              
    arma::vec se_ve;               
    std::vector<arma::mat> rand_eff; // 随机效应及其PEV、r2等 (强制转换为 double)
    arma::mat gwas_res;           // GWAS扫描结果矩阵 (本身即为 double)
};

} // namespace rbp

#endif // RBP_TYPES_HPP
