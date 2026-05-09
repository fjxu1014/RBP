#ifndef RBP_HPP
#define RBP_HPP

#ifndef ARMA_64BIT_WORD
#define ARMA_64BIT_WORD
#endif 

// 包含所有 RBP 库组件
#include "mapped_matrix.hpp"
#include "bed_accessor.hpp"
#include "progress_bar.hpp"
#include "rbp_types.hpp"
#include "lanczos.hpp"
#include <string>

namespace rbp {
    
    void blas_set_num_threads(int n_threads);
    int blas_get_num_threads();

    std::string string_time(double seconds);
    template<typename T>
    arma::Mat<T> sym_safe_inverse(const arma::Mat<T>& A);

    template<typename T>
    arma::Mat<T> block_Kx(const ObjectK<T>& K, const arma::Mat<T>& x_T, int ncpus);
    
    template<typename T>
    bool compute_V(const std::vector<arma::umat>& Z, const std::vector<ObjectK<T>>& K, const arma::Col<T>& scale, arma::Mat<T>& V, int ncpus);

    template<typename T>
    arma::Mat<T> cg_solve_cpp(const std::vector<arma::umat>& Z, const std::vector<ObjectK<T>>& K, const arma::Col<T>& scale, const arma::Mat<T>& b, const arma::Mat<T>& x0, int ncpus, double tol, int maxit, bool verbose);
    // 调用系统命令 wc -l 极速获取文件行数
    size_t get_file_lines(const std::string& filename);

    // GRM 模块
    template<typename T>
    void make_KIN(std::string bed_file, std::string out_file, 
                  const arma::uvec& ind_row, const arma::uvec& ind_col,
                  const arma::Col<T>& weights, int method, 
                  size_t snp_chunk_size, size_t ind_chunk_size, 
                  bool is_chunk, bool calc_inverse, int ncpus);

    // kPCA 模块
    template<typename T>
    void kPCA(std::string grm_file, int nPC, double tol, bool isChunk, int ncpus);

    template<typename T>
    std::vector<arma::Mat<T>> random_effect_cpp(
        const std::vector<arma::umat>& Z,
        const std::vector<ObjectK<T>>& K,
        const arma::Col<T>& scale,
        const arma::Mat<T>& Py,
        const arma::Mat<T>& P, 
        int ncpus,
        bool get_pev
    );

    template<typename T>
    single_result single_model_cpp(
        const arma::Col<T>& y,
        const arma::Mat<T>& X,
        const std::vector<arma::umat>& Z, 
        const std::vector<ObjectK<T>>& K,  //包含残差
        const std::string& solver,
        const std::string& test_method, //for GWAS standard, quasi-standard, gamma
        const int criterion, 
        const std::string out_file,
        const arma::uword num_RanVec,
        const std::string& bed_file,
        const arma::uvec& ind_row,
        const arma::uvec& ind_col,
        const arma::umat& Z_geno, 
        bool is_chunk,
        arma::Col<T> pars0,
        bool init_by_mom, // 决定是否开启 MoM-RHE 初始化
        int random_seed,
        int re_gamma_size, //for reestimate gamma
        bool re_estimate_gamma,
        int maxit_par,
        int maxit_cg,
        double tol_iter,
        double tol_cg,
        int ncpus, 
        bool is_repeat,
        bool get_PEV,
        bool do_test //for GWAS
    );

    template<typename T>
    arma::mat gwas_scan_cpp(
        const std::string& bed_file,
        const arma::uvec& ind_row,
        const arma::uvec& ind_col,
        bool is_chunk,
        const std::string& test_method, 
        const arma::Col<T>& Py,
        const arma::Mat<T>& P,           
        const arma::umat& Z_geno,          
        const arma::Mat<T>& Pu,          
        const arma::Mat<T>& RanVec_in,      
        const arma::Mat<T>& X,           
        const std::vector<arma::umat>& Z,
        const std::vector<ObjectK<T>>& K,
        const arma::Col<T>& scale,       
        const arma::Mat<T>& VX,          
        const arma::Mat<T>& XVXi,        
        int re_gamma_size,
        int random_seed,
        T gamma,
        arma::uword chunk_size,
        double cg_tol,
        int cg_maxit,
        int ncpus
    );

    // =========================================================
    // load_out.cpp 数据加载与结果写回接口声明
    // =========================================================
    template<typename T>
    DataInput<T> load_data_cpp(
        const std::string& phe_file,
        const std::vector<int>& phe_pos,                
        const std::vector<int>& fix_pos,
        const std::vector<int>& qcovar_pos,
        const std::vector<int>& rand_pos,
        const std::vector<std::string>& rand_cov_files, 
        const std::vector<std::string>& grm_files,      
        const std::string& geno_file,                   
        const std::vector<size_t>& geno_row_selected,   
        bool is_chunk
    );

    template<typename T>
    void write_out_single(
        const single_result& res,
        const DataInput<T>& data_info,
        const std::string& out_prefix,
        const std::string& bed_file,
        const arma::uvec& ind_col,
        bool get_PEV,
        bool do_test
    );

    template<typename T>
    void write_out_batch(
        const batch_result& res,
        const DataInput<T>& data_info,
        const std::string& out_prefix,
        const std::string& bed_file,
        const arma::uvec& ind_col,
        bool do_test
    );

        
    template<typename T>
    batch_result batch_model_cpp(
        const arma::Mat<T>& y,
        const arma::Mat<T>& X,
        const std::vector<arma::umat>& Z_o,
        const std::vector<ObjectK<T>>& K_o,   
        const std::string& bed_file,       
        const arma::uvec& ind_row,         
        const arma::uvec& ind_col,        
        const std::string& test_method,   
        const arma::umat& Z_geno, 
        bool do_test, 
        int random_seed,
        int num_random,
        int re_gamma_size,
        int ncpus
    );

    
}


#endif // RBP_HPP
