#include "rbp.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace rbp {

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
) {
    BedAccessor<T> geno(bed_file, ind_row, ind_col, !is_chunk, 3, 0);

    arma::uword m = geno.ncol();
    arma::uword n = geno.nrow();
    arma::uword nu = RanVec_in.n_cols;
    
    int threads_blas = rbp::blas_get_num_threads();

    arma::Row<T> ZPy(n, arma::fill::zeros);
    arma::Mat<T> ZPZ, ZPu, Zu;
    T qs_scaler = 1.0;

    std::vector<std::vector<arma::uword>> tmp_groups(n);
    for (arma::uword j = 0; j < Z_geno.n_rows; ++j) {
        tmp_groups[Z_geno(j, 1)].push_back(Z_geno(j, 0));
    }

    std::vector<arma::uvec> col_groups(n);
    #pragma omp parallel for num_threads(ncpus) schedule(static)
    for (arma::uword j = 0; j < n; ++j) { 
        if (!tmp_groups[j].empty()) { 
            col_groups[j] = arma::conv_to<arma::uvec>::from(tmp_groups[j]);
        }
    }
    tmp_groups.clear(); tmp_groups.shrink_to_fit();

    if (test_method == "standard") {
        std::cout << "Method: Standard MLM\n";
        if (Z_geno.n_elem == 1) {
            ZPZ = P;
            ZPy = Py.t();
        } else {
            ZPZ.set_size(n, n);
            ZPZ.zeros();
            #pragma omp parallel for num_threads(ncpus) schedule(dynamic)
            for (arma::uword j = 0; j < n; ++j) {
                if (col_groups[j].empty()) continue;
                ZPy(j) = arma::sum(Py.elem(col_groups[j]));
                ZPZ(j, j) = arma::sum(arma::sum(P.submat(col_groups[j], col_groups[j]), 0));
                for (arma::uword i = j+1; i < n; ++i) { 
                    if (col_groups[i].empty()) continue;
                    T val = arma::sum(arma::sum(P.submat(col_groups[i], col_groups[j]), 0));
                    ZPZ(i, j) = val;
                    ZPZ(j, i) = val;
                }
            }
        }
        
    } else if (test_method == "repeat") {
        std::cout << "Method: Standard MLM for repeated measures\n";
        std::cout << "Scaling projection matrix into individual level...\n";
        arma::uword q = VX.n_cols;
        ZPZ.set_size(n, n);
        ZPZ.zeros();
        
        arma::Mat<T> ZVX(n, q, arma::fill::zeros);
        arma::uword cg_size = 512;
        arma::uword total0 = (n + cg_size - 1) / cg_size;
        for (arma::uword j = 0; j < n; j += cg_size) { 
            std::cout << "Scaling chunk " << j / cg_size + 1 << " out of " << total0 << ": ";
            arma::uword start = j;
            arma::uword end = std::min(j + cg_size - 1, n - 1);
            arma::Mat<T> Zj(Z_geno.n_rows, end - start + 1, arma::fill::zeros);
            for (arma::uword k = start; k <= end; ++k) {
                if (!col_groups[k].empty()) { 
                    const arma::uvec& rows = col_groups[k];
                    ZPy(k) = arma::sum(Py.elem(rows));
                    ZVX.row(k) = arma::sum(VX.rows(rows), 0);
                    Zj.submat(rows, arma::uvec{ k - start }).fill(1.0);
                }
            }
            arma::Mat<T> VZjsol = cg_solve_cpp(Z, K, scale, Zj, arma::Mat<T>(1, 1, arma::fill::zeros), ncpus, cg_tol, cg_maxit, true);
            for (arma::uword i = 0; i < n; ++i) { 
                if (!col_groups[i].empty()) { 
                    const arma::uvec& rows2 = col_groups[i];
                    ZPZ.submat(start, i, end, i) = arma::sum(VZjsol.rows(rows2), 0).t();
                }
            }
        }
        ZPZ -= ZVX * XVXi * ZVX.t(); 
        
    } else if (test_method == "gamma") {
        std::cout << "Method: Approximated MLM using gamma factor\n";
        if (gamma <= 0.0) {
            std::cout << "Randomly sampled " << re_gamma_size << " SNPs for gamma estimation (Seed: " << random_seed << ")\n";
            Zu.set_size(n, nu);
            Zu.zeros();
            if (Z_geno.n_elem == 1) {
                ZPy = Py.t();
                Zu = RanVec_in;
            } else {
                #pragma omp parallel for num_threads(ncpus) schedule(dynamic)
                for (arma::uword j = 0; j < n; ++j) {
                    if (col_groups[j].empty()) continue;
                    ZPy(j) = arma::sum(Py.elem(col_groups[j]));
                    Zu.row(j) = arma::sum(RanVec_in.rows(col_groups[j]), 0);
                }
            }
            arma::uvec gamma_snps;
            if (re_gamma_size > 0 && re_gamma_size < m) {
                arma::arma_rng::set_seed(random_seed);
                gamma_snps = arma::sort(arma::randperm(m, re_gamma_size));
            } else {
                throw std::runtime_error("Invalid 're_gamma_size', which must be positive and less than SNP number");
            }
            arma::Mat<T> KZu(n, nu, arma::fill::zeros);
            arma::uword col_block_size = 16;
            if (ncpus > 1) rbp::blas_set_num_threads(1);
            #pragma omp parallel for num_threads(ncpus) schedule(static)
            for(arma::uword j = 0; j < (arma::uword)re_gamma_size; j += col_block_size) {
                arma::uword start = j;
                arma::uword end = std::min(j + col_block_size, (arma::uword)re_gamma_size);
                arma::uword chunk_len = end - start;
                arma::Mat<T> M(n, chunk_len);
                for(arma::uword c = 0; c < chunk_len; ++c) {
                    geno.cols(gamma_snps[start + c], gamma_snps[start + c], M.colptr(c));
                }
                arma::Row<T> geno_colsMean = arma::mean(M, 0);
                M.each_row() -= geno_colsMean;
                geno_colsMean.clamp(0.002, 1.998);
                arma::Row<T> scalar = 1.0 / arma::sqrt(arma::var(M, 0, 0));
                scalar *= std::sqrt(1.0 / re_gamma_size); 
                M.each_row() %= scalar;
                arma::Mat<T> KZu_i = M * (M.t() * Zu);
                #pragma omp critical
                { 
                    KZu += KZu_i; 
                }
            }
            if (ncpus > 1) rbp::blas_set_num_threads(threads_blas);
            if (Z_geno.n_elem == 1) {
                gamma = arma::sum(arma::sum(KZu % Pu, 0)) / nu;
            } else {
                arma::Mat<T> Gu(Pu.n_rows, nu, arma::fill::zeros);
                Gu.rows(Z_geno.col(0)) = KZu.rows(Z_geno.col(1));
                gamma = arma::sum(arma::sum(Gu % Pu, 0)) / nu;
            }
            gamma /= n - 1.0;
            std::cout << "Estimated Gamma: " << gamma << "\n";
        } else {
            std::cout << "Global-Gamma factor: " << gamma << "\n";
            if (Z_geno.n_elem == 1) {
                ZPy = Py.t();
            } else {
                #pragma omp parallel for num_threads(ncpus) schedule(dynamic)
                for (arma::uword j = 0; j < n; ++j) {
                    if (col_groups[j].empty()) continue;
                    ZPy(j) = arma::sum(Py.elem(col_groups[j]));
                }
            }
        }
        
    } else if (test_method == "quasi-standard") {
        std::cout << "Method: Quasi-standard MLM\n";
        arma::Mat<T> RanVec = RanVec_in;
        arma::Mat<T> wwiw = sym_safe_inverse(arma::Mat<T>(X.t() * X)) * X.t();
        RanVec -= X * (wwiw * RanVec);
        if (Z_geno.n_elem == 1) {
            Zu = RanVec.t();
            ZPu = Pu.t();
            ZPy = Py.t();
        } else {
            Zu.set_size(nu, n); Zu.zeros();
            ZPu.set_size(nu, n); ZPu.zeros();
            #pragma omp parallel for num_threads(ncpus) schedule(static)
            for (arma::uword j = 0; j < n; ++j) {
                if (col_groups[j].empty()) continue;
                ZPy(j) = arma::sum(Py.elem(col_groups[j]));
                Zu.col(j) = arma::sum(RanVec.rows(col_groups[j]), 0).t();
                ZPu.col(j) = arma::sum(Pu.rows(col_groups[j]), 0).t();
            }
        }
        T sig_e = scale.n_elem > 0 ? scale.back() : 1.0;
        T trP = arma::sum(arma::sum(ZPu % Zu, 0)) / nu;
        T trPP = arma::sum(arma::sum(ZPu % ZPu, 0)) / nu;
        qs_scaler = (trP - trPP * sig_e) / (gamma * (n - X.n_cols - sig_e * trP));
        std::cout << "Estimated quasi-standard scaling factor: " << qs_scaler;
        if (qs_scaler < 1.0 || !std::isfinite(qs_scaler)) {
            std::cout << ", invalid, corrected to 1.2\n";
            qs_scaler = 1.2;
        } else {
            std::cout << "\n";
        }
    } else {
        throw std::runtime_error("Unknown test_method: " + test_method);
    }

    std::cout << "Scanning (Calculating Quadratic Forms)...\n";
    arma::Row<T> geno_colsMean(m);
    arma::Row<T> XPX(m);
    arma::Row<T> XPy(m);

    arma::uword read_size = 2560; 
    arma::uword total_steps = (m + read_size - 1) / read_size;
    arma::uword task_unit = std::max((arma::uword)1, (total_steps + 99) / 100);
    arma::uword n_step = (total_steps + task_unit - 1) / task_unit;
    ProgressBar pb(n_step, 50);

    if (ncpus > 1) rbp::blas_set_num_threads(1);

    if (test_method == "standard" || test_method == "repeat") {
        for(arma::uword j = 0; j < m; j += read_size) {
            arma::uword end = std::min(j + read_size, m);
            arma::uword num_cols = end - j;
            arma::Mat<T> M(n, num_cols, arma::fill::zeros);
            
            geno.cols(j, end - 1, M.memptr());
            geno_colsMean.subvec(j, end - 1) = arma::mean(M, 0);

            #pragma omp parallel for num_threads(ncpus) schedule(static)
            for (arma::uword k = 0; k < num_cols; k += chunk_size) {
                arma::uword inner_end = std::min(k + chunk_size, num_cols);
                arma::Mat<T> M_sub = M.cols(k, inner_end - 1); 
                
                XPX.subvec(j + k, j + inner_end - 1) = arma::sum((ZPZ * M_sub) % M_sub, 0); 
                XPy.subvec(j + k, j + inner_end - 1) = ZPy * M_sub; 
            }
            if (((j / read_size) + 1) % task_unit == 0) { ++pb; pb.display(); }
        }
    } else if (test_method == "gamma") {
        for(arma::uword j = 0; j < m; j += read_size) {
            arma::uword end = std::min(j + read_size, m);
            arma::uword num_cols = end - j;
            arma::Mat<T> M(n, num_cols, arma::fill::zeros);
            
            geno.cols(j, end - 1, M.memptr());
            geno_colsMean.subvec(j, end - 1) = arma::mean(M, 0);

            #pragma omp parallel for num_threads(ncpus) schedule(static)
            for (arma::uword k = 0; k < num_cols; k += chunk_size) {
                arma::uword inner_end = std::min(k + chunk_size, num_cols);
                arma::Mat<T> M_sub = M.cols(k, inner_end - 1);
                
                M_sub.each_row() -= geno_colsMean.subvec(j + k, j + inner_end - 1);
                XPy.subvec(j + k, j + inner_end - 1) = ZPy * M_sub;
                XPX.subvec(j + k, j + inner_end - 1) = arma::sum(M_sub % M_sub, 0); 
            }
            if (((j / read_size) + 1) % task_unit == 0) { ++pb; pb.display(); }
        }
    } else if (test_method == "quasi-standard") {
        for(arma::uword j = 0; j < m; j += read_size) {
            arma::uword end = std::min(j + read_size, m);
            arma::uword num_cols = end - j;
            arma::Mat<T> M(n, num_cols, arma::fill::zeros);
            
            geno.cols(j, end - 1, M.memptr());
            geno_colsMean.subvec(j, end - 1) = arma::mean(M, 0);

            #pragma omp parallel for num_threads(ncpus) schedule(static)
            for (arma::uword k = 0; k < num_cols; k += chunk_size) {
                arma::uword inner_end = std::min(k + chunk_size, num_cols);
                arma::Mat<T> M_sub = M.cols(k, inner_end - 1);
                
                XPy.subvec(j + k, j + inner_end - 1) = ZPy * M_sub;
                arma::Mat<T> MZu = Zu * M_sub;  
                arma::Mat<T> MZPu = ZPu * M_sub;
                arma::Mat<T> uPMMu = MZu % MZPu;
                arma::Row<T> average = arma::sum(uPMMu, 0) / nu; 
                arma::Row<T> msq = arma::sum(arma::square(uPMMu), 0) / nu; 
                XPX.subvec(j + k, j + inner_end - 1) = (arma::sqrt(arma::square(average) + 4.0 * (qs_scaler + 1.0) * msq) - average) / (2.0 * (qs_scaler + 1.0));
            }
            if (((j / read_size) + 1) % task_unit == 0) { ++pb; pb.display(); }
        }
    }

    pb.done();
    if (ncpus > 1) rbp::blas_set_num_threads(threads_blas);

    std::cout << "Statistic testing...";
    XPX = arma::abs(XPX);
    arma::vec XPX_vec = arma::conv_to<arma::vec>::from(XPX);
    arma::vec XPy_vec = arma::conv_to<arma::vec>::from(XPy);
    double gamma_val = static_cast<double>(gamma);
    
    arma::vec Effect, SE;
    if (test_method == "gamma") {
        Effect = ((XPy_vec / XPX_vec) / gamma_val);
        SE = (1.0 / arma::sqrt(XPX_vec * gamma_val));
    } else {
        Effect = (XPy_vec / XPX_vec);
        SE = (1.0 / arma::sqrt(XPX_vec));
    }
    arma::vec Pvalue = 2 * arma::normcdf(- arma::abs(Effect / SE));
    
    arma::mat revl(m, 4);
    revl.cols(0, 3) = arma::join_horiz(Effect, SE, Pvalue, arma::conv_to<arma::vec>::from((geno_colsMean / 2.0).t()));
    std::cout << " Done\n";
    return revl;
}

template arma::mat gwas_scan_cpp<double>(const std::string&, const arma::uvec&, const arma::uvec&, bool, const std::string&, const arma::Col<double>&, const arma::Mat<double>&, const arma::umat&, const arma::Mat<double>&, const arma::Mat<double>&, const arma::Mat<double>&, const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const arma::Col<double>&, const arma::Mat<double>&, const arma::Mat<double>&, int, int, double, arma::uword, double, int, int);
template arma::mat gwas_scan_cpp<float>(const std::string&, const arma::uvec&, const arma::uvec&, bool, const std::string&, const arma::Col<float>&, const arma::Mat<float>&, const arma::umat&, const arma::Mat<float>&, const arma::Mat<float>&, const arma::Mat<float>&, const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const arma::Col<float>&, const arma::Mat<float>&, const arma::Mat<float>&, int, int, float, arma::uword, double, int, int);


} // namespace rbp