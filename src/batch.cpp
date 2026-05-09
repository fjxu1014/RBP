#include "rbp.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

namespace rbp {


template<typename T>
arma::mat exact_gwas_scan_cpp(
  const std::string& bed_file,
  const arma::uvec& ind_row,
  const arma::uvec& ind_col,
  const arma::Mat<T>& Py,    
  const arma::umat& Z_geno,    
  const arma::Mat<T>& U,            
  const arma::Col<T>& eigenVal,     
  const arma::Col<T>& vg,           
  const arma::Col<T>& ve,           
  const arma::Mat<T>& X,            
  const arma::Mat<T>& XXXi,         
  bool is_compress,
  const arma::Mat<T>& Uy, 
  const arma::Mat<T>& Sy,
  arma::uword chunk_size,
  int ncpus
) {
  BedAccessor<T> geno(bed_file, ind_row, ind_col, true, 3, 0);

  arma::uword m = geno.ncol();
  arma::uword n_indiv = geno.nrow();
  arma::uword n_obs = Py.n_rows;
  arma::uword ne = eigenVal.n_elem;
  arma::uword np = Py.n_cols;
  arma::uword q = X.n_cols;

  std::cout << "Method: Exact MLM (Iterative AIREML for " << np << " phenotypes)\n";

  // 1. 数据折叠与预计算 (个体层面)
  arma::Mat<T> ZSy(n_indiv, np, arma::fill::zeros);
  arma::Mat<T> ZU(n_indiv, ne, arma::fill::zeros);
  arma::Mat<T> Z_X(n_indiv, q, arma::fill::zeros);
  arma::Col<T> D_Z(n_indiv, arma::fill::zeros); 
  arma::Mat<T> XXi = sym_safe_inverse(arma::Mat<T>(X.t() * X));

  if (Z_geno.n_elem == 1 || Z_geno.is_empty()) {
    ZSy = Sy; ZU = U; Z_X = X; D_Z.ones();
  } else {
    for (arma::uword j = 0; j < Z_geno.n_rows; ++j) {
      ZSy.row(Z_geno(j, 1)) += Sy.row(Z_geno(j, 0));
      ZU.row(Z_geno(j, 1)) += U.row(Z_geno(j, 0));
      Z_X.row(Z_geno(j, 1)) += X.row(Z_geno(j, 0));
      D_Z(Z_geno(j, 1)) += 1.0;
    }
  }

  // 零空间表型残余能量补偿
  arma::Row<T> y_null_sq(np, arma::fill::zeros);
  if (is_compress) {
    y_null_sq = arma::sum(arma::square(Sy), 0) - arma::sum(arma::square(Uy), 0);
    y_null_sq.transform([](T val) { return val < 0 ? static_cast<T>(0) : val; });
  }

  arma::Mat<T> Effect_mat(m, np, arma::fill::zeros);
  arma::Mat<T> SE_mat(m, np, arma::fill::zeros);
  arma::Mat<T> Vg_mat(m, np, arma::fill::zeros);
  arma::Mat<T> Ve_mat(m, np, arma::fill::zeros);
  arma::Col<T> geno_colsMean(m);

  arma::uword total_steps = (m + chunk_size - 1) / chunk_size;
  arma::uword task_unit = std::max((arma::uword)1, (total_steps + 99) / 100);
  ProgressBar pb((total_steps + task_unit - 1) / task_unit, 50);

  int threads_blas = rbp::blas_get_num_threads();
  arma::Row<T> theta_0 = (vg / ve).t(); // 初始化 Vg/Ve
  int maxit = 15;
  T tol = 1e-5;

  // 2. 分块扫描
  for(arma::uword j = 0; j < m; j += chunk_size) {
    arma::uword end = std::min(j + chunk_size, m);
    arma::uword num_cols = end - j;
    arma::Mat<T> M(n_indiv, num_cols, arma::fill::zeros);
    geno.cols(j, end - 1, M.memptr());
    geno_colsMean.subvec(j, end - 1) = arma::mean(M, 0).t();
    
    // 零空间分母补偿项预计算
    arma::Mat<T> M_sq = M % M; M_sq.each_col() %= D_Z;
    arma::Col<T> term1 = arma::sum(M_sq, 0).t(); 
    arma::Mat<T> M_ZX = M.t() * Z_X;       
    arma::Col<T> term2 = arma::sum(M_ZX % (M_ZX * XXi), 1); 
    arma::Col<T> diag_SXX = term1 - term2;
    diag_SXX.transform([](T val) { return val < 0 ? static_cast<T>(0) : val; });

    arma::Mat<T> GZU = M.t() * ZU; 
    arma::Mat<T> XSy = M.t() * ZSy; 
    
    if (ncpus > 1) rbp::blas_set_num_threads(1);
    #pragma omp parallel for num_threads(ncpus) schedule(static)
    for (arma::uword k = 0; k < num_cols; ++k) {
      arma::Col<T> Ux_i = GZU.row(k).t(); 
      arma::Row<T> x_null_y_j = is_compress ? XSy.row(k) - Ux_i.t() * Uy : arma::Row<T>(np, arma::fill::zeros); 
      T u_sq_sum = arma::sum(arma::square(Ux_i));
      T x_null_sq_j = is_compress ? std::max(static_cast<T>(0), diag_SXX[k] - u_sq_sum) : 0.0;
      arma::Col<T> ux_sq = arma::square(Ux_i); 

      arma::Row<T> theta_i = theta_0;

      for(int it = 0; it < maxit; ++it) {
        arma::Mat<T> Vi_inv = 1.0 / (eigenVal * theta_i + 1.0); 
        arma::Row<T> xvxi_inv = 1.0 / (ux_sq.t() * Vi_inv + x_null_sq_j); 

        arma::Mat<T> t1 = Vi_inv; t1.each_col() %= Ux_i; 
        arma::Mat<T> t1_scaled = t1; t1_scaled.each_row() %= xvxi_inv;
        arma::Row<T> term2_exact = eigenVal.t() * (Vi_inv - (t1_scaled % t1)); 

        arma::Mat<T> VUy = Vi_inv % Uy; 
        arma::Row<T> xUVUy = Ux_i.t() * VUy + x_null_y_j;

        arma::Mat<T> t3 = VUy - t1_scaled.each_row() % xUVUy;
        arma::Mat<T> t4 = t3.each_col() % eigenVal;
        arma::Mat<T> Vt4 = Vi_inv % t4;
        arma::Mat<T> t5 = Vt4 - t1_scaled.each_row() % (Ux_i.t() * Vt4);

        arma::Row<T> yPy = arma::sum(t3 % Uy, 0) + (is_compress ? (y_null_sq - xvxi_inv % xUVUy % x_null_y_j) : arma::Row<T>(np, arma::fill::zeros));
        arma::Row<T> grad = -0.5 * (term2_exact - (n_obs - q - 1) * (arma::sum(t3 % t4, 0) / yPy));
        arma::Row<T> AI = 0.5 * (n_obs - q - 1) * (arma::sum(t4 % t5, 0) / yPy);

        theta_i += grad / AI;
        theta_i.clamp(1e-6, 1e6);

        Effect_mat.row(j + k) = xvxi_inv % xUVUy;
        SE_mat.row(j + k) = arma::sqrt((yPy / (n_obs - q - 1)) % xvxi_inv);
        arma::Row<T> ve_k = yPy / (n_obs - q - 1);
        Vg_mat.row(j + k) = theta_i % ve_k;
        Ve_mat.row(j + k) = ve_k;

        if (arma::abs(grad).max() < tol) break;
      }
      
    }
    rbp::blas_set_num_threads(threads_blas);
    if (((j / chunk_size) + 1) % task_unit == 0) { ++pb; pb.display();  }
  }
  pb.done();
  
  // 3. 组装结果 (MAF, Effect, SE, Pval, Vg, Ve)
  arma::mat revl(m, np * 5 + 1);
  revl.col(0) = arma::conv_to<arma::vec>::from(geno_colsMean / 2.0);
  #pragma omp parallel for num_threads(ncpus) schedule(static)
  for(arma::uword i = 0; i < np; ++i) {
    revl.col(1 + i*5) = arma::conv_to<arma::vec>::from(Effect_mat.col(i));
    revl.col(2 + i*5) = arma::conv_to<arma::vec>::from(SE_mat.col(i));
    revl.col(3 + i*5) = 2 * arma::normcdf(-arma::abs( revl.col(1 + i*5) / revl.col(2 + i*5) ));
    revl.col(4 + i*5) = arma::conv_to<arma::vec>::from(Vg_mat.col(i));
    revl.col(5 + i*5) = arma::conv_to<arma::vec>::from(Ve_mat.col(i));
  }
  return revl;
}
  
template<typename T>
arma::mat batch_gwas_scan_cpp(
  const std::string& bed_file,
  const arma::uvec& ind_row,
  const arma::uvec& ind_col,
  const std::string& test_method, 
  const arma::Mat<T>& Py,    
  const arma::umat& Z_geno,    
  const arma::Mat<T>& U,            // 遗传正交基 N x M_nz
  const arma::Col<T>& eigenVal,     // 特征值 M_nz
  const arma::Col<T>& vg,           // 遗传方差 np
  const arma::Col<T>& ve,           // 环境残差 np 
  const arma::Mat<T>& X,            // 固定效应 N_obs x q
  const arma::Mat<T>& XXXi,         // N_obs x q
  bool re_estimate_gamma,
  bool is_compress,
  int re_gamma_size,
  int num_random,
  int random_seed,
  arma::Col<T> gamma,
  arma::uword chunk_size,
  int ncpus
) {
  BedAccessor<T> geno(bed_file, ind_row, ind_col, true, 3, 0);

  arma::uword m = geno.ncol();
  arma::uword n = geno.nrow();
  arma::uword nu = num_random;
  arma::uword ne = eigenVal.n_elem;
  arma::uword np = Py.n_cols;
  arma::Mat<T> ETA(np, ne, arma::fill::none);
  for(arma::uword i = 0; i < np; ++i) {
    ETA.row(i) = static_cast<T>(1.0) / (vg[i] * eigenVal.t() + ve[i]);
  }
    
  int threads_blas = rbp::blas_get_num_threads();

  arma::Mat<T> ZPy(n, np, arma::fill::zeros);
  arma::Mat<T> ZU;

  if (test_method == "standard") {
    std::cout << "Method: Standard MLM\n";
    if (Z_geno.n_elem == 1) {
      ZU = U;
      ZPy = Py;
    } else {
      ZU = arma::Mat<T>(n, ne, arma::fill::zeros);
      for (arma::uword j = 0; j < Z_geno.n_rows; ++j) {
        ZPy.row(Z_geno(j, 1)) += Py.row(Z_geno(j, 0));
        ZU.row(Z_geno(j, 1)) += U.row(Z_geno(j, 0));
      }
    }
        
  } else if (test_method == "gamma") {
    std::cout << "Method: Approximated MLM using gamma factor\n";
    if (Z_geno.n_elem == 1) {
      ZPy = Py;
    } else {
      for (arma::uword j = 0; j < Z_geno.n_rows; ++j) {
        ZPy.row(Z_geno(j, 1)) += Py.row(Z_geno(j, 0));
      }
    }
    if (re_estimate_gamma) {
      std::cout << "Randomly sampled " << re_gamma_size << " SNPs for gamma estimation (Seed: " << random_seed << ")\n";
      arma::arma_rng::set_seed(random_seed);
      arma::Mat<T> R(U.n_rows, nu, arma::fill::randn);
      R.transform([](T val) { return val >= 0 ? static_cast<T>(1) : static_cast<T>(-1); });
      arma::Mat<T> SR = R - XXXi * (X.t() * R);
      arma::Mat<T> UR = U.t() * R;
      
      arma::uvec gamma_snps;
      if (re_gamma_size > 0 && re_gamma_size < m) {
        gamma_snps = arma::sort(arma::randperm(m, re_gamma_size));
      } else {
        throw std::runtime_error("Invalid 're_gamma_size', which must be positive and less than SNP number");
      }
      arma::Mat<T> M(n, re_gamma_size);
      #pragma omp parallel for num_threads(ncpus) schedule(static)
      for(arma::uword j = 0; j < (arma::uword)re_gamma_size; j++) {
        geno.cols(gamma_snps[j], gamma_snps[j], M.colptr(j));
      }
      arma::Row<T> geno_colsMean = arma::mean(M, 0);
      M.each_row() -= geno_colsMean;
      geno_colsMean.clamp(0.002, 1.998);
      arma::Row<T> scalar = 1.0 / arma::sqrt(arma::var(M, 0, 0));
      scalar *= std::sqrt(1.0 / re_gamma_size); 
      M.each_row() %= scalar;
      M = M.rows(Z_geno.col(1));

      arma::Mat<T> MMR = M * (M.t() * R);
      arma::Mat<T> UMMR = U.t() * MMR;
      arma::Mat<T> RUUMMR = UR % UMMR;
      arma::Mat<T> tr_rand = ETA * RUUMMR;
      if (is_compress) {
        arma::Row<T> diff = arma::sum(SR % MMR, 0) - arma::sum(RUUMMR, 0);
        tr_rand += (1.0 / ve) * diff;
      }
      gamma = arma::mean(tr_rand, 1);
      gamma /= n - 1.0;

      std::cout << "Estimated Gamma mean: " << arma::mean(gamma) << "\n";
    } else {
      std::cout << "Global-Gamma factor mean: " << arma::mean(gamma) << "\n";
      
    }
        
  } else {
    throw std::runtime_error("Unknown test_method: " + test_method);
  }

  std::cout << "Scanning (Calculating Quadratic Forms)...\n";
  arma::Col<T> geno_colsMean(m);
  arma::Mat<T> XPX(m, np);
  arma::Mat<T> XPy(m, np);

  arma::uword read_size = chunk_size; 
  arma::uword total_steps = (m + read_size - 1) / read_size;
  arma::uword task_unit = std::max((arma::uword)1, (total_steps + 99) / 100);
  arma::uword n_step = (total_steps + task_unit - 1) / task_unit;
  ProgressBar pb(n_step, 50);
  if (ncpus > 1) rbp::blas_set_num_threads(1);

  if (test_method == "standard") {
    if (is_compress) {
      arma::Mat<T> Z_X(n, X.n_cols, arma::fill::zeros);
      arma::Col<T> D_Z(n, arma::fill::zeros); // 记录每个个体有几条观测值
      arma::Mat<T> XXi = sym_safe_inverse(arma::Mat<T>(X.t() * X));;
      if (Z_geno.n_elem == 1) {
        Z_X = X;
        D_Z.ones();
      } else {
        for (arma::uword i = 0; i < Z_geno.n_rows; ++i) {
          Z_X.row(Z_geno(i, 1)) += X.row(Z_geno(i, 0));
          D_Z(Z_geno(i, 1)) += 1.0;
        }
      }
      #pragma omp parallel for num_threads(ncpus) schedule(static)
      for(arma::uword j = 0; j < m; j += read_size) {
        arma::uword end = std::min(j + read_size, m);
        arma::uword num_cols = end - j;
        arma::Mat<T> M(n, num_cols, arma::fill::zeros);
        geno.cols(j, end - 1, M.memptr());
        geno_colsMean.subvec(j, end - 1) = arma::mean(M, 0).t();
        XPy.rows(j, end - 1) = M.t() * ZPy;
        arma::Mat<T> XZU = M.t() * ZU; //mi * ne
        XZU %= XZU;
        arma::Mat<T> M_sq = M % M;
        M_sq.each_col() %= D_Z; // 个体层面乘以观测次数权重
        arma::Col<T> term1 = arma::sum(M_sq, 0).t(); // num_cols x 1
        arma::Mat<T> M_ZX = M.t() * Z_X;       // num_cols x q
        arma::Col<T> term2 = arma::sum(M_ZX % (M_ZX * XXi), 1); // num_cols x 1
        arma::Col<T> diff_i = (term1 - term2) - arma::sum(XZU, 1);
        diff_i.transform([](T val) { return val < 0 ? static_cast<T>(0) : val; });
        XPX.rows(j, end - 1) = XZU * ETA.t() + diff_i * (1.0 / ve).t();
        if (((j / read_size) + 1) % task_unit == 0) { 
          #pragma omp critical
          {
            ++pb; pb.display(); 
          }
        }
      }
    } else {
      #pragma omp parallel for num_threads(ncpus) schedule(static)
      for(arma::uword j = 0; j < m; j += read_size) {
        arma::uword end = std::min(j + read_size, m);
        arma::uword num_cols = end - j;
        arma::Mat<T> M(n, num_cols, arma::fill::zeros);
        geno.cols(j, end - 1, M.memptr());
        geno_colsMean.subvec(j, end - 1) = arma::mean(M, 0).t();
        XPy.rows(j, end - 1) = M.t() * ZPy;
        arma::Mat<T> XZU = M.t() * ZU;
        XZU %= XZU;
        XPX.rows(j, end - 1) = XZU * ETA.t();
        if (((j / read_size) + 1) % task_unit == 0) { 
          #pragma omp critical
          {
            ++pb; pb.display(); 
          }
        }
      }
    }
    
  } else if (test_method == "gamma") {
    #pragma omp parallel for num_threads(ncpus) schedule(static)
    for(arma::uword j = 0; j < m; j += read_size) {
      arma::uword end = std::min(j + read_size, m);
      arma::uword num_cols = end - j;
      arma::Mat<T> M(n, num_cols, arma::fill::zeros);
      geno.cols(j, end - 1, M.memptr());
      geno_colsMean.subvec(j, end - 1) = arma::mean(M, 0).t();
      M.each_row() -= geno_colsMean.subvec(j, end - 1).t();
      XPy.rows(j, end - 1) = M.t() * ZPy;
      arma::Col<T> col_vec = arma::sum(M % M, 0).t();
      for(arma::uword c = 0; c < np; ++c) {
        XPX.submat(j, c, end - 1, c) = col_vec;
      }
      if (((j / read_size) + 1) % task_unit == 0) { 
        #pragma omp critical
        {
          ++pb; pb.display(); 
        }
      }
    }
  } 
  pb.done();
  if (ncpus > 1) rbp::blas_set_num_threads(threads_blas);

  std::cout << "Statistic testing...";
  arma::mat XPX_vec = arma::conv_to<arma::mat>::from(XPX);
  arma::mat XPy_vec = arma::conv_to<arma::mat>::from(XPy);
  arma::vec gamma_val = arma::conv_to<arma::vec>::from(gamma);

  if (test_method == "gamma") {
    XPX_vec.each_row() /= gamma_val.t();
  }
  arma::mat& SE = XPX_vec;
  arma::mat& Effect = XPy_vec;
  Effect /= SE;
  SE.transform([](double x) { return 1.0 / std::sqrt(x); }); // in-place operation
  arma::mat revl(m, (np+np+np+1));
  revl.col(0) = arma::conv_to<arma::vec>::from(geno_colsMean / 2.0);
  #pragma omp parallel for num_threads(ncpus) schedule(static)
  for(arma::uword i = 0; i < np; ++i) {
    revl.col(1 + i*3) = Effect.col(i);
    revl.col(2 + i*3) = SE.col(i);
    revl.col(3 + i*3) = 2 * arma::normcdf(- arma::abs(Effect.col(i) / SE.col(i)));
  }
    
  std::cout << " Done\n";
  return revl;
}

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
) {
  std::cout << "=====================[ DFREML Running ]=====================\n";
  int threads_blas = rbp::blas_get_num_threads();
  arma::uword n = y.n_rows;
  arma::uword np = y.n_cols;
  arma::uword q = X.n_cols;
  T constant = static_cast<T>(-0.5 * (n - q) * (1.0 + std::log(2.0 * arma::datum::pi / (1.0 * (n - q)) )));
  const arma::uvec Z = Z_o[0].col(1);
  const ObjectK<T>& K = K_o[0];
  arma::Mat<T> XXXi = X * sym_safe_inverse(arma::Mat<T>(X.t() * X));
  arma::Col<T> eigenVal;
  arma::Mat<T> U;
  // 降维标记：标记是否触发了 M < N 的 SVD 降维加速
  bool is_compact = false; 
  arma::uword M_nz = 0;
  std::cout << "Spectral decomposition for variance components...\n";
  if (K.type == KType::KINSHIP) {
    arma::Mat<T> K_mat = K.kinship.to_arma_mat();
    arma::Mat<T> SKS;
    arma::Mat<T> KX;
    if (Z.n_elem == 1) {
      KX = K_mat * X;
    } else {
      KX = K_mat.submat(Z, Z) * X;
    }
    arma::Mat<T> XKX = X.t() * KX;
    SKS = -KX * XXXi.t();
    SKS += SKS.t();
    SKS += XXXi * XKX * XXXi.t();
    if (Z.n_elem == 1) {
       SKS += K_mat;
    } else {
      SKS += K_mat.submat(Z, Z);
    }
    arma::eig_sym(eigenVal, U, SKS);
    if (q > 0) {
      eigenVal.shed_rows(0, q - 1);
      U.shed_cols(0, q - 1);
    }
    eigenVal.transform([](T val) { return val < static_cast<T>(1e-6) ? static_cast<T>(1e-6) : val; });
  } else if (K.type == KType::COMPRESS) {
    arma::Mat<T> A_full = K.compress.to_arma_mat();
    arma::Mat<T> A = (Z.n_elem == 1) ? A_full : A_full.rows(Z);
    arma::uword M = A.n_cols;
    if (n <= M) {
      // 分支 A：特征多于样本，恢复构建 N x N 矩阵并维持传统 SKS 逻辑
      arma::Mat<T> K_mat = A * A.t();
      arma::Mat<T> SKS;
      arma::Mat<T> KX = K_mat * X;
      arma::Mat<T> XKX = X.t() * KX;
      SKS = -KX * XXXi.t();
      SKS += SKS.t();
      SKS += XXXi * XKX * XXXi.t();
      SKS += K_mat;
      arma::eig_sym(eigenVal, U, SKS);
      if (q > 0) {
        eigenVal.shed_rows(0, q - 1);
        U.shed_cols(0, q - 1);
      }
      eigenVal.transform([](T val) { return val < static_cast<T>(1e-6) ? static_cast<T>(1e-6) : val; });
    } else {
      // 分支 B：样本远多于特征 (M < N)，执行降维特征分解
      // 构建 B = S * A，将特征投影到零空间
      arma::Mat<T> B = A - XXXi * (X.t() * A);
      arma::Mat<T> BtB = B.t() * B;
      arma::Col<T> eval_M;
      arma::Mat<T> V;
      arma::eig_sym(eval_M, V, BtB);
      arma::uvec valid_idx = arma::find(eval_M > static_cast<T>(1e-6));
      eigenVal = eval_M.elem(valid_idx);
      M_nz = eigenVal.n_elem;
      // U_M = B * V * Lambda^(-1/2)，还原真实行对应的特征向量 (大小 N x M_nz)
      U = B * V.cols(valid_idx) * arma::diagmat(1.0 / arma::sqrt(eigenVal));
      is_compact = true;
    }
  } else {
    throw std::runtime_error("Unsupported KType of grm in batch trait processing");
  }

  std::cout << "Maximizing Log-L by Grid Search...\n";
  arma::Col<T> LogL(np, arma::fill::zeros);
  arma::Col<T> h2(np, arma::fill::zeros);
  arma::Col<T> vg(np, arma::fill::zeros);
  arma::Col<T> ve(np, arma::fill::zeros);
  arma::Col<T> se_vg(np, arma::fill::zeros);
  arma::Col<T> se_ve(np, arma::fill::zeros);
  arma::Col<T> trPG(np, arma::fill::zeros);
  arma::Mat<T> Py(n, np, arma::fill::zeros);
  arma::Mat<T> Sy = y - XXXi * (X.t() * y);
  arma::Mat<T> Uy = U.t() * y;
  int omp_thread = ncpus;
  if (np == 1) omp_thread = 1;
  if (omp_thread > 1) rbp::blas_set_num_threads(1);
  arma::uword total = np;
  arma::uword task_unit = std::max((arma::uword)1, (total + 99) / 100);
  arma::uword n_step = (total + task_unit - 1) / task_unit;
  ProgressBar pb(n_step, 50);
  #pragma omp parallel for num_threads(omp_thread) schedule(static)
  for (arma::uword i = 0; i < np; ++i) {
    arma::Col<T> gamma_yi = Uy.col(i);
    arma::Col<T> gamma_sq = arma::square(gamma_yi);
    T best_logl = -std::numeric_limits<T>::max();
    T best_h2 = 0.0;
    T best_yPy = 0.0;
    const T* eig_ptr = eigenVal.memptr();
    const T* gsq_ptr = gamma_sq.memptr();
    // 为降维补偿做准备 (计算 U 之外正交空间的表型残余平方和)
    arma::uword limit = eigenVal.n_elem;
    arma::uword missing_dim = 0;
    T residual_sq = 0.0;
    if (is_compact) {
      missing_dim = (n - q) - M_nz;
      T total_y_sq = arma::as_scalar(Sy.col(i).t() * Sy.col(i));
      residual_sq = std::max(static_cast<T>(0), total_y_sq - arma::sum(gamma_sq));
    }
    auto compute_logl = [&](T h2_val, T& yPy_out) -> T {
      T h2_comp = 1.0 - h2_val;
      T log_det_V = 0.0;
      T yPy_val = 0.0;
      for (arma::uword k = 0; k < limit; ++k) {
        T e = h2_val * eig_ptr[k] + h2_comp;
        log_det_V += std::log(e);
        yPy_val += gsq_ptr[k] / e;
      }
      if (is_compact && missing_dim > 0) {
        log_det_V += missing_dim * std::log(h2_comp);
        yPy_val += residual_sq / h2_comp;
      }
      yPy_out = yPy_val;
      return constant - 0.5 * ((n - q) * std::log(yPy_val) + log_det_V);
    };
    int coarse_grid = 100; //粗略网格搜索，快速定位 h2 范围
    for (int step = 1; step < coarse_grid; ++step) {
      T h2 = static_cast<T>(step) / static_cast<T>(coarse_grid);
      T yPy_tmp = 0.0;
      T logl = compute_logl(h2, yPy_tmp);
      if (logl > best_logl) {
        best_logl = logl;
        best_h2 = h2;
        best_yPy = yPy_tmp;
      }
    }
    T window_radius = 0.1 * best_h2;
    T min_radius = 2.0 / static_cast<T>(coarse_grid); 
    window_radius = std::max(window_radius, min_radius);
    T h2_start = std::max(static_cast<T>(0.001), best_h2 - window_radius);
    T h2_end   = std::min(static_cast<T>(0.999), best_h2 + window_radius);
    int fine_grid = 16; // 细粒度网格搜索，精确定位 h2
    for (int step = 0; step <= fine_grid; ++step) {
      T h2 = h2_start + step * (h2_end - h2_start) / static_cast<T>(fine_grid);
      T yPy_tmp = 0.0;
      T logl = compute_logl(h2, yPy_tmp);
      if (logl > best_logl) {
        best_logl = logl;
        best_h2 = h2;
        best_yPy = yPy_tmp;
      }
    }
    h2[i] = best_h2;
    LogL[i] = best_logl;
    T sigma_y2 = best_yPy / static_cast<T>(n - q);
    vg[i] = sigma_y2 * best_h2;
    ve[i] = sigma_y2 * (1.0 - best_h2);

    arma::Col<T> best_eta = vg[i] * eigenVal + ve[i];
    arma::Col<T> gamma_yi_scaled = gamma_yi / best_eta;
    if (is_compact) {
      arma::Col<T> U_gamma = U * gamma_yi; // 在 U 空间内的预测分量
      arma::Col<T> res_y = Sy.col(i) - U_gamma; // 零空间分量
      Py.col(i) = U * gamma_yi_scaled + res_y / ve[i];
    } else {
      Py.col(i) = U * gamma_yi_scaled;
    }
    
    T Vg = vg[i];
    T Ve = ve[i];
    T I_gg = 0.0;
    T I_ge = 0.0;
    T I_ee = 0.0;
    T tr_PK = 0.0;
    for (arma::uword k = 0; k < limit; ++k) {
      T Vk = Vg * eig_ptr[k] + Ve; 
      T Vk_inv2 = 1.0 / (Vk * Vk);
      I_gg += eig_ptr[k] * eig_ptr[k] * Vk_inv2;
      I_ge += eig_ptr[k] * Vk_inv2;
      I_ee += Vk_inv2;
      tr_PK += eig_ptr[k] / Vk; 
    }
    trPG[i] = tr_PK;
    I_gg *= 0.5;
    I_ge *= 0.5;
    I_ee *= 0.5;
    // 补偿降维时被折叠的环境残差空间
    if (is_compact && missing_dim > 0) {
      I_ee += 0.5 * static_cast<T>(missing_dim) / (Ve * Ve);
    }
    T determinant = I_gg * I_ee - I_ge * I_ge;
    if (determinant > 1e-12) {
      T C11 = I_ee / determinant;  // Var(Vg)
      T C22 = I_gg / determinant;  // Var(Ve)
      // T C12 = -I_ge / determinant; // Cov(Vg, Ve) - 若不需要协方差可省略
      se_vg[i] = std::sqrt(std::max(static_cast<T>(0), C11));
      se_ve[i] = std::sqrt(std::max(static_cast<T>(0), C22));
    } else {
      se_vg[i] = arma::datum::nan;
      se_ve[i] = arma::datum::nan;
    }
        
    if ((i + 1) % task_unit == 0) {
      #pragma omp critical
      {
        ++pb;
        pb.display();
      }    
    }

  }
  ++pb; pb.done(); 
  rbp::blas_set_num_threads(threads_blas);
  
  arma::uvec non_finite = arma::find_nonfinite(h2 + vg + ve);
  if (non_finite.n_elem > 0) {
    throw std::runtime_error(
      "\n*** ERROR ***: " + std::to_string(non_finite.n_elem) + 
      " pheno(s) produced non-finite values (NaN or Inf).\n" +
      "    -> Action: Check if your phenotypes have missing values (NA), " +
      "or if some traits are completely constant (variance = 0).\n"
    );
  }
  if (np > 1) { 
    std::cout << "[Summary] h2 across traits - Min: " << std::fixed << std::setprecision(5) <<arma::min(h2) << " | Max: " << arma::max(h2) << "\n";
  }
  arma::uvec boundary_hits = arma::find(h2 <= static_cast<T>(0.001001) ||  h2 >= static_cast<T>(0.998999));
  if (boundary_hits.n_elem > 0) {
    std::cout << "*** Warning ***: " << boundary_hits.n_elem 
              << " pheno(s) hit the search boundaries (0.001 or 0.999).\n"
              << "    -> This implies the true peak might be exactly 0 or 1. Check data scale or model structure.\n";
  }

  batch_result res;
  res.vg = arma::conv_to<arma::vec>::from(vg);
  res.ve = arma::conv_to<arma::vec>::from(ve);
  res.se_vg = arma::conv_to<arma::vec>::from(se_vg);
  res.se_ve = arma::conv_to<arma::vec>::from(se_ve);

  std::cout << "Calculating random effects..." << std::endl;
  std::vector<arma::Mat<T>> rand_T = random_effect_cpp(Z_o, K_o, arma::Col<T>(arma::join_cols(vg, ve)), Py, arma::Mat<T>(), ncpus, false);
  for (const auto& r : rand_T) {
    res.rand_eff.push_back(arma::conv_to<arma::mat>::from(r));
  }

  if (do_test) {
    std::cout << "=====================[ GWAS Scanning ]=====================\n";
    if (test_method == "exact") {
      res.gwas_res = exact_gwas_scan_cpp(
        bed_file, ind_row, ind_col, Py, Z_geno, U, eigenVal, vg, ve, X, XXXi, 
        is_compact, Uy, Sy, 4096, ncpus
      );
    } else {
      arma::Col<T> gamma(np, arma::fill::value(-1.0));
      bool re_estimate_gamma = false;
      if (K.type != KType::KINSHIP) re_estimate_gamma = true;
      if (!re_estimate_gamma) {
        gamma = trPG / (ind_row.n_elem - 1.0);
      }

      res.gwas_res = batch_gwas_scan_cpp(
        bed_file, ind_row, ind_col, test_method, 
        Py, Z_geno, U, eigenVal, vg, ve, X, XXXi, 
        re_estimate_gamma, is_compact, re_gamma_size, 
        num_random, random_seed, gamma, 8, ncpus
      );
    }
  }

  return res;
}

template batch_result batch_model_cpp<double>(const arma::Mat<double>&, const arma::Mat<double>&, const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const std::string&, const arma::uvec&, const arma::uvec&, const std::string&, const arma::umat&, bool, int, int, int, int);
template batch_result batch_model_cpp<float>(const arma::Mat<float>&, const arma::Mat<float>&, const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const std::string&, const arma::uvec&, const arma::uvec&, const std::string&, const arma::umat&, bool, int, int, int, int);
} // namespace rbp